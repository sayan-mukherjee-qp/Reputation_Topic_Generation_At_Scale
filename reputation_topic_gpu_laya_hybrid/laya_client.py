#!/usr/bin/env python3
"""Laya client for matching incoming records to existing topics.

Laya (github.com/NandhaKishorM/laya) is an open-weight decision model served
by `laya-serve` on the same wire protocol as TypeSafe's Jev:

    POST {base_url}/v1/systemone/batch
    header  Authorization: Bearer <key>        (only when the server sets LAYA_API_KEY)
    body    {"states": [{"post": ...}, ...],               at most 64 per call
             "questions": {"topic": {"type": "choice", "instructions": ...,
                                     "criteria": {"<key>": "<description>", ...}}},
             "model"?, "min_confidence"?, "max_len"?, "head_max_len"?}
    answer  {"results": [{"answers": {"topic": {"choice", "probabilities",
                                                 "confidence", "answer_confidence"}}}, ...],
             "total_usage": {...}}

Every state in one call shares one question, so records are grouped by their
option list (see `decide`). Two limits shape the requests:

  * Options share a token budget (`head_max_len`, 192-256 tokens), so accuracy
    falls past ~20 options; the server refuses more than 100. Callers send a
    shortlist of the nearest topics, not the brand's whole registry.
  * `confidence` is entropy-based and shifts with the option count; the
    calibrated number is `answer_confidence`, which is what callers gate on.

Settings are read from the environment, falling back to the .env beside this
file:

    LAYA_BASE_URL     required, e.g. http://127.0.0.1:8000
    LAYA_API_KEY      bearer token, only if the server was started with one
    LAYA_MODEL        english | multilingual | typed-decisions (default: server routes per state)
    LAYA_TIMEOUT      seconds per request (default 60)
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import urllib.error
import urllib.request

MAX_STATES_PER_CALL = 64          # laya-serve's cap on /v1/systemone/batch
MAX_OPTIONS = 100                 # laya-serve refuses more with 413
MAX_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.0
# After this many failed calls in a row the server is treated as down: the
# remaining calls fail at once instead of each sitting through its retries.
MAX_CONSECUTIVE_FAILURES = 5

# Option keys are rendered verbatim and Laya can follow a key's own words over
# its description, so the two exits get plain semantic keys and topics get
# their opaque IDs.
OTHER_KEY = "other"
NO_ISSUE_KEY = "no_issue"
OTHER_TEXT = "none of these topics; a different subject"
NO_ISSUE_TEXT = "thanks, greetings, DM sent or small talk with no issue"

# One decision: (chosen key, calibrated P(chosen), {key: probability}).
Decision = Tuple[str, float, Dict[str, float]]


class LayaConfigError(RuntimeError):
    """Raised when no endpoint is configured, so the run stops before it starts."""


def _dotenv() -> Dict[str, str]:
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


class LayaClient:
    def __init__(self, min_confidence: Optional[float] = None,
                 max_len: Optional[int] = None, head_max_len: Optional[int] = None) -> None:
        file_env = _dotenv()

        def get(name: str, default: str = "") -> str:
            return os.environ.get(name) or file_env.get(name) or default

        self.base = get("LAYA_BASE_URL").rstrip("/")
        if not self.base:
            raise LayaConfigError("missing env: LAYA_BASE_URL (set it in .env or the environment)")
        self.api_key = get("LAYA_API_KEY")
        self.model = get("LAYA_MODEL") or None
        self.timeout = float(get("LAYA_TIMEOUT", "60"))
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.head_max_len = head_max_len
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "states": 0, "failures": 0, "failed_states": 0,
                      "http_retries": 0, "input_tokens": 0, "latency_s": []}
        self.last_error: Optional[str] = None
        self._consecutive_failures = 0

    def _bump(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                if k == "latency_s":
                    self.stats[k].append(v)
                else:
                    self.stats[k] += v

    def config(self) -> dict:
        """What run_metadata.json records about the matching layer."""
        return {"backend": "laya-serve", "base_url": self.base, "model": self.model or "router",
                "max_len": self.max_len, "head_max_len": self.head_max_len}

    def summary(self) -> dict:
        with self._lock:
            st = dict(self.stats)
            lat = sorted(st.pop("latency_s"))
        if self.last_error:
            st["last_error"] = self.last_error
        if lat:
            st["latency_mean_s"] = round(sum(lat) / len(lat), 3)
            st["latency_p95_s"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3)
        return st

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def health(self) -> None:
        """Fail fast on an unreachable endpoint rather than after embedding."""
        req = urllib.request.Request(f"{self.base}/health", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 15)) as r:
                r.read()
        except Exception as exc:
            raise LayaConfigError(f"Laya endpoint {self.base} is not reachable: "
                                  f"{type(exc).__name__}: {exc}") from exc

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode()
        last: Optional[Exception] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                req = urllib.request.Request(f"{self.base}{path}", data=data,
                                             headers=self._headers(), method="POST")
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode())
                self._bump(latency_s=time.perf_counter() - t0)
                return out
            except urllib.error.HTTPError as exc:
                # 413 (too many options) and 422 (bad schema) will not improve on retry.
                if exc.code < 500 and exc.code != 429:
                    raise
                last = exc
            except Exception as exc:                      # network, timeout
                last = exc
            self._bump(http_retries=1)
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        raise last  # type: ignore[misc]

    @staticmethod
    def _parse(result: dict) -> Optional[Decision]:
        ans = ((result or {}).get("answers") or {}).get("topic")
        if not isinstance(ans, dict) or ans.get("choice") is None:
            return None
        probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
        choice = str(ans["choice"])
        conf = ans.get("answer_confidence")
        if conf is None:                                  # strict-Jev servers omit it
            conf = probs.get(choice, ans.get("confidence") or 0.0)
        return choice, float(conf), probs

    def _call(self, instructions: str, criteria: Dict[str, str],
              states: Sequence[str]) -> List[Optional[Decision]]:
        body: Dict[str, object] = {
            "states": [{"post": s} for s in states],
            "questions": {"topic": {"type": "choice", "instructions": instructions,
                                    "criteria": criteria}},
        }
        for k, v in (("model", self.model), ("min_confidence", self.min_confidence),
                     ("max_len", self.max_len), ("head_max_len", self.head_max_len)):
            if v is not None:
                body[k] = v
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._bump(failed_states=len(states))
            return [None] * len(states)
        self._bump(calls=1, states=len(states))
        try:
            out = self._post("/v1/systemone/batch", body)
        except Exception as exc:
            self._bump(failures=1, failed_states=len(states))
            with self._lock:
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                self._consecutive_failures += 1
            return [None] * len(states)
        with self._lock:
            self._consecutive_failures = 0
        self._bump(input_tokens=int(((out or {}).get("total_usage") or {}).get("input_tokens") or 0))
        results = (out or {}).get("results") or []
        parsed = [self._parse(r) for r in results[:len(states)]]
        parsed += [None] * (len(states) - len(parsed))
        n_bad = sum(1 for p in parsed if p is None)
        if n_bad:
            self._bump(failed_states=n_bad)
        return parsed

    def decide(self, requests: Sequence[Tuple[str, Dict[str, str], str]],
               workers: int = 4) -> List[Optional[Decision]]:
        """Decide (instructions, criteria, text) requests, in order; None where Laya failed.

        Requests with the same instructions and criteria share one question, so
        they ride one /batch call together (up to 64 states). Callers order each
        shortlist canonically so neighbouring records produce identical option
        lists and actually share calls.
        """
        groups: Dict[str, List[int]] = {}
        for i, (instr, crit, _) in enumerate(requests):
            if len(crit) > MAX_OPTIONS:
                raise ValueError(f"{len(crit)} options exceeds laya-serve's cap of {MAX_OPTIONS}")
            groups.setdefault(json.dumps([instr, crit], sort_keys=True), []).append(i)
        jobs = []
        for idx in groups.values():
            instr, crit, _ = requests[idx[0]]
            for lo in range(0, len(idx), MAX_STATES_PER_CALL):
                part = idx[lo:lo + MAX_STATES_PER_CALL]
                jobs.append((part, instr, crit, [requests[i][2] for i in part]))
        out: List[Optional[Decision]] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for (part, *_), res in zip(jobs, pool.map(lambda j: self._call(*j[1:]), jobs)):
                for i, r in zip(part, res):
                    out[i] = r
        return out


if __name__ == "__main__":
    try:
        c = LayaClient()
        c.health()
    except LayaConfigError as exc:
        raise SystemExit(f"not configured: {exc}")
    crit = {"T1": "iOS 11 battery drain after the update",
            "T2": "Apple Pay card not working",
            OTHER_KEY: OTHER_TEXT, NO_ISSUE_KEY: NO_ISSUE_TEXT}
    instr = "Which topic does this customer post to AppleSupport belong to?"
    texts = ["my battery dies in 2 hours since iOS 11", "thanks, sent you a DM",
             "where is my parcel"]
    for text, res in zip(texts, c.decide([(instr, crit, t) for t in texts])):
        print(f"{text!r:45} -> {res}")
    print(c.summary())
