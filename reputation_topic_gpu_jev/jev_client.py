#!/usr/bin/env python3
"""Jev client: LayaClient's decide() contract, answered by TypeSafe's hosted Jev.

Drop-in for laya_client.LayaClient / gpt_decision_client.GptDecisionClient
(--decision-backend jev, the default in this copy). Jev is a "System One"
model: it takes a state and typed questions and returns typed answers with
probabilities, no generated text.

    POST {TYPESAFE_BASE_URL}/v1/systemone        (default https://api.typesafe.ai)
    header  Authorization: Bearer <TYPESAFE_API_KEY>
    body    {"model": "jev-latest", "state": "<post>",
             "questions": {"topic": {"type": "choice", "instructions": ...,
                                     "criteria": {"<key>": "<description>", ...}}}}
    answer  {"model": "jev-1.x", "answers": {"topic": {"type": "choice", "choice",
             "confidence", "probabilities": {key: p}}}, "usage": {...}}

What differs from laya-serve, measured or documented (Oct 2026):
  * No batch endpoint: /v1/systemone/batch returns 404 on the hosted API, so
    every post is its own request. Requests run concurrently under a
    client-side limit below Jev's 1,200 requests/minute.
  * Up to 255 options per choice question (laya-serve: 100).
  * The answer carries `confidence` and `probabilities`; where
    `answer_confidence` is present it is used, otherwise P(choice).
  * Errors: 401/403 bad or missing key, 422 bad request, 429 rate limited,
    529 overloaded. 429/529/5xx are retried with backoff.
Pricing at launch: $0.042 per million input tokens, output free.

Settings come from the environment, falling back to the .env beside this file:

    TYPESAFE_API_KEY   required
    TYPESAFE_BASE_URL  default https://api.typesafe.ai
    JEV_MODEL          default jev-latest
    JEV_TIMEOUT        seconds per request (default 30)
    JEV_MAX_RPM        client-side request rate cap (default 1000)
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

from laya_client import NO_ISSUE_KEY, NO_ISSUE_TEXT, OTHER_KEY, OTHER_TEXT

DEFAULT_BASE = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
MAX_OPTIONS = 255                 # Jev's documented cap per choice question
MAX_ATTEMPTS = 5
RETRY_BASE_DELAY = 1.0
MAX_CONSECUTIVE_FAILURES = 20     # concurrent requests fail together; allow a burst

# One decision: (chosen key, confidence, {key: probability}) -- as laya_client.
Decision = Tuple[str, float, Dict[str, float]]


class JevConfigError(RuntimeError):
    """No key configured, or Jev does not answer: stop before embedding."""


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


class _RateLimiter:
    """At most `per_minute` request starts in any rolling minute, across threads."""

    def __init__(self, per_minute: float) -> None:
        self.interval = 60.0 / max(1.0, per_minute)
        self._next = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


class JevClient:
    def __init__(self, model: Optional[str] = None, max_rpm: Optional[float] = None) -> None:
        file_env = _dotenv()

        def get(name: str, default: str = "") -> str:
            return os.environ.get(name) or file_env.get(name) or default

        self.api_key = get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevConfigError("missing env: TYPESAFE_API_KEY (set it in .env or the environment)")
        self.base = get("TYPESAFE_BASE_URL", DEFAULT_BASE).rstrip("/")
        self.model = model or get("JEV_MODEL", DEFAULT_MODEL)
        self.timeout = float(get("JEV_TIMEOUT", "30"))
        self.max_rpm = float(max_rpm or get("JEV_MAX_RPM", "1000"))
        self._limiter = _RateLimiter(self.max_rpm)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "failures": 0, "http_retries": 0, "rate_limited": 0,
                      "invalid_choices": 0, "input_tokens": 0, "latency_s": []}
        self.served_model: Optional[str] = None
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
        return {"backend": "typesafe-jev", "base_url": self.base, "model": self.model,
                "served_model": self.served_model, "max_rpm": self.max_rpm}

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

    def _post(self, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last: Optional[Exception] = None
        for attempt in range(MAX_ATTEMPTS):
            self._limiter.wait()
            try:
                req = urllib.request.Request(f"{self.base}/v1/systemone", data=data,
                                             headers=headers, method="POST")
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode())
                self._bump(latency_s=time.perf_counter() - t0)
                return out
            except urllib.error.HTTPError as exc:
                # 401/403 (key), 422 (bad request) will not improve on retry.
                if exc.code < 500 and exc.code != 429:
                    raise
                if exc.code == 429:
                    self._bump(rate_limited=1)
                last = exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = (float(retry_after) if retry_after and retry_after.isdigit()
                         else RETRY_BASE_DELAY * (2 ** attempt))
            except Exception as exc:                      # network, timeout
                last = exc
                delay = RETRY_BASE_DELAY * (2 ** attempt)
            self._bump(http_retries=1)
            time.sleep(delay)
        raise last  # type: ignore[misc]

    @staticmethod
    def _parse(out: dict, allowed: set) -> Optional[Decision]:
        ans = ((out or {}).get("answers") or {}).get("topic")
        if not isinstance(ans, dict) or ans.get("choice") is None:
            return None
        choice = str(ans["choice"])
        if choice not in allowed:
            return None
        probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
        conf = ans.get("answer_confidence")
        if conf is None:
            conf = probs.get(choice, ans.get("confidence") or 0.0)
        return choice, float(conf), probs

    def _one(self, instructions: str, criteria: Dict[str, str], text: str) -> Optional[Decision]:
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._bump(failures=1)
            return None
        body = {"model": self.model, "state": str(text)[:4000],
                "questions": {"topic": {"type": "choice", "instructions": instructions,
                                        "criteria": criteria}}}
        self._bump(calls=1)
        try:
            out = self._post(body)
        except Exception as exc:
            self._bump(failures=1)
            with self._lock:
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                self._consecutive_failures += 1
            return None
        with self._lock:
            self._consecutive_failures = 0
            self.served_model = self.served_model or (out or {}).get("model")
        self._bump(input_tokens=int(((out or {}).get("usage") or {}).get("input_tokens") or 0))
        res = self._parse(out, set(criteria))
        if res is None:
            self._bump(invalid_choices=1)
        return res

    def health(self) -> None:
        """One tiny real decision, so a bad key fails before the embedding stage."""
        res = self._one("Which topic does this customer post belong to?",
                        {"T1": "Late parcel delivery: orders arriving days late",
                         OTHER_KEY: OTHER_TEXT, NO_ISSUE_KEY: NO_ISSUE_TEXT},
                        "my parcel is three days late")
        if res is None:
            raise JevConfigError(f"Jev at {self.base} did not answer: {self.last_error}")

    def decide(self, requests: Sequence[Tuple[str, Dict[str, str], str]],
               workers: int = 16) -> List[Optional[Decision]]:
        """Decide (instructions, options, text) requests, in order; None where Jev failed."""
        for _instr, crit, _text in requests:
            if len(crit) > MAX_OPTIONS:
                raise ValueError(f"{len(crit)} options exceeds Jev's cap of {MAX_OPTIONS}")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(lambda r: self._one(*r), requests))


if __name__ == "__main__":
    try:
        c = JevClient()
        c.health()
    except JevConfigError as exc:
        raise SystemExit(f"not configured: {exc}")
    crit = {"T1": "iOS 11 battery drain: battery empties within hours since the update",
            "T2": "Apple Pay card not working: cards rejected at checkout",
            OTHER_KEY: OTHER_TEXT, NO_ISSUE_KEY: NO_ISSUE_TEXT}
    instr = "Which topic does this customer post to AppleSupport belong to? Answer 'other' if none fits."
    texts = ["my battery dies in 2 hours since iOS 11", "thanks, sent you a DM",
             "where is my parcel", "apple pay keeps declining my card and battery is dying too"]
    for text, res in zip(texts, c.decide([(instr, crit, t) for t in texts])):
        print(f"{text!r:62} -> {res}")
    print(json.dumps({**c.config(), **c.summary()}, indent=1))
