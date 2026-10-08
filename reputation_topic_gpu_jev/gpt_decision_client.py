#!/usr/bin/env python3
"""GPT decision client: LayaClient's decide() contract, answered by GPT via AI Router v1.

Drop-in for laya_client.LayaClient (--decision-backend gpt, the default in
this copy). Each request is (instructions, options, post) and each answer is
(chosen key, confidence, {key: probability}), exactly as Laya returns, so the
matcher's gates (--laya-min-confidence, --laya-secondary-prob) and fallbacks
apply unchanged.

What differs from Laya is how posts share a call. Laya batches only posts
whose option lists are identical, which nearest-topic shortlists rarely are.
GPT reads per-post option lists, so here ~20 posts of one brand ride one call,
each keeping its own shortlist, with the union of their options sent once.
Posts are ordered by shortlist before chunking so neighbours overlap and the
union stays small.

Confidence is GPT's own 0-1 estimate, not a calibrated probability like
Laya's answer_confidence. Treat the 0.50 gate as a starting point to re-check.

Transport, credentials and retries come from airouter_v1_client (api_key,
base_url, use_case in .env; AIROUTER_MODEL picks the model, default
gpt-4.1-mini-chat-completion).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from airouter_v1_client import AiRouterConfigError, AiRouterV1Client
from laya_client import NO_ISSUE_KEY, NO_ISSUE_TEXT, OTHER_KEY, OTHER_TEXT

# One decision: (chosen key, confidence, {key: probability}) -- as laya_client.
Decision = Tuple[str, float, Dict[str, float]]

PROMPT_VERSION = "topic-decision-v1"

DECISION_PROMPT = """\
You file customer posts into an existing topic registry for a brand reputation
monitoring system.

The input is a JSON object with:
  - brand: the brand every post was sent to
  - question: what to decide
  - options: topic ID -> "label: what customers in that topic report"
  - exits: "other" and "no_issue", with what each means
  - posts: a list of {id, text, options}; each post's "options" lists the topic
    IDs offered for THAT post

For EACH post choose exactly one key:
  - one of that post's own option IDs, when the post is about that topic's
    specific issue, or
  - "other" when the post reports a real issue that none of its options covers, or
  - "no_issue" when the post is thanks, a greeting, "DM sent", small talk or
    otherwise reports no issue.

RULES

1. Only choose a topic ID listed in that post's own "options". Never choose a
   topic offered only to another post.
2. Sharing a broad area is not enough. "My flight was delayed" does not belong
   to a topic about lost baggage; choose "other" instead.
3. Judge what the post says. Posts may be in any language.
4. "confidence" is your probability, from 0 to 1, that your choice is right.
5. If a SECOND offered topic also clearly applies (the post raises two issues),
   give it as "second" with its own "second_confidence"; otherwise null.
6. Return one answer per post, with the post's id, in any order.

OUTPUT

Return only this JSON object, nothing else:

{"answers": [{"id": "<post id>", "choice": "<topic id | other | no_issue>",
              "confidence": <0.0-1.0>, "second": "<topic id>" | null,
              "second_confidence": <0.0-1.0> | null}]}
"""

POSTS_PER_CALL = 20
MAX_OPTIONS_PER_CALL = 40
TOKENS_PER_ANSWER = 45            # output budget per post, with room to spare
MAX_CONSECUTIVE_FAILURES = 5


class GptDecisionConfigError(RuntimeError):
    """No router configured, or the router does not answer: stop before embedding."""


class GptDecisionClient:
    def __init__(self, posts_per_call: int = POSTS_PER_CALL,
                 max_options: int = MAX_OPTIONS_PER_CALL,
                 model: Optional[str] = None, use_case: Optional[str] = None) -> None:
        try:
            self.router = AiRouterV1Client(model=model, use_case=use_case)
        except AiRouterConfigError as exc:
            raise GptDecisionConfigError(str(exc)) from exc
        self.posts_per_call = max(1, int(posts_per_call))
        self.max_options = max(4, int(max_options))
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "posts": 0, "failed_calls": 0, "failed_posts": 0,
                      "invalid_choices": 0, "options_sent": 0}
        self.last_error: Optional[str] = None
        self._consecutive_failures = 0

    def _bump(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                self.stats[k] += v

    def config(self) -> dict:
        r = self.router.config()
        return {"backend": "gpt-airouter-v1", "base_url": r["base_url"], "model": r["model"],
                "use_case": r["use_case"], "prompt_version": PROMPT_VERSION,
                "posts_per_call": self.posts_per_call, "max_options_per_call": self.max_options}

    def summary(self) -> dict:
        with self._lock:
            st = dict(self.stats)
        rs = self.router.summary()
        st.update({k: rs[k] for k in ("prompt_tokens", "completion_tokens", "http_retries",
                                      "latency_mean_s", "latency_p95_s") if k in rs})
        st["mean_options_per_call"] = (round(st["options_sent"] / st["calls"], 1)
                                       if st["calls"] else 0.0)
        if self.last_error:
            st["last_error"] = self.last_error
        return st

    def health(self) -> None:
        """One tiny real decision, so a bad key or URL fails before the embedding stage."""
        res = self.decide([("Which topic does this customer post belong to?",
                            {"T1": "Late parcel delivery: orders arriving days late",
                             OTHER_KEY: OTHER_TEXT, NO_ISSUE_KEY: NO_ISSUE_TEXT},
                            "my parcel is three days late")], workers=1)
        if res[0] is None:
            raise GptDecisionConfigError(f"GPT decision endpoint did not answer: {self.last_error}")

    # ---- batching ---------------------------------------------------------
    def _chunks(self, requests) -> List[Tuple[str, List[int]]]:
        """Group by question (one brand), order by shortlist, cut by posts and union size."""
        groups: Dict[str, List[int]] = {}
        for i, (instr, _crit, _text) in enumerate(requests):
            groups.setdefault(instr, []).append(i)
        out = []
        for instr, idx in groups.items():
            topic_keys = lambda i: tuple(sorted(k for k in requests[i][1]
                                                if k not in (OTHER_KEY, NO_ISSUE_KEY)))
            idx = sorted(idx, key=topic_keys)
            cur, union = [], set()
            for i in idx:
                keys = set(topic_keys(i))
                if cur and (len(cur) >= self.posts_per_call or len(union | keys) > self.max_options):
                    out.append((instr, cur))
                    cur, union = [], set()
                cur.append(i)
                union |= keys
            if cur:
                out.append((instr, cur))
        return out

    @staticmethod
    def _answers(body: dict) -> Optional[list]:
        for doc in ((body or {}).get("result") or {}).get("documents") or []:
            for item in doc.get("output") or []:
                if not isinstance(item, dict) or item.get("key") == "reasoning":
                    continue
                value = item.get("value")
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                if isinstance(value, dict) and isinstance(value.get("answers"), list):
                    return value["answers"]
        return None

    @staticmethod
    def _prob(x) -> Optional[float]:
        try:
            return min(1.0, max(0.0, float(x)))
        except (TypeError, ValueError):
            return None

    def _call(self, instr: str, idx: List[int], requests) -> List[Optional[Decision]]:
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._bump(failed_posts=len(idx))
            return [None] * len(idx)
        options: Dict[str, str] = {}
        posts, allowed = [], {}
        for n, i in enumerate(idx):
            _instr, crit, text = requests[i]
            keys = [k for k in crit if k not in (OTHER_KEY, NO_ISSUE_KEY)]
            for k in keys:
                options[k] = crit[k]
            pid = f"p{n + 1}"
            posts.append({"id": pid, "text": str(text)[:1000], "options": keys})
            allowed[pid] = set(keys) | {OTHER_KEY, NO_ISSUE_KEY}
        brand = instr.split(" to ", 1)[1].split(" belong", 1)[0] if " to " in instr else None
        payload = {"brand": brand, "question": instr, "options": options,
                   "exits": {OTHER_KEY: OTHER_TEXT, NO_ISSUE_KEY: NO_ISSUE_TEXT}, "posts": posts}
        self._bump(calls=1, posts=len(idx), options_sent=len(options))
        try:
            body = self.router._run(DECISION_PROMPT, json.dumps(payload, ensure_ascii=False),
                                    max_tokens=60 + TOKENS_PER_ANSWER * len(idx))
            answers = self._answers(body)
            if answers is None:
                raise ValueError("no 'answers' list in the response")
        except Exception as exc:
            self._bump(failed_calls=1, failed_posts=len(idx))
            with self._lock:
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                self._consecutive_failures += 1
            return [None] * len(idx)
        with self._lock:
            self._consecutive_failures = 0
        by_id = {str(a.get("id")): a for a in answers if isinstance(a, dict)}
        out: List[Optional[Decision]] = []
        for p in posts:
            a = by_id.get(p["id"])
            choice = str((a or {}).get("choice") or "")
            conf = self._prob((a or {}).get("confidence"))
            if not a or choice not in allowed[p["id"]] or conf is None:
                self._bump(failed_posts=1, invalid_choices=int(bool(a)))
                out.append(None)
                continue
            probs = {choice: conf}
            second = str(a.get("second") or "")
            sconf = self._prob(a.get("second_confidence"))
            if second and second != choice and second in allowed[p["id"]] \
                    and second not in (OTHER_KEY, NO_ISSUE_KEY) and sconf is not None:
                probs[second] = sconf
            out.append((choice, conf, probs))
        return out

    def decide(self, requests: Sequence[Tuple[str, Dict[str, str], str]],
               workers: int = 8) -> List[Optional[Decision]]:
        """Decide (instructions, options, text) requests, in order; None where GPT failed."""
        jobs = self._chunks(requests)
        out: List[Optional[Decision]] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for (instr, idx), res in zip(jobs, pool.map(lambda j: self._call(j[0], j[1], requests),
                                                         jobs)):
                for i, r in zip(idx, res):
                    out[i] = r
        return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        c = GptDecisionClient()
        c.health()
    except GptDecisionConfigError as exc:
        raise SystemExit(f"not configured: {exc}")
    crit = {"T1": "iOS 11 battery drain: battery empties within hours since the update",
            "T2": "Apple Pay card not working: cards rejected at checkout",
            OTHER_KEY: OTHER_TEXT, NO_ISSUE_KEY: NO_ISSUE_TEXT}
    instr = "Which topic does this customer post to AppleSupport belong to? Answer 'other' if none fits."
    texts = ["my battery dies in 2 hours since iOS 11", "thanks, sent you a DM",
             "where is my parcel", "apple pay keeps declining my card and battery is dying too"]
    for text, res in zip(texts, c.decide([(instr, crit, t) for t in texts])):
        print(f"{text!r:62} -> {res}")
    print(json.dumps(c.summary(), indent=1), f"{time.perf_counter() - t0:.1f}s")
