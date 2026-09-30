#!/usr/bin/env python3
"""AI Router v1 client for topic labelling, with the prompt sent inline.

Unlike AiRouterV2 (air2_client.py), v1 accepts the prompt on the request, so
nothing has to be provisioned in the console beyond the use-case and API key.

    POST {base_url}/v1/prompt-routes
    header  api-key: <key>
    body    {"use_case_name", "user_id", "organization_id", "data_center",
             "model", "prompt_template", "params",
             "input_data": {"messages": [{"key": "content", "value": ...}],
                            "response_format": {"type": "json_object"}}}

`prompt_template` becomes the system prompt and the `content` message the user
turn. `model` must carry an endpoint suffix, e.g. `gpt-4.1-mini-chat-completion`.
The answer comes back at result.documents[0].output[] as
{"key": "content", "value_type": "json_object", "value": {...}}.

Settings are read from the environment, falling back to the .env beside this
file (whose keys are lower-case):

    api_key / AIROUTER_API_KEY     required
    base_url / AIROUTER_BASE_URL   default https://airouter-api.questionpro.com
    use_case / AIROUTER_USE_CASE   default reputation-topic-generation
    AIROUTER_MODEL                 default gpt-4.1-mini-chat-completion
    AIROUTER_USER_ID, AIROUTER_ORG_ID  default 0 (not validated by the router)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import urllib.error
import urllib.request

DEFAULT_BASE = "https://airouter-api.questionpro.com"
DEFAULT_USE_CASE = "reputation-topic-generation"
DEFAULT_MODEL = "gpt-4.1-mini-chat-completion"
REQUEST_TIMEOUT = 90
MAX_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.0

# Taken from AI_ROUTER_TOPIC_LABEL_PROMPT.md. Avoid "{{" here: the router
# substitutes {{name}} placeholders from the message keys.
#
# v2 (out_s300_6_llm review, P3): the model is now shown the cluster's measured
# coherence and residual flag, asked to abstain when the samples do not share
# a subject, and asked for a confidence score. v1 labelled 28 junk-flagged,
# anti-coherent clusters with confident, plausible, invented names
# ("Excessive packaging complaints" over members about wrong items, Prime
# cancellations and cashback) -- a label that sounds right is not evidence the
# topic is real, so the prompt now gives the model the evidence that it is not.
PROMPT_VERSION = "topic-label-v2"

TOPIC_LABEL_PROMPT = """\
You name customer-feedback topics for a brand reputation monitoring system.

You receive one topic discovered by clustering customer tweets about a brand.
The input is a JSON object with:
  - brand: the brand the tweets were sent to
  - terms: statistically distinctive terms for the cluster
  - samples: a RANDOM sample of real tweets from the cluster (not the best ones)
  - member_count: how many tweets the cluster holds
  - coherence: measured word co-occurrence coherence of the cluster, from -1 to 1.
    Above 0.05 the members usually share vocabulary; below 0 they usually do not.
    Short or multilingual tweets can score low while still sharing a subject.
  - residual_cluster: true when the pipeline already suspects the cluster is a
    grab-bag of chatter or unrelated leftovers

Return a short, specific name for the topic.

RULES

1. Name the ISSUE, not the brand and not the sentiment.
   Good: "Late parcel delivery"
   Bad:  "Amazon complaints", "Angry customers", "Delivery feedback"

2. Two to six words. A noun phrase in sentence case. No trailing punctuation.

3. Always answer in English, even when the terms or tweets are in another
   language. Topics frequently mix languages; the label must still be English.

4. Describe only what the tweets actually show. Never invent a cause, a product
   name, a version number, or a location that does not appear in the input.
   If the tweets say a feature broke but not why, name the broken feature.

5. Be specific enough to act on. If the tweets identify a particular product,
   feature, route, version or fee, put it in the name.
   Prefer "iOS 11 battery drain" over "Software problems".
   Prefer "Euston train delays" over "Train service issues".

6. The terms and the tweets can disagree, because terms are statistical and
   tweets are real. When they conflict, TRUST THE TWEETS.

7. If the tweets are mostly social pleasantries with no reportable issue
   (thanks, greetings, acknowledgements, "DM sent", yes/no replies), name it
   plainly as such, for example "Customer thank-you replies" or
   "Support DM acknowledgements", and set "is_substantive" to false.

8. ABSTAIN when there is no shared subject. First decide: do most of the
   samples (at least 6 in 10) describe the SAME specific issue? Generic
   customer-service complaints about different things do NOT count as the
   same issue -- "wrong item", "Prime cancelled" and "cashback dispute" are
   three subjects, not one. If there is no majority subject, return
   "Unclear topic" and set "is_substantive" to false. A plausible
   customer-service label for a mixed bag is WRONG, not helpful: an honest
   "Unclear topic" is more useful than a confident invented name.

9. Treat low coherence or residual_cluster = true as a warning, not a verdict.
   When either is present, require clear evidence in the samples before
   naming a specific issue; otherwise abstain as in rule 8.

10. Do not mention the brand name in the label. The brand is already recorded
    separately, so repeating it wastes the label.

11. Also write a "description": one or two plain sentences (at most 40 words)
    saying what customers in this topic are reporting or asking about, and
    the concrete specifics the tweets show (product, feature, route, fee,
    symptom). Same honesty rules as the label: English only, nothing that is
    not in the tweets, no brand name needed. For pleasantries or unclear
    topics, say so plainly.

12. Give "confidence": a number from 0 to 1 for how sure you are that the
    label describes MOST of the samples. Use below 0.5 when the samples are
    split between subjects or you had to generalise to find a common name.

OUTPUT

Return only this JSON object, nothing else:

{"label": "<2-6 word topic name>",
 "description": "<1-2 sentence summary>",
 "is_substantive": <true|false>,
 "confidence": <0.0-1.0>}
"""

TEMPERATURE = 0
MAX_TOKENS = 220

# One labelling result: label, description, is_substantive, confidence.
LabelResult = Tuple[str, Optional[str], Optional[bool], Optional[float]]


class AiRouterConfigError(RuntimeError):
    """Raised when the router is not configured, so callers can fall back."""


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


class AiRouterV1Client:
    def __init__(self, model: Optional[str] = None, use_case: Optional[str] = None) -> None:
        file_env = _dotenv()

        def get(*names: str, default: str = "") -> str:
            for n in names:
                v = os.environ.get(n) or file_env.get(n)
                if v:
                    return v
            return default

        self.api_key = get("AIROUTER_API_KEY", "api_key")
        if not self.api_key:
            raise AiRouterConfigError("missing env: AIROUTER_API_KEY (or api_key in .env)")
        self.base = get("AIROUTER_BASE_URL", "base_url", default=DEFAULT_BASE).rstrip("/")
        self.use_case = use_case or get("AIROUTER_USE_CASE", "use_case", default=DEFAULT_USE_CASE)
        self.model = model or get("AIROUTER_MODEL", default=DEFAULT_MODEL)
        self.user_id = int(get("AIROUTER_USER_ID", default="0"))
        self.org_id = int(get("AIROUTER_ORG_ID", default="0"))
        # Observability (P2): every call is counted, so a silent labelling
        # failure shows up in run_metadata.json instead of as a quiet fallback.
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "ok": 0, "failures": 0, "http_retries": 0,
                      "parse_failures": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "latency_s": []}

    def _bump(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                if k == "latency_s":
                    self.stats[k].append(v)
                else:
                    self.stats[k] += v

    def config(self) -> dict:
        """What run_metadata.json records about the labelling layer."""
        return {"backend": "airouter-v1", "base_url": self.base, "use_case": self.use_case,
                "model": self.model, "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(TOPIC_LABEL_PROMPT.encode()).hexdigest()[:16],
                "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}

    def summary(self) -> dict:
        with self._lock:
            st = dict(self.stats)
            lat = sorted(st.pop("latency_s"))
        if lat:
            st["latency_mean_s"] = round(sum(lat) / len(lat), 3)
            st["latency_p95_s"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3)
        return st

    def _run(self, prompt: str, content: str) -> dict:
        payload = json.dumps({
            "use_case_name": self.use_case,
            "user_id": self.user_id,
            "organization_id": self.org_id,
            "data_center": "US",
            "model": self.model,
            "prompt_template": prompt,
            "params": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
            "input_data": {
                "messages": [{"key": "content", "value": content}],
                "response_format": {"type": "json_object"},
            },
        }).encode()
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        last: Optional[Exception] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                req = urllib.request.Request(f"{self.base}/v1/prompt-routes",
                                             data=payload, headers=headers, method="POST")
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                    body = json.loads(r.read().decode())
                self._bump(latency_s=time.perf_counter() - t0)
                for doc in ((body or {}).get("result") or {}).get("documents") or []:
                    u = doc.get("usage") or {}
                    self._bump(prompt_tokens=int(u.get("prompt_tokens") or 0),
                               completion_tokens=int(u.get("completion_tokens") or 0))
                return body
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                last = exc
            except Exception as exc:                      # network, timeout
                last = exc
            self._bump(http_retries=1)
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        raise last  # type: ignore[misc]

    @staticmethod
    def _parse(body: dict) -> Optional[dict]:
        docs = ((body or {}).get("result") or {}).get("documents") or []
        for doc in docs:
            for item in doc.get("output") or []:
                if not isinstance(item, dict) or item.get("key") == "reasoning":
                    continue
                value = item.get("value")
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        return {"label": value.strip()} if value.strip() else None
                if isinstance(value, dict) and value.get("label"):
                    return value
        return None

    def label_topic(self, brand: str, terms: Sequence[str], samples: Sequence[str],
                    context: Optional[dict] = None) -> Optional[LabelResult]:
        """Return (label, description, is_substantive, confidence), or None on failure.

        `context` carries the cluster evidence the prompt asks for (member_count,
        coherence, residual_cluster). Never raises, so a labelling failure
        degrades to the c-TF-IDF label instead of failing the pipeline.
        """
        payload = {
            "brand": brand,
            "terms": list(terms)[:10],
            "samples": [str(s)[:280] for s in samples][:12],
        }
        payload.update({k: v for k, v in (context or {}).items() if v is not None})
        content = json.dumps(payload, ensure_ascii=False)
        self._bump(calls=1)
        try:
            parsed = self._parse(self._run(TOPIC_LABEL_PROMPT, content))
        except Exception:
            self._bump(failures=1)
            return None
        if not parsed:
            self._bump(failures=1, parse_failures=1)
            return None
        self._bump(ok=1)
        label = str(parsed["label"]).strip().rstrip(".")
        desc = str(parsed.get("description") or "").strip() or None
        sub = parsed.get("is_substantive")
        try:
            conf = float(parsed.get("confidence"))
            conf = min(1.0, max(0.0, conf))
        except (TypeError, ValueError):
            conf = None
        return label, desc, (sub if isinstance(sub, bool) else None), conf

    def label_topics(self, items: Sequence[Tuple[str, Sequence[str], Sequence[str], Optional[dict]]],
                     workers: int = 8) -> List[Optional[LabelResult]]:
        """label_topic over (brand, terms, samples, context) tuples, in parallel, order kept."""
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(lambda it: self.label_topic(*it), items))
        # label_topic swallows every failure, so give the misses one serial retry;
        # they are usually transient under concurrency. The first failure stays
        # counted, so `failures` reports how often the router misbehaved and
        # `ok` how many topics ended up labelled.
        for i, res in enumerate(results):
            if res is None:
                results[i] = self.label_topic(*items[i])
        return results


if __name__ == "__main__":
    try:
        c = AiRouterV1Client()
    except AiRouterConfigError as exc:
        raise SystemExit(f"not configured: {exc}")
    print(c.label_topic(
        "AppleSupport",
        ["ios", "ios 11", "11", "phone", "fix"],
        ["my phone is garbage after the iOS 11 update", "iOS 11 ruins my phone!",
         "I hate iOS 11 so much it has totally ruined my phone experience, fix this"],
        {"member_count": 3, "coherence": 0.21, "residual_cluster": False},
    ))
    print(c.label_topic("comcastcares", ["sent", "dm", "sent dm", "thanks"],
                        ["DM sent", "just sent you a DM thanks", "sent! thank you"]))
