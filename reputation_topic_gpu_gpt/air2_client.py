#!/usr/bin/env python3
"""Minimal AiRouterV2 client for topic labelling.

Modelled on
  interview-ai/backends/backend-main/src/modules/ai-router-v2/application/services/ai-router-v2.service.ts

IMPORTANT — how this router differs from a normal LLM API:

AiRouterV2 does not take a prompt. The system prompt, the model, and the output
JSON schema all live server-side on a *use-case*, selected by `useCaseName` plus
`promptVersion`. The client sends only `content`. So before `--label-method llm`
can work, somebody must provision a use-case (default name
`reputation-topic-label`) in the AiRouterV2 console, configured to accept the
JSON content this client sends and to return `{"label": "<short topic name>"}`.

Content this client sends, as a JSON string:

    {"brand": "AmazonHelp",
     "terms": ["late delivery", "parcel", "carrier"],
     "samples": ["my parcel never arrived", ...]}

Credentials come from the environment; nothing is read from another project's
.env automatically:

    AIR2_BASE_URL           default https://airouter-v2.questionpro.com/cs/api/air2
    AIR2_QP_OAUTH_HOST      host issuing the client_credentials token
    AIR2_S2S_CLIENT_ID      service client id
    AIR2_S2S_CLIENT_SECRET  service client secret
    AIR2_CONSUMER_ID        consumer id header value
    AIR2_USER_ID            QuestionPro user the call bills against
    AIR2_PROMPT_VERSION     default 'v1'
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Sequence

import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE = "https://airouter-v2.questionpro.com/cs/api/air2"
REQUEST_TIMEOUT = 120
MAX_ATTEMPTS = 4
RETRY_BASE_DELAY = 1.0
# The service token is short-lived; the reference implementation notes expires_in
# of 300s, so refresh well before that.
TOKEN_TTL_MARGIN = 60


class Air2ConfigError(RuntimeError):
    """Raised when the router is not configured, so callers can fall back."""


class Air2Client:
    def __init__(self) -> None:
        self.base = os.environ.get("AIR2_BASE_URL", DEFAULT_BASE).rstrip("/")
        self.oauth_host = os.environ.get("AIR2_QP_OAUTH_HOST", "").rstrip("/")
        self.client_id = os.environ.get("AIR2_S2S_CLIENT_ID", "")
        self.client_secret = os.environ.get("AIR2_S2S_CLIENT_SECRET", "")
        self.consumer_id = os.environ.get("AIR2_CONSUMER_ID", "")
        self.user_id = os.environ.get("AIR2_USER_ID", "")
        self.prompt_version = os.environ.get("AIR2_PROMPT_VERSION", "v1")
        missing = [k for k, v in {
            "AIR2_QP_OAUTH_HOST": self.oauth_host,
            "AIR2_S2S_CLIENT_ID": self.client_id,
            "AIR2_S2S_CLIENT_SECRET": self.client_secret,
            "AIR2_USER_ID": self.user_id,
        }.items() if not v]
        if missing:
            raise Air2ConfigError(f"missing env: {', '.join(missing)}")
        self._token: Optional[str] = None
        self._token_expires_at = 0.0

    # -- auth ---------------------------------------------------------------

    def _fetch_token(self) -> str:
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "userId": self.user_id,
        }).encode()
        req = urllib.request.Request(
            f"{self.oauth_host}/a/oauth2/token", data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            payload = json.loads(r.read().decode())
        token = payload.get("access_token")
        if not token:
            raise Air2ConfigError(f"no access_token in token response: {list(payload)}")
        self._token_expires_at = time.time() + max(30, int(payload.get("expires_in", 300))) - TOKEN_TTL_MARGIN
        return token

    def _token_value(self) -> str:
        if not self._token or time.time() >= self._token_expires_at:
            self._token = self._fetch_token()
        return self._token

    # -- run ----------------------------------------------------------------

    def _run(self, use_case: str, content: str) -> Optional[dict]:
        payload = json.dumps({
            "useCaseName": use_case,
            "promptVersion": self.prompt_version,
            "content": content,
        }).encode()
        last: Optional[Exception] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                headers = {
                    "Authorization": f"Bearer {self._token_value()}",
                    "Content-Type": "application/json",
                }
                if self.consumer_id:
                    headers["consumerId"] = self.consumer_id
                req = urllib.request.Request(f"{self.base}/public/v1/run",
                                             data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as exc:
                # 401 once means the short-lived token aged out mid-run.
                if exc.code == 401:
                    self._token = None
                elif exc.code < 500 and exc.code != 429:
                    raise
                last = exc
            except Exception as exc:                      # network, timeout
                last = exc
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        if last:
            raise last
        return None

    # -- labelling ----------------------------------------------------------

    def label_topic(self, use_case: str, brand: str, terms: Sequence[str],
                    samples: Sequence[str]) -> Optional[str]:
        """Ask the router for a short human-readable name for one topic.

        Returns None rather than raising when the router answers in an
        unexpected shape, so a labelling failure degrades to the c-TF-IDF label
        instead of failing the pipeline.
        """
        content = json.dumps({
            "brand": brand,
            "terms": list(terms)[:10],
            "samples": [str(s)[:280] for s in samples][:10],
        }, ensure_ascii=False)
        try:
            body = self._run(use_case, content)
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        data = body.get("data") or {}
        for item in (data.get("output") or []):
            if not isinstance(item, dict):
                continue
            if item.get("key") == "reasoning":
                continue
            value = item.get("content", item.get("value"))
            if isinstance(value, dict) and value.get("label"):
                return str(value["label"]).strip()
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict) and parsed.get("label"):
                        return str(parsed["label"]).strip()
                except json.JSONDecodeError:
                    return value.strip() or None
        return None


if __name__ == "__main__":
    try:
        c = Air2Client()
    except Air2ConfigError as exc:
        raise SystemExit(f"not configured: {exc}")
    print(c.label_topic(
        os.environ.get("AIR2_LABEL_USECASE", "reputation-topic-label"),
        "AmazonHelp",
        ["late delivery", "parcel", "carrier"],
        ["my parcel never arrived", "delivery was 3 days late"],
    ))
