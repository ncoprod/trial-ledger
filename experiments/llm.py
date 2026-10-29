"""OpenAI-compatible chat client with an on-disk response cache, retries and a global
concurrency limit.

Every response is cached in ``experiments/cache/<model>.jsonl`` keyed by a SHA-256 of the full
request (model, messages, tools, sampling parameters and a per-episode salt), so re-running the
experiment replays the exact same transcripts without calling the API.
The API key is read from ``NVIDIA_API_KEY`` and is never written anywhere.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

from trial_ledger.hashing import canonical_json, sha256_hex

CACHE_DIR = Path(__file__).parent / "cache"
BASE_URL = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
_GLOBAL_SEM = threading.Semaphore(int(os.environ.get("LLM_MAX_CONCURRENCY", "2")))


class CacheMiss(RuntimeError):
    pass


def slug(model: str) -> str:
    return model.replace("/", "__")


class ChatCache:
    def __init__(self, model: str):
        self.path = CACHE_DIR / f"{slug(model)}.jsonl"
        self.lock = threading.Lock()
        self.data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self.data[row["key"]] = row["response"]

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, key: str, response: dict[str, Any]) -> None:
        with self.lock:
            self.data[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": response}, sort_keys=True) + "\n")


_CACHES: dict[str, ChatCache] = {}
_CACHES_LOCK = threading.Lock()


def get_cache(model: str) -> ChatCache:
    with _CACHES_LOCK:
        if model not in _CACHES:
            _CACHES[model] = ChatCache(model)
        return _CACHES[model]


class ChatClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        offline: bool = False,
        timeout: float = 150.0,
        max_retries: int = 7,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.offline = offline
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = get_cache(model)
        self._client = None
        self.n_api_calls = 0

    def _api(self):
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY")
            if not key:
                raise RuntimeError("set NVIDIA_API_KEY (or LLM_API_KEY) to call the API")
            self._client = OpenAI(
                base_url=BASE_URL, api_key=key, max_retries=0, timeout=self.timeout
            )
        return self._client

    def request_key(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], salt: str
    ) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "salt": salt,
                }
            )
        )

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], salt: str
    ) -> dict[str, Any]:
        """Return ``{"message": {...}, "finish_reason": str, "usage": {...}}``."""
        key = self.request_key(messages, tools, salt)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if self.offline:
            raise CacheMiss(key)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with _GLOBAL_SEM:
                    self.n_api_calls += 1
                    resp = self._api().chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                choice = resp.choices[0]
                msg = choice.message
                out = {
                    "message": {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (msg.tool_calls or [])
                        ],
                    },
                    "finish_reason": choice.finish_reason,
                    "usage": resp.usage.model_dump() if resp.usage else None,
                }
                self.cache.put(key, out)
                return out
            except Exception as exc:  # retry on rate limits, timeouts and 5xx
                status = getattr(exc, "status_code", None)
                retriable = status is None or status == 429 or status >= 500
                last_exc = exc
                if not retriable:
                    raise
                wait = min(120.0, 5.0 * 2**attempt) * (0.5 + random.random())
                print(
                    f"[retry] {self.model} attempt {attempt + 1}: {type(exc).__name__} "
                    f"status={status}; sleeping {wait:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"{self.model}: giving up after {self.max_retries} attempts: "
            f"{type(last_exc).__name__}: {str(last_exc)[:200]}"
        )
