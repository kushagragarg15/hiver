"""One thin client for both providers, with an on-disk response cache.

Both providers are driven through the OpenAI SDK:
  * provider=openai  -> api.openai.com, key from OPENAI_API_KEY
  * provider=ollama  -> http://localhost:11434/v1 (OpenAI-compatible), key ignored

Why a cache: the eval harness makes ~1-2k calls. Caching keeps a re-run of the
headline numbers well under the 15-minute budget and makes results deterministic
for a fixed model. Cache entries are keyed by (provider, model, messages,
params) so changing the prompt or model misses cleanly. Delete .llm_cache/ to
force a cold run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import llm_cfg, rel

_JSON_SPAN = re.compile(r"\{.*\}", re.S)


class _KeyPool:
    """Thread-safe round-robin over N API keys for one provider.

    Multiple free-tier keys (e.g. Gemini's 500 req/day PER KEY cap) are
    supplied as a comma-separated list in the same env var
    (GEMINI_API_KEY="key1,key2,key3"). Rotating on every call spreads load
    across all keys before any one comes back around, and a quota/429 on the
    key just used marks it exhausted so subsequent calls skip it -- no sleep
    needed as long as another key is still good. Pools are keyed by env-var
    name and shared across LLM instances (agent + judge use the same keys).
    """
    _pools: dict[str, "_KeyPool"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, keys: list[str]):
        self.keys = keys
        self._idx = 0
        self._exhausted: set[int] = set()
        self._last_call: dict[int, float] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls, cache_key: str, raw: str) -> "_KeyPool":
        with cls._registry_lock:
            pool = cls._pools.get(cache_key)
            if pool is None:
                keys = [k.strip() for k in raw.split(",") if k.strip()]
                pool = cls(keys)
                cls._pools[cache_key] = pool
            return pool

    def next(self) -> tuple[int, str]:
        """Pick the next non-exhausted key (round-robin), skipping exhausted ones."""
        with self._lock:
            n = len(self.keys)
            for _ in range(n):
                i = self._idx
                self._idx = (self._idx + 1) % n
                if i not in self._exhausted:
                    return i, self.keys[i]
            # every key looked exhausted -- reset so callers get a real error/backoff
            # instead of a permanent lockout (quotas reset daily; a stale mark
            # from earlier shouldn't wedge the process).
            self._exhausted.clear()
            i = self._idx
            self._idx = (self._idx + 1) % n
            return i, self.keys[i]

    def mark_exhausted(self, i: int) -> None:
        with self._lock:
            self._exhausted.add(i)

    def throttle(self, i: int, min_interval: float) -> None:
        if min_interval <= 0:
            return
        with self._lock:
            last = self._last_call.get(i, 0.0)
            wait = min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_call[i] = time.time()


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, pt: int, ct: int) -> None:
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls, "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class LLM:
    # Built-in defaults for OpenAI-compatible providers. `config.yaml` can add
    # more or override these; every one of these speaks /chat/completions.
    _PRESETS = {
        "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
        "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": None},
        "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                   "api_key_env": "GEMINI_API_KEY"},
        "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
        "deepseek": {"base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
        "xai": {"base_url": "https://api.x.ai/v1", "api_key_env": "XAI_API_KEY"},
        "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    }

    def __init__(self, model: str | None = None, judge: bool = False):
        cfg = llm_cfg(judge=judge)
        self.provider = cfg["provider"]
        self.model = model or cfg["model"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self.timeout_s = cfg["timeout_s"]
        self._min_interval = 60.0 / cfg["rpm"] if cfg.get("rpm") else 0.0
        # Gemini 3.x "flash" are thinking models -- without this they burn the
        # output budget on hidden reasoning and truncate the JSON.
        self._reasoning_effort = cfg.get("reasoning_effort")
        self.cache_dir = rel(cfg["cache_dir"]) / self.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._cfg = cfg
        self._base_url: str | None = None
        self._key_pool: _KeyPool | None = None
        self._clients: dict[int, Any] = {}   # key-index -> OpenAI client, built lazily

    def _ensure_pool(self) -> None:
        """Deferred so a fully-cached `make eval` needs no API key at all."""
        if self._key_pool is not None:
            return
        preset = self._PRESETS.get(self.provider, {})
        self._base_url = self._cfg.get("base_url") or preset.get("base_url")
        key_env = self._cfg.get("api_key_env") or preset.get("api_key_env")
        if key_env:
            raw = os.environ.get(key_env)
            if not raw:
                raise RuntimeError(
                    f"{key_env} not set (provider={self.provider}). "
                    f"Export it or switch llm.provider in config.yaml. "
                    f"(Multiple keys: set {key_env} to a comma-separated list.)")
            self._key_pool = _KeyPool.get(key_env, raw)
        else:
            self._key_pool = _KeyPool.get(f"__keyless__{self.provider}", "not-needed")

    def _client_for(self, idx: int, key: str):
        client = self._clients.get(idx)
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=self._base_url, api_key=key, timeout=self.timeout_s)
            self._clients[idx] = client
        return client

    # ------------------------------------------------------------------ #
    def _key(self, messages, json_mode, temperature, max_tokens) -> str:
        blob = json.dumps({
            "provider": self.provider, "model": self.model,
            "messages": messages, "json": json_mode,
            "t": temperature, "m": max_tokens,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, messages: list[dict], *, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None,
             use_cache: bool = True) -> str:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.max_tokens
        # Reasoning models (Gemini 3.x flash, gpt-oss, qwen3) spend completion
        # tokens on hidden thinking before the answer. A tight caller budget
        # (e.g. 120 for the classifier) then truncates the JSON. Give them room;
        # the cache key uses the *effective* value so this stays reproducible.
        if self._reasoning_effort:
            max_tokens = max(max_tokens, 800)
        ck = self._key(messages, json_mode, temperature, max_tokens)
        cpath = self.cache_dir / f"{ck}.json"

        if use_cache and cpath.exists():
            self.usage.cache_hits += 1
            return json.loads(cpath.read_text(encoding="utf-8"))["content"]

        if self.provider == "mock":
            content = _mock_response(messages)
            self.usage.add(0, 0)
            return content

        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        self._ensure_pool()
        n_keys = len(self._key_pool.keys)
        # With several keys, try every one at least once before giving up.
        max_attempts = max(5, n_keys + 2)

        last_err = None
        for attempt in range(max_attempts):
            idx, key = self._key_pool.next()
            self._key_pool.throttle(idx, self._min_interval)
            try:
                resp = self._client_for(idx, key).chat.completions.create(**kwargs)
                break
            except Exception as e:                      # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                if "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg:
                    self._key_pool.mark_exhausted(idx)
                    if n_keys > 1:
                        continue      # another key is likely still good -- no sleep
                    time.sleep(min(60, 8 * (attempt + 1)))   # single-key: free-tier cool-down
                else:
                    time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"LLM call failed after retries: {last_err}")

        content = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        self.usage.add(getattr(u, "prompt_tokens", 0) or 0,
                       getattr(u, "completion_tokens", 0) or 0)
        cpath.write_text(json.dumps({
            "content": content, "model": self.model, "provider": self.provider,
        }, ensure_ascii=False), encoding="utf-8")
        return content

    def chat_json(self, messages: list[dict], *, temperature: float | None = None,
                  max_tokens: int | None = None, use_cache: bool = True) -> dict:
        """chat() + defensive JSON parsing. One cache-busting retry on bad JSON."""
        for bust in (False, True):
            raw = self.chat(messages, json_mode=True, temperature=temperature,
                            max_tokens=max_tokens, use_cache=use_cache and not bust)
            parsed = _loads_loose(raw)
            if parsed is not None:
                return parsed
        raise ValueError(f"model did not return JSON: {raw[:300]!r}")


def _loads_loose(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    m = _JSON_SPAN.search(raw or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def sys_user(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Offline stub. `provider: mock` returns deterministic, schema-valid responses
# so the pipeline + eval harness can be exercised end-to-end with NO backend
# (CI, tests, `make smoke`). It is not a quality signal -- never report numbers
# from the mock provider.
# --------------------------------------------------------------------------- #
def _mock_response(messages: list[dict]) -> str:
    sysmsg = ((messages[0].get("content") if messages else "") or "").lower()
    user = (messages[-1].get("content") if messages else "") or ""
    low = user.lower()

    if "label a single inbound" in sysmsg:                       # classifier
        intent = "how_to_feature_question"
        for slug, kws in (
            ("account_access_security", ("apple id", "locked", "password", "2fa", "hacked")),
            ("billing_appstore_subscription", ("refund", "charged", "subscription", "billing")),
            ("device_hardware_battery", ("battery", "screen", "charge", "overheat")),
            ("software_update_issue", ("update", "ios", "install", "bug")),
            ("connectivity_sync", ("wifi", "bluetooth", "sync", "icloud")),
            ("complaint_churn_risk", ("worst", "switching", "manager", "unacceptable")),
        ):
            if any(k in low for k in kws):
                intent = slug
                break
        return json.dumps({"intent": intent, "confidence": 0.72,
                           "rationale": "mock stub keyword match"})

    if "you draft a public reply" in sysmsg:                     # drafter
        weak = "no strong precedent" in low
        return json.dumps({
            "reply": "Sorry for the trouble - we'd like to help. Which iOS "
                     "version are you on, and when did this start? We can pick "
                     "this up in DM if that's easier.",
            "grounded_in": [] if weak else [1],
            "used_weak_precedent": weak,
        })

    if "safety gate" in sysmsg:                                  # escalation gate
        return json.dumps({"safe_to_auto": True, "reason": "mock stub allows"})

    if "llm-as-judge" in sysmsg or "you are scoring" in sysmsg:  # judge
        return json.dumps({
            "groundedness": 4, "relevance": 4, "correctness_safety": 4,
            "tone": 4, "completeness": 3, "overall_pass": True,
            "notes": "mock stub score",
        })
    return json.dumps({"result": "mock", "ok": True})
