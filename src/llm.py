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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import llm_cfg, rel

_JSON_SPAN = re.compile(r"\{.*\}", re.S)


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
    def __init__(self, model: str | None = None, judge: bool = False):
        cfg = llm_cfg()
        self.provider = cfg["provider"]
        self.model = model or (cfg["judge_model"] if judge else cfg["model"])
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self.timeout_s = cfg["timeout_s"]
        self.cache_dir = rel(cfg["cache_dir"]) / self.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._client = self._make_client(cfg)

    def _make_client(self, cfg: dict[str, Any]):
        if self.provider == "mock":
            return None
        from openai import OpenAI

        if self.provider == "ollama":
            base = (cfg.get("base_url") or "http://localhost:11434").rstrip("/")
            return OpenAI(base_url=f"{base}/v1", api_key="ollama",
                          timeout=self.timeout_s)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set (provider=openai)")
        return OpenAI(api_key=key, timeout=self.timeout_s)

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

        last_err = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:                      # noqa: BLE001
                last_err = e
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
