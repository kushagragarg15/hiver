"""Load config.yaml once and expose it as a plain dict + a few helpers."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of the src/ directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("SUPPORT_AGENT_CONFIG", ROOT / "config.yaml"))


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def rel(path: str | os.PathLike) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def llm_cfg(judge: bool = False) -> dict[str, Any]:
    """Flattened view of the active LLM provider's settings.

    judge=True resolves against a *separate* provider when one is configured
    (LLM_JUDGE_PROVIDER env, else llm.judge_provider in config.yaml) -- lets
    the judge run on a different model family/vendor than the agent for
    independence (e.g. agent on groq, judge on gemini) without touching the
    agent's provider.
    """
    c = load_config()["llm"]
    provider = os.environ.get("LLM_PROVIDER", c["provider"])
    if judge:
        provider = os.environ.get("LLM_JUDGE_PROVIDER", c.get("judge_provider", provider))
    prov = c.get(provider) or {"model": "mock", "judge_model": "mock"}
    model = prov.get("judge_model", prov["model"]) if judge else prov["model"]
    model_env = "LLM_JUDGE_MODEL" if judge else "LLM_MODEL"
    return {
        "provider": provider,
        "model": os.environ.get(model_env, model),
        "base_url": prov.get("base_url"),
        "api_key_env": prov.get("api_key_env"),   # env var holding the key; None => keyless (local)
        "rpm": prov.get("rpm"),                    # client-side rate cap for free tiers
        "reasoning_effort": prov.get("reasoning_effort"),  # e.g. "none" to disable Gemini 3.x thinking
        "temperature": c.get("temperature", 0.0),
        "max_tokens": c.get("max_tokens", 512),
        "cache_dir": c.get("cache_dir", ".llm_cache"),
        "timeout_s": c.get("timeout_s", 120),
    }


def brand() -> str:
    return load_config()["data_prep"]["brand"]
