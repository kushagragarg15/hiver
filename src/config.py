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


def llm_cfg() -> dict[str, Any]:
    """Flattened view of the active LLM provider's settings."""
    c = load_config()["llm"]
    provider = os.environ.get("LLM_PROVIDER", c["provider"])
    prov = c.get(provider) or {"model": "mock", "judge_model": "mock"}
    return {
        "provider": provider,
        "model": os.environ.get("LLM_MODEL", prov["model"]),
        "judge_model": os.environ.get("LLM_JUDGE_MODEL", prov.get("judge_model", prov["model"])),
        "base_url": prov.get("base_url"),
        "temperature": c.get("temperature", 0.0),
        "max_tokens": c.get("max_tokens", 512),
        "cache_dir": c.get("cache_dir", ".llm_cache"),
        "timeout_s": c.get("timeout_s", 120),
    }


def brand() -> str:
    return load_config()["data_prep"]["brand"]
