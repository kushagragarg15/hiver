"""Point every test at a tiny self-contained fixture repo + the mock LLM.

Runs before any `src`/`eval` import, so config.load_config() (lru-cached) picks
up SUPPORT_AGENT_CONFIG here.
"""
import json
import os
import tempfile
from pathlib import Path

import yaml

os.environ["LLM_PROVIDER"] = "mock"
os.environ["PYTHONIOENCODING"] = "utf-8"

_TMP = Path(tempfile.mkdtemp(prefix="support_agent_test_"))
_PROC = _TMP / "processed"
_GOLD = _TMP / "golden"
_REP = _TMP / "reports"
for d in (_PROC, _GOLD, _REP):
    d.mkdir(parents=True, exist_ok=True)

_HISTORY = [
    {"thread_id": 1, "brand": "TestBrand", "n_turns": 3,
     "first_ts": "2017-01-01T00:00:00+00:00", "last_ts": "2017-01-01T01:00:00+00:00",
     "customer_opening": "my battery drains super fast since the ios 11 update",
     "customer_all": "my battery drains super fast since the ios 11 update",
     "brand_resolution": "Try Settings > Battery to spot the culprit app, then reboot. That fixes most post-update drain.",
     "turns": [{"role": "customer", "text": "my battery drains super fast since the ios 11 update"},
               {"role": "agent", "text": "Try Settings > Battery to spot the culprit app, then reboot."},
               {"role": "customer", "text": "that worked, thanks"}]},
    {"thread_id": 2, "brand": "TestBrand", "n_turns": 3,
     "first_ts": "2017-01-02T00:00:00+00:00", "last_ts": "2017-01-02T01:00:00+00:00",
     "customer_opening": "how do I move photos from my old iphone to a new one",
     "customer_all": "how do I move photos from my old iphone to a new one",
     "brand_resolution": "Sign in to the same Apple ID on both, turn on iCloud Photos, and they sync across.",
     "turns": [{"role": "customer", "text": "how do I move photos from my old iphone to a new one"},
               {"role": "agent", "text": "Turn on iCloud Photos on both devices with the same Apple ID."},
               {"role": "customer", "text": "got it"}]},
    {"thread_id": 3, "brand": "TestBrand", "n_turns": 2,
     "first_ts": "2017-01-03T00:00:00+00:00", "last_ts": "2017-01-03T01:00:00+00:00",
     "customer_opening": "wifi keeps dropping on my macbook every few minutes",
     "customer_all": "wifi keeps dropping on my macbook every few minutes",
     "brand_resolution": "Remove the network in System Preferences > Network, then re-add it and renew the DHCP lease.",
     "turns": [{"role": "customer", "text": "wifi keeps dropping on my macbook every few minutes"},
               {"role": "agent", "text": "Remove and re-add the network, then renew the DHCP lease."}]},
]

_GOLDEN = [
    {"id": "gold_0001", "thread_id": 1, "message": "battery dies by noon since ios 11 update",
     "thread_context": "customer: battery dies by noon since ios 11 update",
     "reference_resolution": "Check Settings > Battery and reboot.",
     "gold_intent": "software_update_issue", "gold_action": "auto",
     "gold_action_reason": "known fix", "notes": "", "label_source": "human"},
    {"id": "gold_0002", "thread_id": 99, "message": "my apple id is locked and 2fa codes never arrive",
     "thread_context": "customer: my apple id is locked and 2fa codes never arrive",
     "reference_resolution": "Directed to iforgot.apple.com and account recovery.",
     "gold_intent": "account_access_security", "gold_action": "escalate",
     "gold_action_reason": "identity-bound", "notes": "", "label_source": "human"},
    {"id": "gold_0003", "thread_id": 2, "message": "how do I move photos to a new iphone",
     "thread_context": "customer: how do I move photos to a new iphone",
     "reference_resolution": "Use iCloud Photos with the same Apple ID.",
     "gold_intent": "how_to_feature_question", "gold_action": "auto",
     "gold_action_reason": "informational", "notes": "", "label_source": "human"},
    {"id": "gold_0004", "thread_id": 3, "message": "you charged me twice for apple music, i want a refund",
     "thread_context": "customer: you charged me twice for apple music, i want a refund",
     "reference_resolution": "Directed to reportaproblem.apple.com.",
     "gold_intent": "billing_appstore_subscription", "gold_action": "escalate",
     "gold_action_reason": "financial action", "notes": "", "label_source": "human"},
]

(_PROC / "TestBrand_history.jsonl").write_text(
    "\n".join(json.dumps(r) for r in _HISTORY), encoding="utf-8")
(_PROC / "TestBrand_eval_pool.jsonl").write_text(
    "\n".join(json.dumps(r) for r in _HISTORY), encoding="utf-8")
(_GOLD / "golden_set.jsonl").write_text(
    "\n".join(json.dumps(r) for r in _GOLDEN), encoding="utf-8")

_CFG = {
    "paths": {"raw_csv": "nonexistent.csv", "processed_dir": str(_PROC),
              "golden_dir": str(_GOLD), "reports_dir": str(_REP)},
    "data_prep": {"brand": "TestBrand", "split_date": "2017-11-15",
                  "min_thread_turns": 2, "max_history_threads": 100,
                  "eval_pool_size": 100, "random_seed": 13},
    "llm": {"provider": "mock", "temperature": 0.0, "max_tokens": 256,
            "cache_dir": str(_TMP / "cache"), "timeout_s": 30,
            "mock": {"model": "mock", "judge_model": "mock"}},
    "retrieval": {"method": "bm25", "k": 3, "min_score": 1.0},
    "escalation": {"min_intent_confidence": 0.6, "min_retrieval_score": 1.0,
                   "always_escalate_intents": ["account_access_security",
                                               "billing_appstore_subscription",
                                               "complaint_churn_risk"],
                   "hard_escalation_patterns": [r"\b(lawyer|legal|sue)\b",
                                                r"\b(refund|compensat)"]},
    "eval": {"golden_set": str(_GOLD / "golden_set.jsonl"),
             "judge_human_scores": str(_GOLD / "judge_human_scores.csv"),
             "results_json": str(_REP / "results.json"),
             "results_md": str(_REP / "results.md"),
             "judge_sample": 4},
}
_CFG_PATH = _TMP / "config.yaml"
_CFG_PATH.write_text(yaml.safe_dump(_CFG), encoding="utf-8")
os.environ["SUPPORT_AGENT_CONFIG"] = str(_CFG_PATH)
