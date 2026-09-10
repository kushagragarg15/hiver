"""Wiring tests on the mock backend. Not a quality signal -- just 'nothing throws
and the shapes are right'."""
import json

import pytest

from src.agent import SupportAgent
from src.classify import MajorityClassifier, classify_keyword, classify_llm
from src.escalate import decide
from src.intents import SLUGS, keyword_classify
from src.retrieve import Retriever
from eval.metrics import escalation_metrics, intent_metrics


def test_taxonomy_wellformed():
    assert len(SLUGS) == len(set(SLUGS)) == 8
    from src.intents import INTENTS
    for it in INTENTS:
        assert it.default_action in ("auto", "escalate")
        assert it.examples and it.description


@pytest.mark.parametrize("msg,expected", [
    ("my apple id is locked, cant get a 2fa code", "account_access_security"),
    ("charged twice, i want a refund", "billing_appstore_subscription"),
    ("battery drains fast since the update", "software_update_issue"),
    ("wifi keeps dropping bluetooth wont connect", "connectivity_sync"),
])
def test_keyword_classifier(msg, expected):
    assert keyword_classify(msg) == expected == classify_keyword(msg)


def test_majority_classifier():
    clf = MajorityClassifier().fit(["a", "a", "b"])
    assert clf.predict("whatever") == "a"


def test_retriever_finds_relevant():
    r = Retriever.from_config()
    hits = r.search("my phone battery dies after updating to ios 11", k=2)
    assert hits and hits[0].thread_id == 1
    assert hits[0].score > 0


def test_classify_llm_mock_shape():
    out = classify_llm("how do I set up a new iphone")
    assert out["intent"] in SLUGS + ["other"]
    assert 0.0 <= out["confidence"] <= 1.0


def test_escalate_hard_pattern_wins():
    d = decide("i will get my lawyer involved", "how_to_feature_question", 0.9, 9.0)
    assert d["action"] == "escalate" and "pattern" in d["reason"]


def test_escalate_always_intent():
    d = decide("cant log in to my account", "account_access_security", 0.95, 9.0)
    assert d["action"] == "escalate"


def test_escalate_auto_path():
    d = decide("how do I enable dark mode", "how_to_feature_question", 0.9, 9.0,
               draft={"used_weak_precedent": False})
    assert d["action"] == "auto"


def test_agent_end_to_end():
    a = SupportAgent()
    res = a.handle("my battery dies by noon since the ios 11 update")
    d = res.to_dict()
    for key in ("intent", "confidence", "draft_reply", "action", "reason", "retrieved"):
        assert key in d
    assert res.action in ("auto", "escalate")
    assert res.intent in SLUGS + ["other"]


def test_metrics_math():
    im = intent_metrics(["a", "b", "a"], ["a", "b", "b"], ["a", "b"])
    assert im["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    em = escalation_metrics(["escalate", "auto", "escalate"],
                            ["escalate", "auto", "auto"])
    assert em["counts"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}
    assert em["missed_escalation_rate"] == 0.5


def test_run_eval_smoke(tmp_path):
    from eval.run_eval import run
    res = run(limit=4, do_judge=True, judge_sample=4)
    assert res["intent"]["agent"]["n"] == 4
    assert "missed_escalation_rate" in res["escalation"]["agent"]
    assert res["reply_quality"]["agent"]["n"] == 4
