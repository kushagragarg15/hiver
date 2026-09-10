"""Baselines the agent must beat.

Intent:
  trivial   MajorityClassifier  -- always the most common golden intent
  simple    keyword weak-labeller (src/intents.py)

Reply:
  trivial   fixed canned line ("Sorry to hear that, please DM us.")
  simple    retrieval-only -- echo the top precedent's first brand turn verbatim

Escalation:
  trivial-A always auto        (max automation, max risk)
  trivial-B always escalate    (zero risk, zero automation -- the "do nothing" bot)
  simple    escalate iff predicted intent's default_action == "escalate"
"""
from __future__ import annotations

from src.classify import MajorityClassifier, classify_keyword
from src.intents import BY_SLUG
from src.retrieve import Retriever

CANNED_REPLY = "Sorry to hear that. Please send us a DM with more detail and we'll help."


def intent_trivial(train_intents: list[str]):
    clf = MajorityClassifier().fit(train_intents)
    return lambda msg: clf.predict(msg)


def intent_simple():
    return classify_keyword


def reply_trivial():
    return lambda msg, exemplars: CANNED_REPLY


def reply_simple(retriever: Retriever, k: int = 4):
    def _fn(msg, exemplars=None):
        ex = exemplars or retriever.search(msg, k=k)
        if not ex:
            return CANNED_REPLY
        return ex[0].resolution.split(" || ")[0].strip() or CANNED_REPLY
    return _fn


def escalation_always_auto():
    return lambda **_: "auto"


def escalation_always_escalate():
    return lambda **_: "escalate"


def escalation_by_intent_prior():
    def _fn(intent: str = "", **_):
        it = BY_SLUG.get(intent)
        return it.default_action if it else "escalate"
    return _fn
