"""Auto-handle vs escalate, with a stated reason.

Design: a transparent rule stack first (cheap, auditable, safety-biased), then
an optional LLM veto. Auto-handling requires ALL green lights; anything unsure
escalates. Every decision carries a human-readable `reason` and a `signals`
dict so failure analysis can see *why*.
"""
from __future__ import annotations

import re

from .config import load_config
from .intents import BY_SLUG
from .llm import LLM, sys_user

_CFG = load_config()["escalation"]
_HARD = [re.compile(p, re.I) for p in _CFG.get("hard_escalation_patterns", [])]
_ALWAYS = set(_CFG.get("always_escalate_intents", []))
_MIN_CONF = float(_CFG.get("min_intent_confidence", 0.6))
_MIN_RET = float(_CFG.get("min_retrieval_score", 3.0))

_LLM_SYS = """You are a safety gate for an automated support bot. Given a customer
message and the bot's proposed reply, answer whether it is safe to send this
WITHOUT a human reviewing it first.

Escalate (safe=false) if: the reply promises an account/order/refund action the
bot cannot verify; the message implies legal/safety/security risk; the customer
is highly distressed; the reply could be wrong in a costly way; or the reply is
evasive/non-responsive.

Return ONLY JSON: {"safe_to_auto": <bool>, "reason": "<=20 words"}"""


def decide(message: str, intent: str, confidence: float,
           retrieval_score: float, draft: dict | None = None,
           llm: LLM | None = None, use_llm_check: bool = False) -> dict:
    signals = {
        "intent": intent, "confidence": round(confidence, 3),
        "retrieval_score": round(retrieval_score, 3),
        "hard_pattern": None, "always_escalate_intent": intent in _ALWAYS,
        "weak_precedent": bool(draft and draft.get("used_weak_precedent")),
        "llm_gate": None,
    }

    for rx in _HARD:
        if rx.search(message or ""):
            signals["hard_pattern"] = rx.pattern
            return _out("escalate", f"message matched hard-escalation pattern /{rx.pattern}/", signals)

    if intent in _ALWAYS:
        return _out("escalate",
                    f"intent '{intent}' is account/identity/relationship sensitive; policy: never auto",
                    signals)

    if confidence < _MIN_CONF:
        return _out("escalate",
                    f"intent confidence {confidence:.2f} < {_MIN_CONF:.2f} threshold", signals)

    if retrieval_score < _MIN_RET:
        return _out("escalate",
                    f"no strong precedent (top BM25 {retrieval_score:.2f} < {_MIN_RET:.2f}); "
                    f"can't ground a trustworthy reply", signals)

    if draft and draft.get("used_weak_precedent"):
        return _out("escalate", "drafter flagged it leaned on weak/absent precedent", signals)

    if use_llm_check and draft:
        llm = llm or LLM()
        g = llm.chat_json(sys_user(_LLM_SYS,
            f"CUSTOMER: {message}\n\nPROPOSED REPLY: {draft.get('reply','')}"),
            max_tokens=80)
        signals["llm_gate"] = g
        if not bool(g.get("safe_to_auto", False)):
            return _out("escalate", f"LLM safety gate: {g.get('reason','flagged')}", signals)

    return _out("auto",
                f"confident intent ({confidence:.2f}), strong precedent "
                f"({retrieval_score:.2f}), non-sensitive category", signals)


def _out(action: str, reason: str, signals: dict) -> dict:
    return {"action": action, "reason": reason, "signals": signals}
