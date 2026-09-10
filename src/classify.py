"""Intent classification.

Public API
    classify_llm(msg, llm)     -> {"intent", "confidence", "rationale"}
    classify_keyword(msg)      -> intent slug           (simple baseline)
    MajorityClassifier         -> fit on golden set, predict constant (trivial baseline)
"""
from __future__ import annotations

from collections import Counter

from .intents import OTHER, SLUGS, keyword_classify, taxonomy_prompt_block
from .llm import LLM, sys_user

_SYS = f"""You label a single inbound customer-support tweet with exactly one intent.

Intents:
{taxonomy_prompt_block()}

Rules:
- Choose the intent that matches the customer's PRIMARY need.
- Security/identity problems (login, Apple ID, 2FA, hacked) are always \
account_access_security, even if phrased as a complaint.
- Money problems (charges, refunds, subscriptions) are \
billing_appstore_subscription, even if phrased as a complaint.
- Use complaint_churn_risk only when there is no concrete technical/billing ask.
- confidence is your calibrated probability the label is correct, 0.0-1.0.

Return ONLY JSON: {{"intent": "<slug>", "confidence": <float>, "rationale": "<=15 words"}}"""


def classify_llm(message: str, llm: LLM | None = None, thread_context: str = "") -> dict:
    llm = llm or LLM()
    user = message if not thread_context else f"{message}\n\n[earlier in thread]\n{thread_context}"
    out = llm.chat_json(sys_user(_SYS, user), max_tokens=120)
    intent = str(out.get("intent", OTHER)).strip()
    if intent not in SLUGS and intent != OTHER:
        intent = OTHER
    try:
        conf = float(out.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(1.0, max(0.0, conf))
    return {"intent": intent, "confidence": round(conf, 3),
            "rationale": str(out.get("rationale", ""))[:200]}


def classify_keyword(message: str) -> str:
    """Simple baseline: keyword weak-labeller from src/intents.py."""
    return keyword_classify(message)


class MajorityClassifier:
    """Trivial baseline: always predict the most frequent training intent."""

    def __init__(self) -> None:
        self.label = OTHER

    def fit(self, intents: list[str]) -> "MajorityClassifier":
        self.label = Counter(intents).most_common(1)[0][0] if intents else OTHER
        return self

    def predict(self, _message: str) -> str:
        return self.label
