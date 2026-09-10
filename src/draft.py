"""Grounded reply drafting.

The drafter is given the customer message, the predicted intent, and up to k
retrieved precedents (past customer issue + how the brand actually resolved it).
It must ground its reply in those precedents and must NOT invent policy,
timelines, or account-specific facts. If precedent is weak it says so and keeps
the reply to a safe next step.
"""
from __future__ import annotations

from .intents import BY_SLUG
from .llm import LLM, sys_user
from .retrieve import Exemplar

_SYS = """You draft a public reply for a brand's customer-support Twitter account.

Ground your reply in the PRECEDENTS provided (how this brand has actually
resolved similar issues). Hard rules:
- Do not invent policies, prices, repair times, or compensation.
- Do not claim to have looked at the customer's account or order.
- Never ask the customer to post personal info publicly; if you need account
  details, ask them to move to DM.
- Match the brand's precedent tone: calm, brief, first-person plural ("we").
- <= 55 words. One clear next step. No hashtags. At most one link, only if a
  precedent used one.
- If the precedents are weak or absent, acknowledge, ask the one most useful
  diagnostic question, and offer to continue in DM.

Return ONLY JSON:
{"reply": "<text>", "grounded_in": [<precedent numbers you used>],
 "used_weak_precedent": <true|false>}"""


def _format_precedents(exemplars: list[Exemplar], min_score: float) -> tuple[str, bool]:
    if not exemplars or exemplars[0].score < min_score:
        return ("(no strong precedent found)", True)
    lines = []
    for n, ex in enumerate(exemplars, 1):
        lines.append(
            f"[{n}] (score {ex.score}) customer: {ex.customer[:240]}\n"
            f"    brand resolved with: {ex.resolution[:400]}"
        )
    return ("\n".join(lines), False)


def draft_reply(message: str, intent: str, exemplars: list[Exemplar],
                llm: LLM | None = None, min_score: float = 3.0) -> dict:
    llm = llm or LLM()
    prec_block, weak = _format_precedents(exemplars, min_score)
    intent_label = BY_SLUG[intent].label if intent in BY_SLUG else intent
    user = (
        f"CUSTOMER MESSAGE:\n{message}\n\n"
        f"PREDICTED INTENT: {intent_label}\n\n"
        f"PRECEDENTS:\n{prec_block}"
    )
    out = llm.chat_json(sys_user(_SYS, user), max_tokens=260)
    reply = str(out.get("reply", "")).strip()
    return {
        "reply": reply,
        "grounded_in": out.get("grounded_in", []),
        "used_weak_precedent": bool(out.get("used_weak_precedent", weak)) or weak,
        "n_precedents": len(exemplars),
        "top_score": exemplars[0].score if exemplars else 0.0,
    }


def draft_retrieval_only(exemplars: list[Exemplar]) -> dict:
    """Simple baseline: return the top precedent's brand resolution verbatim."""
    if not exemplars:
        return {"reply": "", "grounded_in": [], "used_weak_precedent": True,
                "n_precedents": 0, "top_score": 0.0}
    top = exemplars[0]
    return {"reply": top.resolution.split(" || ")[0].strip(), "grounded_in": [1],
            "used_weak_precedent": False, "n_precedents": len(exemplars),
            "top_score": top.score}
