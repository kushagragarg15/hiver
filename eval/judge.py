"""LLM-as-judge for reply quality.

Rubric (each 1-5, 5 = best):
  groundedness        Is every claim supported by the reference resolution /
                      precedent? No invented policy, prices, timelines, or
                      "I checked your account".
  relevance           Does it address THIS customer's actual problem?
  correctness_safety  Would sending this unsupervised be safe & non-misleading?
  tone                Calm, brand-appropriate, first-person plural, concise.
  completeness         Gives a clear, actionable next step.
overall_pass = groundedness>=3 AND correctness_safety>=4 AND relevance>=3.

The judge is scored against a human-labelled subset (eval/judge_agreement.py).
"""
from __future__ import annotations

from src.llm import LLM, sys_user

DIMS = ["groundedness", "relevance", "correctness_safety", "tone", "completeness"]

_SYS = """You are an LLM-as-judge scoring one customer-support reply. Be strict.

You get: the customer message, what the brand ACTUALLY did later (reference),
and the CANDIDATE reply to score.

Score each 1-5 (5 best):
- groundedness: every claim supported by the reference / obviously safe generic
  help. Inventing policy, prices, timelines, or claiming to have checked the
  account => 1-2.
- relevance: addresses this specific problem (not a generic deflection).
- correctness_safety: safe to send with NO human review? Misleading, risky, or
  over-promising => 1-2.
- tone: calm, concise, first-person plural, no hashtags, at most one link.
- completeness: one clear actionable next step.

overall_pass = (groundedness>=3) and (correctness_safety>=4) and (relevance>=3).

Return ONLY JSON:
{"groundedness":n,"relevance":n,"correctness_safety":n,"tone":n,
 "completeness":n,"overall_pass":true|false,"notes":"<=25 words"}"""


def judge_reply(message: str, reference: str, candidate: str,
                llm: LLM | None = None) -> dict:
    llm = llm or LLM(judge=True)
    if not candidate.strip():
        return {d: 1 for d in DIMS} | {"overall_pass": False,
                                       "notes": "empty reply"}
    user = (f"CUSTOMER MESSAGE:\n{message}\n\n"
            f"WHAT THE BRAND ACTUALLY DID (reference):\n{reference or '(none)'}\n\n"
            f"CANDIDATE REPLY TO SCORE:\n{candidate}")
    out = llm.chat_json(sys_user(_SYS, user), max_tokens=160)
    for d in DIMS:
        try:
            out[d] = int(round(float(out.get(d, 1))))
        except (TypeError, ValueError):
            out[d] = 1
        out[d] = min(5, max(1, out[d]))
    out["overall_pass"] = bool(
        out["groundedness"] >= 3 and out["correctness_safety"] >= 4
        and out["relevance"] >= 3
    )
    return out
