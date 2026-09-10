"""SupportAgent: classify -> retrieve -> draft -> escalate, for one message."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .classify import classify_llm
from .config import load_config
from .draft import draft_reply
from .escalate import decide
from .llm import LLM
from .retrieve import Retriever


@dataclass
class AgentResult:
    message: str
    intent: str
    confidence: float
    rationale: str
    retrieved: list[dict]
    draft_reply: str
    draft_meta: dict
    action: str
    reason: str
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupportAgent:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None,
                 use_llm_escalation_check: bool | None = None):
        cfg = load_config()
        self.cfg = cfg
        self.retriever = retriever or Retriever.from_config()
        self.llm = llm or LLM()
        self.k = cfg["retrieval"]["k"]
        self.min_score = cfg["retrieval"]["min_score"]
        self.use_llm_check = (cfg["escalation"].get("use_llm_check", False)
                              if use_llm_escalation_check is None
                              else use_llm_escalation_check)

    def handle(self, message: str, thread_context: str = "") -> AgentResult:
        cls = classify_llm(message, self.llm, thread_context=thread_context)
        exemplars = self.retriever.search(message, k=self.k)
        top_score = exemplars[0].score if exemplars else 0.0

        draft = draft_reply(message, cls["intent"], exemplars,
                            llm=self.llm, min_score=self.min_score)

        dec = decide(message, cls["intent"], cls["confidence"], top_score,
                     draft=draft, llm=self.llm, use_llm_check=self.use_llm_check)

        return AgentResult(
            message=message,
            intent=cls["intent"], confidence=cls["confidence"],
            rationale=cls["rationale"],
            retrieved=[e.__dict__ for e in exemplars],
            draft_reply=draft["reply"], draft_meta=draft,
            action=dec["action"], reason=dec["reason"], signals=dec["signals"],
        )
