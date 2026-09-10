"""Intent taxonomy for the AI support agent.

The taxonomy was defined *from the data*: scripts/explore_intents.py samples
~600 inbound messages for the chosen brand, embeds + KMeans-clusters them, and
prints cluster exemplars. The eight buckets below are the human-readable
consolidation of those clusters. If you switch brands, re-run that script and
revise this file -- it is meant to be edited.

Each intent carries:
  slug          canonical id used everywhere (metrics, configs, golden set)
  label         human label for reports
  description   what belongs here (fed to the LLM classifier)
  examples      2-4 real-flavoured messages (few-shot for the classifier)
  default_action  "auto" | "escalate"  -- the *prior* before per-message signals
  rationale     why that default
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    slug: str
    label: str
    description: str
    default_action: str
    rationale: str
    examples: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


INTENTS: list[Intent] = [
    Intent(
        slug="software_update_issue",
        label="Software / iOS update issue",
        description=(
            "A software bug or regression: something broke after an OS or app "
            "update, an update won't install, or a feature stopped working. "
            "Not hardware, not billing."
        ),
        default_action="auto",
        rationale="Usually answerable with a known troubleshooting sequence the brand has posted many times.",
        examples=[
            "since the ios 11 update my battery percentage just disappears and the phone dies at 30%",
            "every time I try to update my apple watch it gets stuck on 'preparing update'",
            "the new update killed my keyboard, it lags for like 3 seconds before typing",
        ],
        keywords=["update", "ios", "ios11", "ios 11", "ios10", "upgrade", "install",
                  "beta", "bug", "glitch", "since the update", "after updating",
                  "new update", "latest update", "since the latest", "version",
                  "ruined my", "fix this", "fix your", "fix it", "autocorrect",
                  "keyboard", "freezing", "freezes", "lag", "laggy", "crashing",
                  "keeps crashing", "slowed down", "slowing down"],
    ),
    Intent(
        slug="device_hardware_battery",
        label="Device hardware / battery",
        description=(
            "A physical-device problem: battery health/drain not tied to an "
            "update, overheating, cracked or unresponsive screen, won't charge, "
            "water damage, buttons, speakers, camera hardware."
        ),
        default_action="auto",
        rationale="Triage + first-line steps are scriptable; a genuine RMA still gets handed off inside the flow.",
        examples=[
            "my iphone 6s shuts off randomly even when it says 40% battery",
            "phone gets really hot while charging and the screen has a green line down it",
            "airpod right side just stopped producing any sound at all",
        ],
        keywords=["battery", "overheat", "hot", "screen", "cracked", "charge",
                  "charging", "won't turn on", "wont turn on", "dead", "hardware",
                  "speaker", "microphone", "camera", "button", "water damage"],
    ),
    Intent(
        slug="account_access_security",
        label="Account access / security",
        description=(
            "Apple ID / account lockouts, forgotten passwords, two-factor "
            "problems, verification codes not arriving, suspicious sign-ins, "
            "account hacked or disabled, recovery."
        ),
        default_action="escalate",
        rationale="Identity-bound. The agent cannot verify the account holder; wrong help here is a security incident.",
        examples=[
            "my apple id has been disabled for security reasons and I can't get it back",
            "someone changed my recovery email and I'm locked out of everything",
            "not getting the two factor code on any of my devices, can't log in",
        ],
        keywords=["apple id", "appleid", "password", "locked out", "lock out",
                  "2fa", "two factor", "two-factor", "verification code", "hacked",
                  "disabled", "can't log in", "cant login", "recovery", "account"],
    ),
    Intent(
        slug="billing_appstore_subscription",
        label="Billing / App Store / subscription",
        description=(
            "Money: unexpected charges, App Store or iTunes purchases, "
            "subscription cancellation or renewal, refund requests, gift cards, "
            "payment method failures."
        ),
        default_action="escalate",
        rationale="Account-specific financial action the agent cannot take or promise; refund policy calls need a human.",
        examples=[
            "I've been charged twice for the same app, want a refund",
            "cancelled my apple music months ago and you're still billing me £9.99",
            "bought a game for my son by accident, how do I get my money back",
        ],
        keywords=["charge", "charged", "refund", "billing", "bill", "invoice",
                  "subscription", "subscribe", "cancel", "itunes", "app store",
                  "apple music", "icloud storage", "payment", "receipt", "gift card"],
    ),
    Intent(
        slug="connectivity_sync",
        label="Connectivity / sync",
        description=(
            "Wi-Fi, Bluetooth, cellular, hotspot, AirDrop, Handoff, or iCloud "
            "sync between devices not working. Data present but not moving."
        ),
        default_action="auto",
        rationale="Standard connectivity troubleshooting ladder; low blast radius.",
        examples=[
            "wifi keeps dropping every few minutes only on my macbook since yesterday",
            "photos won't sync to icloud, stuck on 'uploading 1 item' for two days",
            "bluetooth won't stay connected to my car after the last update",
        ],
        keywords=["wifi", "wi-fi", "bluetooth", "hotspot", "airdrop", "handoff",
                  "sync", "syncing", "icloud sync", "won't connect", "wont connect",
                  "keeps dropping", "no signal", "cellular"],
    ),
    Intent(
        slug="how_to_feature_question",
        label="How-to / feature question",
        description=(
            "A question about how to do or configure something that is working "
            "normally -- setup, a setting, whether a feature exists. Not a fault "
            "report."
        ),
        default_action="auto",
        rationale="Informational; answerable from documentation and precedent.",
        examples=[
            "how do I move my photos from an old iphone to a new one without a computer",
            "is there a way to schedule a text message to send later on ios",
            "where did the home button gesture setting go, can't find it",
        ],
        keywords=["how do i", "how to", "how can i", "is there a way", "where is",
                  "where do i", "can i", "what does", "setup", "set up", "enable"],
    ),
    Intent(
        slug="complaint_churn_risk",
        label="Complaint / churn risk",
        description=(
            "Expresses anger or disappointment, threatens to switch brands, "
            "demands escalation or compensation, or is a public reputational "
            "complaint -- often with no specific technical ask."
        ),
        default_action="escalate",
        rationale="Relationship risk and possible compensation/PR handling; a templated bot reply makes it worse.",
        examples=[
            "worst customer service on the planet, switching to android after 10 years",
            "third phone in a year with the same fault and you keep fobbing me off. done.",
            "you've had my laptop for 3 weeks for a 2 day repair. I want a manager.",
        ],
        keywords=["worst", "terrible", "disgusting", "switching to", "switch to android",
                  "going to android", "never buying", "unacceptable", "manager",
                  "complaint", "compensation", "sue", "lawyer", "trading standards",
                  "ombudsman", "ripoff", "rip off", "you suck", "u suck", "so done",
                  "i'm done", "im done", "class action"],
    ),
    Intent(
        slug="praise_or_non_actionable",
        label="Praise / non-actionable",
        description=(
            "Thanks, compliments, jokes, general chatter, or messages with no "
            "resolvable request. Includes 'resolved, thanks' closers."
        ),
        default_action="auto",
        rationale="A short acknowledgement is safe and sufficient; nothing to resolve.",
        examples=[
            "just want to say the support person on the phone today was amazing, thank you",
            "10 years an iphone user and still love it",
            "nvm figured it out, thanks anyway",
        ],
        keywords=["thank you", "thanks", "love my", "best phone", "appreciate",
                  "figured it out", "nvm", "never mind", "resolved", "sorted now"],
    ),
]

SLUGS: list[str] = [i.slug for i in INTENTS]
BY_SLUG: dict[str, Intent] = {i.slug: i for i in INTENTS}
OTHER = "other"  # escape hatch for the classifier; counts against coverage in metrics


def taxonomy_prompt_block() -> str:
    """Rendered into the classifier system prompt."""
    lines = []
    for i in INTENTS:
        lines.append(f"- {i.slug}: {i.description}")
    lines.append(f"- {OTHER}: none of the above / cannot tell.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Keyword weak-labeller -- used ONLY as the "simple baseline" classifier.
# --------------------------------------------------------------------------- #
_COMPILED = {
    i.slug: [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in i.keywords]
    for i in INTENTS
}
# order matters: earlier = higher priority on ties (safety-relevant first)
_PRIORITY = [
    "account_access_security",
    "billing_appstore_subscription",
    "complaint_churn_risk",
    "software_update_issue",
    "device_hardware_battery",
    "connectivity_sync",
    "how_to_feature_question",
    "praise_or_non_actionable",
]


def keyword_classify(text: str) -> str:
    """Score each intent by keyword hits; break ties by _PRIORITY."""
    text = text or ""
    scores = {slug: sum(bool(rx.search(text)) for rx in rxs)
              for slug, rxs in _COMPILED.items()}
    best = max(scores.values())
    if best == 0:
        return OTHER
    winners = [s for s in _PRIORITY if scores.get(s, 0) == best]
    return winners[0] if winners else max(scores, key=scores.get)
