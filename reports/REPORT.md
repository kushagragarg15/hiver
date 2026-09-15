# AI support agent — report

**Brand: AppleSupport** · Dataset: *Customer Support on Twitter* (Kaggle,
thoughtvector), Oct–Dec 2017 slice · Agent: intent → retrieval → grounded draft
→ escalate/auto.

> **Status of the numbers.** One run, committed with its warm cache
> (`make eval` replays it keyless in ~2 min):
>
> | worker | judge | n | gives |
> |---|---|---|---|
> | gemini `3.1-flash-lite` | gemini `3.1-flash-lite` (self) | **233** — the full golden set | intent + escalation + LLM-as-judge |
>
> Golden set: **233 rows** — 169 human-labelled (30 seed + 139 reviewed), 64
> still model-assisted drafts (`SAMPLING_NOTES.md`). The review pass changed
> 12 of 139 labels and moved macro-F1 0.63 → 0.60, missed-escalation 0.53 →
> 0.55 (§5.5). The judge's agreement
> with a human is now measured on 36 rows, with two judge vendors: **κ = 0.532**
> gemini (self), **κ = 0.566** groq/qwen3 (cross-vendor) — both below the ≥0.6
> bar, "directional only" (§3.4). Five free-tier API keys were pooled to get
> past Gemini's 500-requests/day cap (§5.8).
>
> **What the run shows:**
> 1. **The classifier clears both baselines.** Intent macro-F1 **0.60** /
>    accuracy **0.67** vs keyword **0.43** and majority **0.08**. Two tiny
>    classes (`complaint_churn_risk` F1 0.35 / 9 rows, `praise_or_non_actionable`
>    0.32 / 7) drag the macro; the six real classes are 0.58–0.78.
> 2. **The escalate/auto decision is not deployable.** Missed-escalation
>    **0.55** — 43 of the 78 messages that needed a human were auto-handled —
>    and the rule stack is one row away from the intent-prior baseline (0.55 vs
>    0.56). The confidence and retrieval floors never fire (§4 F1); 15 of the 43
>    misses are physical hardware faults the taxonomy routes to *auto* (F2).
> 3. **The reply-quality "win" is thinner than it looks.** Agent judge
>    pass-rate **0.91** vs canned "please DM us" **0.81** — but 88% of agent
>    replies *are* a DM redirect, groundedness is identical to canned (4.80),
>    and the whole gap is the relevance dimension: the agent appends a
>    diagnostic question. 15 of its 22 fails are fabricated capability (§3.3,
>    §4 F5, §5).

---

## 1. Problem framing

### 1.1 What the traffic actually looks like
The Oct–Dec 2017 AppleSupport slice is **dominated by the iOS 11 rollout**.
Clustering ~1,200 inbound openings (`scripts/explore_intents.py`) gives, in
round terms:

| theme | ~share | notes |
|---|---|---|
| iOS 11 bugs — battery drain, keyboard lag, the "I️" Unicode autocorrect bug, app crashes, freezes | ~45–50% | one mega-topic; several sub-bugs |
| generic frustration / "fix this" with no detail | ~15% | often still an iOS 11 bug underneath |
| device hardware / battery (not update-linked) | ~6% | AirPods pairing, overheating, cracked screens |
| billing / App Store / subscriptions | ~5% | refunds, wrong purchases, payment-method |
| how-to / feature questions | ~3% | "how do I…", settings |
| connectivity / sync | ~2% | Wi-Fi, Bluetooth, iCloud |
| account / Apple ID / security | ~2% | lockouts, 2FA, "can't buy — billing info" |
| praise / feature feedback / non-actionable | ~2% | |
| complaint / churn threat with no concrete ask | ~1–2% | "switching to Samsung" |

**Two structural facts drive every design choice:**

1. **Severe class imbalance.** One intent is ~half the traffic. Macro-F1 will be
   dragged down by the rare classes and that is the *honest* headline, not
   accuracy.
2. **AppleSupport barely resolves anything in-channel.** ~65% of its replies in
   this period are "join us in DM" + "which iOS version are you on?", regardless
   of how routine the question is. Twitter was an **intake funnel**, not a
   resolution channel. So "reply grounded in how the brand resolved it" mostly
   means grounding in *diagnostic questions and DM redirects* — see §5.

### 1.2 What "good" means here
The agent fronts a human queue. Good = **safely removes volume**, in priority
order:

1. **Near-zero missed escalations.** An auto-reply on something that needed a
   human is a wrong answer sent under Apple's name, unsupervised. Target: missed-
   escalation rate < 5% overall, **0% on the billing/account/hardware-safety
   slices**.
2. **Grounded replies** — every claim traceable to a precedent or to standard,
   verifiable troubleshooting. No invented policy / prices / repair times, no
   "I checked your account".
3. **Useful automation rate** — of what *should* be auto-handled, a large share
   is. A bot that escalates everything is safe and worthless (explicit baseline).
4. **Legible decisions** — every action carries a one-line reason (`signals` +
   `reason` in the output).

### 1.3 What I deliberately did NOT build
- **Multi-turn dialogue.** The agent answers the *opening* message only.
- **Account/order tools.** Those cases are *defined as* escalate; the job is to
  recognise them.
- **Fine-tuning.** ~200 labels — budget went to evaluation, not training.
- **Embedding retrieval** — BM25 only; embeddings stubbed behind
  `retrieval.method` for a later A/B.
- **Banking77** — its intents are retail-banking and would pull the taxonomy
  off-brand. Used for nothing.
- **A separate "I️ bug" intent** — it's a point-in-time defect; folded into
  `software_update_issue`.
- **Sentiment/abuse model** beyond routing `complaint_churn_risk` to a human.

---

## 2. Method (short)
`src/data_prep.py` reconstructs threads from the reply graph (union-find on
`in_response_to_tweet_id`) and splits by date: threads before **2017-11-15** are
the agent's retrieval memory (30k sampled across 2013–2017), threads on/after
are the eval pool (1,500). Taxonomy: **8 intents** from clustering +
hand-consolidation (`src/intents.py`). Per message: LLM classifies intent +
calibrated confidence → **BM25** retrieves `k=4` pre-split precedents → LLM
drafts a reply *grounded in them* with an explicit no-fabrication instruction →
a **rule stack** decides auto/escalate (hard regex → always-escalate intents →
confidence floor 0.60 → BM25-score floor 3.0 → weak-precedent flag → optional
LLM veto); **anything uncertain escalates**. Full rationale in `DECISION_LOG.md`.
*Result: §4 F1 shows the confidence and BM25 floors never fire in practice
(the model reports 0.85–1.0 on every row, BM25 never drops below 18), so only
the intent categories, the regexes and the weak-precedent flag do any work —
and the stack misses 55% of true escalations.*

Golden set: **233 rows** (169 `human` — 30 seed + 139 reviewed — and 64
`model_assisted` drafts — `SAMPLING_NOTES.md`). `gold_action` = *should a good
bot have handled this unsupervised* — deliberately not "what Apple did".

---

## 3. Results vs baselines

Run: **`gemini-3.1-flash-lite`, n = 233** (all of `golden_set.jsonl`; 155
`auto` / 78 `escalate`). Tables: `results.md`; per-row output:
`agent_golden_preds.jsonl`.

### 3.1 Intent classification
| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| **agent — LLM classifier** | **0.67** | **0.60** | **0.69** |
| simple baseline — keyword weak-labeller | 0.42 | 0.43 | 0.49 |
| trivial baseline — majority intent | 0.46 | 0.08 | 0.29 |

Per class (F1 / support): `software_update_issue` 0.78 / 108 ·
`billing_appstore_subscription` 0.77 / 18 · `device_hardware_battery` 0.73 / 28 ·
`account_access_security` 0.70 / 11 · `connectivity_sync` 0.59 / 15 ·
`how_to_feature_question` 0.58 / 37 · `complaint_churn_risk` **0.35** / 9 ·
`praise_or_non_actionable` **0.32** / 7.

76 errors. The three clusters that matter: **how-to → software fault** (10;
"Keynote crashes when I reorder slides", "lost the hashtag key" — is that a
question or a bug?); **software fault → praise/non-actionable** (8; terse or
sarcastic bug reports — "Updated and still seeing boxes 🤬", "will 11.2 fix
black screen?" — land in the *praise* bucket, precision 0.22); and **anything
angry → `complaint_churn_risk`** (15 of 20 predictions are wrong, precision
0.25 — §4 F4). The software → hardware boundary accounts for 7 more (§4 F5).

### 3.2 Escalate vs auto (positive class = escalate)
| system | esc-recall | **missed-esc ↓** | unnec-esc ↓ | auto-rate | esc-F1 |
|---|---|---|---|---|---|
| **agent — rule stack** | 0.45 | **0.55** | 0.08 | 0.80 | 0.56 |
| simple — escalate by intent prior | 0.44 | 0.56 | 0.06 | 0.82 | 0.56 |
| trivial — always-auto | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| trivial — always-escalate | 1.00 | **0.00** | 1.00 | 0.00 | 0.50 |

**This is the broken part of the system.** Three readings:
- **The stack is the intent prior plus five rows.** Of 47 agent escalations,
  42 come from the three always-escalate intents, 4 from the weak-precedent
  flag and 1 from a hard regex. Missed-escalation 0.55 vs the prior's 0.56 is a
  one-row difference (43 vs 44 misses; esc-F1 identical at 0.56). Everything
  downstream of the classifier is dead weight.
- **The floors never fire.** Confidence is 0.85–1.00 on every one of 233 rows
  (median 0.95); BM25 top score is 18–112 (median 38). `min_intent_confidence:
  0.60` and `min_retrieval_score: 3.0` triggered zero times. §4 F1.
- **43 of 78 true escalations were auto-handled, every one at confidence
  ≥ 0.85.** They fall into nameable groups: 15 rows the classifier put in
  `device_hardware_battery` — 14 physical faults correctly labelled and auto'd
  *by design*, plus `gold_0063`, a third-party accessory warranty question
  the review relabelled to `complaint_churn_risk` (F2); 8 sensitive cases
  (account/billing) the classifier put in a benign bucket (F3); 5 image/
  video-only openings and 2 non-English ones (`gold_0032` pt, `0102` it) (F5);
  3 pro-app / specialist routing (`gold_0022`/`0033` iMovie, `0099` Apple TV);
  3 distress or churn riding a real technical issue (`gold_0023`, `0026`,
  `0198`); 2 "prior support already failed" (`gold_0013`, `0119`); one safety
  case, `gold_0140` (iPhone dialled 911 by itself); and 4 ordinary-looking
  software/sync faults the labeller judged beyond first-line (`gold_0011`
  macOS install + suspected hardware, `0069` Podcasts playback, `0112`
  third-party-app lockout with photo loss, `0216` iCloud data loss) — the
  last group is the one where the label itself is most arguable.

The 12 unnecessary escalations are 6 tone-driven `complaint_churn_risk`
mislabels (F4); 4 weak-precedent flags, two of them on thank-you notes
(`gold_0041`, `0075`); one always-escalate-intent misfire (`gold_0068`, a
disabled-iPhone recovery the classifier read as `account_access_security`);
and one *correctly* classified billing row (`gold_0195`, "purchased song still
shows a price") where the labeller judged the standard check safe to
automate.

### 3.3 Reply quality — LLM-as-judge (1–5), n=233, judge = worker model
| system | groundedness | corr./safety | relevance | tone | completeness | pass-rate |
|---|---|---|---|---|---|---|
| **agent (retrieval-grounded)** | 4.80 | **4.84** | **4.57** | 4.92 | **4.56** | **0.91** |
| trivial — canned "please DM us" | 4.80 | 4.81 | 3.68 | 4.42 | 3.68 | 0.81 |
| simple — retrieval-only (echo top precedent) | 4.10 | 4.24 | 3.52 | 4.44 | 3.55 | 0.64 |

The agent passes 0.91 to canned 0.81 — a 22-row gap. Read it carefully:
- **Groundedness is identical (4.80) and safety within 0.03.** The judge treats
  "please DM us" as maximally grounded and safe, and **204 of 233 agent replies
  (88%) are a DM redirect** too — median 31 words, typically *acknowledge → one
  diagnostic question → DM link*. The agent isn't more grounded than a constant
  string; it's the same deflection with a question attached.
- **The entire gap is relevance.** Canned fails 44 rows, 37 of them on
  `relevance < 3`: a fixed line reads as irrelevant on how-tos, on praise, and
  on account-security cases where the judge notes "offering DM is unsafe". The
  agent's appended question buys back those rows.
- **The agent's 22 fails are mostly fabricated capability, not deflection.** 15
  score `groundedness < 3`: "send us a DM with *your location*" on the 911 case
  (`gold_0140`); "confirm the Apple ID you used" for a 4-year-old invoice the
  reference says can't be retrieved (`gold_0208`); "it will be fixed in a
  future update" contradicting a reference that says it was already fixed
  (`gold_0105`); "please DM us" as the answer to a phishing report
  (`gold_0108`). The other 7 are pure deflections on simple how-tos
  (`gold_0005`, `0073`, `0182`) and the four non-English messages, which get
  the "we support in English" precedent verbatim (`gold_0032`, `0079`, `0166`,
  `0201`).
- **An earlier 30-row run had canned *ahead* (0.93 vs 0.90).** That was one
  row. At n=233 the ordering flips and the gap is real — but it is a gap in
  *relevance-of-deflection*, not in resolution. On this brand the pipeline's
  contribution over a constant string is "ask which iOS version".

### 3.4 Is the judge trustworthy?  *(measured, cross-vendor too — and it misses the bar either way)*
`make worksheet` (36 rows: 12 messages × {agent, retrieval-only, canned},
source hidden) → hand-scored → `make judge` → `reports/judge_agreement.json`.
Bar: **κ ≥ 0.6 and safety within-1 ≥ 0.8**. Run twice — once with the
gemini judge that graded §3.3 (self-grading its own worker), once with
`LLM_JUDGE_PROVIDER=groq` (qwen3-27b, a different vendor and model family,
live — `reports/judge_agreement_groq.json`) — to separate "judge disagrees
with a human" from "judge is grading its own output":

| | gemini judge (self) | groq judge (qwen3, cross-vendor) |
|---|---|---|
| overall_pass accuracy | 0.778 | 0.778 |
| **Cohen's κ (overall_pass)** | **0.532** | **0.566** |
| correctness_safety within-1 | 0.861 | 0.722 |
| human pass-rate | 0.556 | 0.556 |
| judge (model) pass-rate | 0.722 | 0.389 |

**Verdict both times: "directional only — report with a caveat."** Neither
clears κ ≥ 0.6. Switching vendors doesn't fix it, it *inverts* it: the gemini
judge is lenient relative to the human (72% vs 56% pass), the groq judge is
harsh (39% vs 56%) and drags every dimension down 1–1.7 points (e.g.
groundedness mean 3.06 vs the human's 4.75, exact agreement 0.111). Per-row
agreement is worse cross-vendor even though κ ticks up slightly — the two
judges don't converge on the *same* rows, they just miss the human in
opposite directions at similar overall rates. **This means §3.3's pass-rates
(0.91 agent / 0.81 canned / 0.64 retrieval-only) should be read as directional
only** — the disagreement isn't an artifact of self-grading, it's the rubric
or the 36-row sample; a larger human-scored set is needed before either
judge's pass-rate is reportable to two decimal places.

A second, independent reason to discount §3.3: the judge's own notes show it
grading against `reference_resolution` — i.e. it rewards *imitating Apple*,
while `gold_action` was labelled against *what a good bot should do*
(`SAMPLING_NOTES.md`). Those two targets disagree by construction on this
brand. (`config.yaml`'s `judge_provider` default is left unset so the
committed §3.3 headline stays the reproducible gemini-self-judge run; the
groq comparison above used `LLM_JUDGE_PROVIDER=groq` for this one check only,
not baked into the committed headline.)

---

## 4. Top 5 failure modes

From `reports/agent_golden_preds.jsonl` (n=233). Ordered by how much of the
missed-escalation rate each explains.

### F1 — the escalation stack is the intent prior wearing a costume
- **Mechanism.** Auto-handle needs confidence ≥ 0.60 **and** BM25 ≥ 3.0 **and**
  intent ∉ always-escalate **and** no hard-regex hit **and** no weak-precedent
  flag. The classifier reports **0.85–1.00 on all 233 rows** (median 0.95,
  zero below 0.60), and AppleSupport's shared filler ("@115858", "fix this",
  "since the update") keeps the BM25 top score at **18–112** (zero below 3.0).
  So of five gates, two are inert; what fires is: always-escalate intents 42×,
  weak-precedent 4×, hard regex 1× (`emergency`).
- **Evidence.** Agent missed-escalation 0.55 vs the intent-prior baseline's
  0.56 — one row. Every one of the 43 misses was predicted at ≥ 0.85.
- **Fix.** Stop gating on the raw confidence scalar — it carries no
  information here. Either calibrate it against held-out labels (it would need
  ~200 escalate rows to fit a reliability curve) or drop it and make the LLM
  escalation veto (`use_llm_check`, off in this run) mandatory, with the
  retrieved precedents in context so it can answer "is there a safe public
  answer to this". Longer term: a learned P(escalate) head (§6).

### F2 — physical hardware faults are auto-handled *by design*
- **Mechanism.** `device_hardware_battery` was defined as a triage bucket
  (battery drain, AirPods pairing → "which iOS version?") with
  `default_action: auto`. But the same bucket is where the model — correctly —
  puts things that need inspection, warranty or a service appointment.
  Nothing downstream distinguishes them.
- **Evidence.** **15 of the 43 misses** land in this bucket (14 correctly
  classified; `gold_0063` is an accessory-warranty question the review moved
  to `complaint_churn_risk`), all auto'd at 0.95–1.00: `gold_0082` "iPhone 7
  Plus appears to have **bent**";
  `gold_0219` "fell into the **pool** for 30 seconds"; `gold_0057` MacBook
  **logic board**; `gold_0015` iPhone X **overheating**; `gold_0191` rear
  **camera dead** after troubleshooting; `gold_0107` **spacebar** failing;
  `gold_0110` **lines and discoloration** on screen; `gold_0180` "battery needs
  **service**" on a 6-month-old phone; `gold_0184` cold-shutdown *after* the
  recall battery replacement; `gold_0127` broken charging port + AppleCare
  question; `gold_0147` dead after a drop; `gold_0002` warranty claim;
  `gold_0109` speaker crackle; `gold_0185` iPhone SE panel-spec dispute.
- **Fix.** Split the bucket: `device_fault_physical` (default escalate) vs
  `device_triage` (default auto). Until relabelling, a forced-escalate lexicon
  covers most of this list: `bent|swollen|overheat|burning|water|pool|drop|
  cracked|lines on|logic board|camera (won't|doesn't|not) work|needs? (to be
  )?serviced|warranty|AppleCare`.

### F3 — the always-escalate guard has a single point of failure: the classifier
- **Mechanism.** Account, billing and churn are escalated *only if the
  classifier says so*. Recall on the two sensitive classes is 0.64 (account)
  and 0.67 (billing); the misses go to benign buckets and straight to auto.
- **Evidence.** `gold_0108` "I assume it's a **scam**?" (a phishing report) →
  `praise_or_non_actionable`, confidence **1.00**, auto, reply "please DM us".
  `gold_0116` "my Apple account has been **broken for 10 weeks**, family
  sharing doesn't work" → `connectivity_sync`, auto. `gold_0151` botched
  number transfer, case number, stranded abroad → `connectivity_sync`, auto.
  `gold_0113` locked iPhone 6 → `connectivity_sync`, auto. Billing:
  `gold_0054` "subscribed to Apple Music and it doesn't work" →
  `software_update_issue`; `gold_0039`/`0224` Developer Portal / iTunes
  Connect → software/how-to; `gold_0218` week-old pre-order → praise. **8 of
  the 43 misses.**
- **Fix.** Sensitive-category detection must not depend on argmax intent. Ask
  the classifier for a separate `sensitive: {account, payment, security,
  none}` field scored independently of intent, and escalate on *either*. Add
  `scam|phishing|fraud|hacked` to the hard regex (only `fraud|hacked` are there
  now).

### F4 — tone drives `complaint_churn_risk`, in both directions
- **Mechanism.** The model reads profanity/caps as the *intent*. 20
  `complaint_churn_risk` predictions, 5 correct (precision 0.25). Because it's
  an always-escalate class, this is the main source of unnecessary escalation —
  and, by luck, of caught escalations too.
- **Evidence.** 6 of the 12 unnecessary escalations: `gold_0065` "really
  annoyed with this I.T bullshit"; `gold_0066` "anyone else's battery terrible
  since the X launch!!??"; `gold_0145` "why my shit start fucking up when a
  new phone drops"; `gold_0176` "how can I get support today?
  #unhappycustomer"; `gold_0209` "why is it so hard to get somebody on the
  phone?"; `gold_0079` a Turkish "I'll switch to Samsung" over the Control
  Center Wi-Fi toggle. Meanwhile 9 of the 15 mislabels were `gold_action: escalate` for a
  *different* reason (`gold_0009` deleted music, `gold_0149` rep dropped the
  chat) — the right action for the wrong reason, which inflates recall.
  The mirror image: sarcastic bug reports land in `praise_or_non_actionable`
  (precision 0.22) — `gold_0110` "Thanks apple for the lines and the
  decoloration 😱", `gold_0084` "Updated and still seeing boxes 🤬".
- **Fix.** Separate *what* from *how*: intent = the concrete need; `tone:
  {neutral, frustrated, hostile, churn_threat}` as a second field. Escalate on
  `churn_threat` or `hostile` regardless of intent (this also catches
  `gold_0023`/`0026`/`0198`, distress riding a technical issue, which are among
  the misses). Delete `complaint_churn_risk` as an intent.

### F5 — the agent answers messages it cannot actually read
- **Mechanism.** Five openings are an image or video plus ≤ 8 words. The
  classifier assigns a confident intent from the words alone, the drafter
  produces a fluent "we'd like to help — DM us", the stack auto-handles. Four
  more are not in English; BM25 retrieves the brand's "we support in English,
  see this link" precedent and the drafter echoes it — and on review two of
  those four (`gold_0032`, `0102`) were relabelled *escalate* ("route to
  localised support"), so they now count as misses too.
- **Evidence.** `gold_0014` "how can I fix this? [link]" → `software_update`
  0.95, auto. `gold_0124` "What's this? [link]" → 0.95, auto. `gold_0161`
  "every minute this happens [video]" → 0.85, auto. `gold_0174` "WHAT IS THIS?
  I just want to use my MacBook [image]" → 0.85, auto. `gold_0128` "how do I
  get rid of this pop up? [image]" → auto. Non-English: `gold_0032` (pt),
  `0079` (tr), `0166`/`0201` (es) — all pass the escalation stack, all fail
  the judge on relevance (score 2). Also in this group, a **groundedness
  failure the anti-fabrication prompt didn't stop**: `gold_0140` "iPhone
  dialled 911 by itself" → reply asks the customer to "DM us with your
  location".
- **Fix.** Two pre-classifier checks, no LLM needed: `has_media and n_words <
  10 → escalate ("cannot see attachment")`; `langdetect != en → escalate
  ("route to localised support")`. For the fabrication: add "never ask for
  location, ID, or payment details" to the drafter's suppression list and
  make the judge's `correctness_safety` a hard fail on it.

---

## 5. What is misleading about my headline number? *(mandatory)*

The line that looks like a win is **"intent macro-F1 0.60 beats every baseline;
auto-rate 0.80; reply pass-rate 0.91, ahead of the canned baseline."** Why not
to trust it:

1. **"Auto-rate 0.80" is not automation — it's a rubber stamp.** The same run
   auto-handles **55% of the messages that needed a human** (43 of 78), every
   one at confidence ≥ 0.85. The only safe system in §3.2 is always-escalate,
   which automates nothing. Quoting auto-rate without missed-escalation next
   to it is the misleading framing.
2. **The agent's escalation logic adds one row over a lookup table.** Missed
   0.55 vs the intent-prior baseline's 0.56 (§4 F1). Everything in the stack
   except "which intent is it" is inert on this data. The result is really
   "intent classification, with three classes hard-wired to escalate".
3. **Reply pass-rate 0.91 vs canned 0.81 is a gap in relevance, not in
   help.** Groundedness is identical (4.80), 88% of agent replies are DM
   redirects, and the agent's edge is the diagnostic question it appends
   (§3.3). AppleSupport's real replies *are* deflections, so a rubric anchored
   on `reference_resolution` rewards deflection — and the agent's 22 fails are
   mostly it *inventing* capability ("DM us your location") rather than
   deflecting. On an earlier 30-row slice canned was *ahead*; the ordering at
   n=233 is real but the effect it measures is small.
4. **Judge-human agreement misses the trust bar, on two different vendors.**
   κ vs a human on 36 hand-scored rows: **0.532** with the gemini judge
   (self-grading its own worker), **0.566** with a cross-vendor groq/qwen3
   judge (§3.4). Neither clears 0.6. Switching vendors doesn't converge on
   the human — it flips direction (gemini over-passes 72% vs human 56%; groq
   under-passes 39% vs 56%). §3.3's pass-rates are real signal but not
   precise to two decimals. The judge also scores against what Apple *did*,
   while `gold_action` was labelled against what a good bot *should* do —
   two different targets.
5. **One annotator, and the labels move the headline.** 169 of 233 rows are
   human-labelled (30 seed + 139 reviewed); 64 are still model-assisted drafts
   (`SAMPLING_NOTES.md`). The review pass changed **12 of 139** labels (10
   intents, 4 actions — ~9%) and that alone moved macro-F1 **0.63 → 0.60**,
   accuracy 0.69 → 0.67 and missed-escalation **0.53 → 0.55** with the model's
   outputs held fixed (cache replay). So the second decimal of every number in
   §3 is label noise. 95% CI on missed-escalation at n=78 positives is
   **±0.11**; on the 7- and 9-row classes it's meaningless. `gold_action` is
   *my* call on what's automatable, made with hindsight
   (`reference_resolution` visible) the live agent never has.
6. **Macro-F1 is dragged by two tiny classes that shouldn't be intents.**
   `complaint_churn_risk` (F1 0.35, 9 rows) and `praise_or_non_actionable`
   (0.32, 7 rows) are *tone* labels; the six real classes are 0.58–0.78 (§4
   F4). Deleting those two
   buckets would raise macro-F1 without the model improving — and the taxonomy
   also conflates triageable and physical hardware faults (F2).
7. **The majority-intent baseline is fit on the golden labels** (peeks at
   test priors), so "beats majority" understates the gap — that column isn't a
   clean floor.
8. **Five free-tier keys pooled to reach n=233.** Gemini's cap is 500
   requests/day per key; this run is 1,132 live calls. `src/llm.py` rotates a
   comma-separated key list. Reproducing the exact numbers depends on the
   committed `.llm_cache/`; Gemini model ids churned mid-project
   (`gemini-2.x-flash` went "not available to new users").
9. **Frozen 2017 world + opening-message only.** Grounding is in iOS-11-era
   precedent; real resolution takes the whole thread. Both inflate any
   "helpfulness" read.

**Bottom line.** The one solid result is narrow: **the LLM classifier beats the
keyword and majority baselines on intent (macro-F1 0.60 vs 0.43 / 0.08), and the
head classes are usable (F1 0.70–0.78 on the four largest).** The escalate/auto
decision as built is not deployable — it misses 55% of true escalations because it reduces to intent
routing and the taxonomy sends physical hardware faults to auto. Reply quality
can't be judged meaningfully on this brand until there's a non-deflecting
baseline and a human-anchored judge from a different model family.

---

## 6. One more week (in priority order)
1. **Redesign the escalation decision** (the actual broken thing). Day 1–2,
   no relabelling needed: (a) split `device_hardware_battery` by a
   physical-fault lexicon and default the physical half to escalate (F2 — 15
   rows); (b) media-only and non-English pre-checks (F5 — 7 rows); (c) add
   `scam|phishing` to the hard regex (F3). Day 3: ask the classifier for two
   extra fields — `sensitive` and `tone` — scored independently of intent, and
   escalate on either (F3/F4). Drop the confidence floor; it's inert. Turn the
   LLM veto on with precedents in context. Re-run: target missed-escalation
   < 0.15 at auto-rate ≥ 0.6, then chase 0.05.
2. **Finish the label review and get a second annotator.** 139 of the 203
   model-assisted rows have been reviewed (`label_tool.py --review`); 64
   remain, and the pass so far changed ~9% of labels and moved the headline
   by 0.02–0.03 (§5.5). A second annotator on 100 rows for inter-annotator κ
   matters more than the last 64 — the F2/F4 fixes are worthless if the
   hardware/tone labels themselves are hindsight guesses.
3. **Judge trust — done, and it doesn't clear the bar either way.** Measured
   both self-judge (gemini, κ=0.532) and cross-vendor (groq/qwen3, κ=0.566);
   both below 0.6, and they miss the human in *opposite* directions (§3.4).
   Next: 2-shot the rubric with a couple of hand-scored examples in-prompt, add
   an explicit "asks for location / ID / payment" hard-fail on
   `correctness_safety` (the fabrication pattern in §3.3 slipped through at
   scores of 2–3), and grow the hand-scored set past 36 — that sample is too
   small to tell rubric noise from real judge bias.
4. **Re-baseline reply quality honestly.** Make the canned line the *primary*
   comparison and report the agent as a delta on *relevance* and
   *completeness* only — the two dimensions where it can differ from a
   constant string on this brand. Add a "same model, no anti-fabrication
   rules" baseline to measure what the suppression prompt actually buys.
5. **Fix the taxonomy** (F4/F5): delete `complaint_churn_risk` and
   `praise_or_non_actionable` as intents (they become `tone`), split
   `device_hardware_battery` into physical vs triage, and merge
   software/hardware *triage* into one bucket (identical `default_action`).
   Relabel; re-run.
6. **Second brand (SpotifyCares)** — 56% in-channel resolution vs Apple's ~35%,
   so the reply task isn't all "DM us". This is the real test of whether
   retrieval + drafting adds value, or this brand just doesn't suit it.
7. **Retrieval A/B** — BM25 vs MiniLM vs BM25+cross-encoder + a brand-filler
   stoplist ("@115858", "fix this"), scored by "did the drafter cite a
   precedent with the right intent". Today BM25 scores 18–112 on everything,
   which means it's matching filler.

---

## Appendix — reproduce
`README.md`. `make smoke` (no backend, 30 s) → `make eval` (replays the
committed cache, no key, ~2 min; `data/processed/` is committed so `make prep`
and the raw CSV are not needed) → `make worksheet` (hand-score) → `make judge`.
`make eval-cold` re-runs live (needs `GEMINI_API_KEY`). All thresholds in
`config.yaml`; the non-obvious calls in `DECISION_LOG.md`.
