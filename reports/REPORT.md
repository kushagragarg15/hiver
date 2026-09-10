# AI support agent — report

**Brand: AppleSupport** · Dataset: *Customer Support on Twitter* (Kaggle,
thoughtvector), Oct–Dec 2017 slice · Agent: intent → retrieval → grounded draft
→ escalate/auto.

> **Status of the numbers.** Golden set: **233 rows** (30 hand-labelled seed +
> 203 model-assisted drafts pending a human review pass — `SAMPLING_NOTES.md`).
> Runs, all committed with warm caches:
>
> | run | worker | judge | n | what it gives |
> |---|---|---|---|---|
> | **primary** | groq `gpt-oss-120b` | qwen3.8-27b | **150** | intent + escalation |
> | reply-quality | gemini `3.1-flash-lite` | (self) | 30 | LLM-as-judge table |
> | local baseline | ollama `llama3.1:8b` | (self) | 30 | model-capacity contrast |
>
> Every free LLM tier (Gemini 500 req/day, Groq 200k tok/day) capped a full
> 233-row + judged run; n=150 is where the primary run landed before the token
> cap, and the reply-quality judge numbers are from the earlier 30-row Gemini
> run. The judge's agreement with a human (κ) is **not yet measured**
> (`make judge`).
>
> **What the runs show:**
> 1. **A capable classifier clears the baselines.** `gpt-oss-120b` intent
>    macro-F1 **0.65** (n=150) vs keyword **0.49** and majority **0.08**;
>    `llama3.1:8b` managed only **0.32** on the seed — *below* keyword. Model
>    capacity is the deciding variable, not the method.
> 2. **The escalate/auto decision is the weak point.** `gpt-oss-120b`
>    missed-escalation **0.47** (n=150) — better than Gemini's 0.73 on the seed,
>    and the rule stack now measurably beats the intent-prior baseline (0.47 vs
>    0.51 missed) instead of collapsing onto it — but **47% of messages that
>    needed a human were auto-handled.** Not deployable (§4 F1/F2).
> 3. **The reply-quality "win" is a brand artefact.** On the 30-row Gemini run
>    the agent's judge pass-rate was 0.90 — but the canned "please DM us"
>    one-liner scored **0.93**. AppleSupport's real replies *are* deflections,
>    so the rubric rewards deflection (§3.3, §5).

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
(models are uniformly over-confident, BM25 always clears), so only the intent
categories, the regexes and the weak-precedent flag do any work — and the stack
misses 36–73% of true escalations depending on the worker model.*

Golden set: **233 rows** (30 hand-labelled seed + 203 model-assisted drafts,
review pending — `SAMPLING_NOTES.md`). `gold_action` = *should a good bot have
handled this unsupervised* — deliberately not "what Apple did".

---

## 3. Results vs baselines

Primary run: **`groq/gpt-oss-120b`, n = 150** (`results_groq-gptoss120b_n150.md`;
89 auto / 61 escalate on this slice). `gemini-3.1-flash-lite` (n=30) and
`llama3.1:8b` (n=30) shown for contrast (`results_*_n30`-equivalents).

### 3.1 Intent classification
| system (run) | accuracy | macro-F1 |
|---|---|---|
| **agent — gpt-oss-120b (n=150)** | **0.69** | **0.65** |
| agent — gemini-3.1-flash-lite (n=30) | 0.70 | 0.50 |
| agent — llama3.1:8b (n=30) | 0.37 | 0.32 |
| simple baseline — keyword weak-labeller (n=150) | 0.45 | 0.49 |
| trivial baseline — majority intent (n=150) | 0.49 | 0.08 |

Both capable models (`gpt-oss-120b`, `gemini`) clear the keyword and majority
baselines; `llama3.1:8b` does not. `gpt-oss-120b`'s 46/150 errors are
dominated by the genuinely fuzzy **software-vs-hardware fault** boundary
(`software_update_issue`↔`device_hardware_battery`, 11 cases — §4 F5) and
how-to↔fault confusion (9). Only 4 errors are the tone-driven
`→ complaint_churn_risk` slip that sank the weaker models. Same method — model
capacity is the deciding variable.

### 3.2 Escalate vs auto (positive class = escalate) — n=150
| system | esc-recall | **missed-esc ↓** | unnec-esc ↓ | auto-rate | esc-F1 |
|---|---|---|---|---|---|
| **agent — rule stack (gpt-oss-120b)** | 0.53 | **0.47** | 0.10 | 0.75 | 0.62 |
| simple — escalate by intent prior | 0.49 | 0.51 | 0.04 | 0.80 | 0.63 |
| trivial — always-auto | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| trivial — always-escalate | 1.00 | **0.00** | 1.00 | 0.00 | 0.52 |

**The escalate/auto decision is the weak point of the system.** Two things to
read here:
- The rule stack **does** beat the intent-prior baseline now (missed-esc 0.47 vs
  0.51) — the hard-regex patterns and the drafter's weak-precedent flag fire on
  ~10% of cases beyond the intent categories, trading a little unnecessary
  escalation (0.10 vs 0.04) for a little more caught. On the weaker Gemini run
  the stack collapsed *exactly* onto the prior (§ note below); a
  better-calibrated model lets the other rules matter.
- But **47% of the messages that needed a human were auto-handled anyway.** Every
  missed escalation (25 of them) was predicted at confidence 0.78–0.97, so the
  `min_intent_confidence: 0.60` floor never fires, and BM25 scores (21–61)
  always clear the `min_retrieval_score: 3.0` floor. The misses are
  systematically the categories no rule covers: hardware safety
  (`gold_0015` overheating, `gold_0082` bent phone, `gold_0057` logic board),
  recalls (`gold_0013`), pro-app routing (`gold_0022`/`0033` iMovie),
  disabled-device recovery (`gold_0068`), phishing (`gold_0108`), and
  distress/hostility over a real issue (`gold_0009`, `gold_0023`). See §4 F1/F2.

_Gemini n=30 for reference: missed-esc 0.73, bit-identical to the intent-prior
baseline — the safety was leaning entirely on the 3 always-escalate intent
categories. llama n=30: missed-esc 0.36, but only because its noisier confidence
tripped the 0.60 floor by luck._

### 3.3 Reply quality — LLM-as-judge (1–5)  *(gemini n=30 run only)*
The n=150 judge pass hit the free-tier token cap before completing; these are
from the earlier `gemini-3.1-flash-lite` run (n=30, judge = same model):

| system | groundedness | corr./safety | relevance | completeness | pass-rate |
|---|---|---|---|---|---|
| agent (retrieval-grounded) | 4.80 | 4.83 | 4.53 | 4.50 | 0.90 |
| trivial — canned "please DM us" | **4.90** | **4.90** | 3.90 | 3.90 | **0.93** |
| simple — retrieval-only (echo top precedent) | 4.10 | 4.20 | 3.83 | 3.73 | 0.70 |

**The canned one-liner beats the agent on groundedness, safety and pass-rate.**
Every agent reply in the run is a variant of *"we'd like to help — please DM us
your iOS version"*, because that's what ~65% of the retrieved precedents say. A
fixed string is the purest safe deflection, and the rubric rewards deflection.
The agent edges ahead only on *relevance* / *completeness*. **On AppleSupport the
retrieval + drafting pipeline is close to dead weight over a constant string** —
a fact about the brand's 2017 channel strategy, measured directly by the
baseline. (The n=150 `gpt-oss-120b` replies inspected by hand look the same —
all DM redirects.)

### 3.4 Is the judge trustworthy?  *(open — and load-bearing)*
Not measured. `make worksheet` → hand-score 36 rows → `make judge` →
per-dimension exact/within-1/Spearman + **Cohen's κ on `overall_pass`**. Bar:
**κ ≥ 0.6 and safety within-1 ≥ 0.8**. The n=30 judge above graded its own
model's output; the n=150 primary run at least uses a *different* judge model
(`qwen3.8-27b` vs worker `gpt-oss-120b`), but with no human anchor yet, §3.3 is
directional only.

---

## 4. Top 5 failure modes

From `reports/agent_golden_preds_groq-gptoss120b.jsonl` (n=150, worker
`gpt-oss-120b`), with the Gemini/llama seed runs cited where they diverge.

### F1 — the confidence and BM25 floors never fire, so the escalation stack barely reaches past its intent-category rules
- **Mechanism.** Auto-handle needs confidence ≥ 0.60 **and** BM25 ≥ 3.0 **and**
  intent ∉ always-escalate. Across all three models the classifier reports
  confidence **0.78–0.99** on essentially everything, and AppleSupport's shared
  filler ("@115858", "fix this", "ever since the update") keeps BM25 at
  **21–61** — so neither floor ever triggers. What actually escalates: the 3
  always-escalate intents, five hard-regex patterns, and the drafter's
  weak-precedent flag.
- **Evidence (n=150).** 25 missed escalations, **every one** at confidence
  0.78–0.97. On the weaker Gemini seed run the stack was *bit-identical* to the
  intent-prior baseline; `gpt-oss-120b` only escapes that because its
  calibration lets the regex/weak-precedent rules matter at all (missed-esc
  0.47 vs prior 0.51).
- **Fix.** Escalation must not gate on a raw confidence scalar. Calibrate it
  against held-out labels first; or replace the whole stack with a learned
  P(escalate) head (§6); or make the LLM escalation-veto (`use_llm_check`,
  currently off) mandatory, with the precedents in context so it can judge "is
  there actually a safe answer here".

### F2 — no rule for hardware-safety / recall / prior-support-failed / image-only
- **Mechanism.** Even with a working gate, nothing in the stack encodes these
  "needs a human regardless of confidence" cases.
- **Evidence (n=150 misses).**
  `gold_0015` iPhone X **overheating** + slow charge → auto (conf 0.93);
  `gold_0082` "iPhone 7 Plus appears to have **bent**" → auto (conf 0.97);
  `gold_0057` MacBook **logic-board** repair → auto (conf 0.97);
  `gold_0013` Apple TV **under recall**, prior support failed → auto (conf 0.88);
  `gold_0022`/`gold_0033` **iMovie** workflow (specialist-routed) → auto;
  `gold_0068` **disabled** iPhone recovery → auto;
  `gold_0108` suspected **phishing** email → auto;
  `gold_0014`/`gold_0067` "how can I fix this? [image only]" → auto.
- **Fix.** A short forced-escalate lexicon:
  `overheat|swollen|burning|smoke|bent`; `recall`;
  `already (called|contacted|emailed)|case ?number`;
  `replace(d|ment)`; `logic board|water damage`; and "opening < 8 words AND
  contains a media URL".

### F3 — churn risk rides along with a real technical intent and gets auto-handled
- **Mechanism.** A capable model (correctly, vs my labels) keeps the *technical*
  intent for angry-but-concrete messages — but then `software_update_issue` +
  high confidence → auto and the churn signal is lost. (The weaker models did
  the opposite: mislabelled these `complaint_churn_risk` and over-escalated.)
- **Evidence.** `gold_0009` "FIX MY PHONE … YOU SUCK SO MUCH … I PAY MY MONEY"
  → `software_update_issue` / auto (conf 0.88). `gold_0023` "SICK AND TIRED …
  **DONE WITH YOU** … switch to Samsung" → `software_update_issue` / auto.
- **Fix.** Emit `churn_risk` as a separate boolean beside the intent and force
  escalate on it — so recognising the battery issue and recognising the
  relationship risk aren't in tension.

### F4 — the "weak precedent → escalate" rule mostly fires on trivial messages
- **Mechanism.** When BM25 finds nothing strong the drafter flags
  `used_weak_precedent` and the stack escalates. But the messages with no good
  precedent are disproportionately *thanks / opinions / niche how-tos* —
  exactly the ones a bot should close cheaply, not escalate.
- **Evidence (n=150).** 6 of the 10 unnecessary escalations are this rule on
  `gold_0041` (a thank-you), `gold_0075` (a thank-you), `gold_0056`
  (off-topic), `gold_0076` (an opinion), `gold_0035` (gloves + Touch ID
  how-to), `gold_0093` (AppleCare transfer — a documented self-serve process).
- **Fix.** Don't escalate on weak precedent alone; require it to co-occur with a
  non-trivial intent. For `praise_or_non_actionable`, auto-acknowledge
  regardless.

### F5 — the software-vs-hardware fault boundary is genuinely underdetermined
- **Evidence.** The single biggest intent-error cluster at n=150:
  `software_update_issue` ↔ `device_hardware_battery`, **11 cases**.
  Battery-drain / slowdown / black-screen messages with no explicit "since the
  update" get split unpredictably — and often *my* label is a hindsight guess
  (2017 context = "it was the iOS 11 bug") the live text doesn't support.
- **Fix.** Merge the two fault buckets for v1 — their `default_action` is
  identical (auto + triage) — and only re-split if a downstream routing target
  needs it. Part of the intent macro-F1 gap is penalising a distinction the
  data doesn't carry.

---

## 5. What is misleading about my headline number? *(mandatory)*

The line that looks like a win is **"`gpt-oss-120b` agent: intent macro-F1 0.65,
beats every baseline; missed-escalation 0.47, better than the intent-prior
baseline."** Why not to trust it:

1. **"Better than the intent-prior baseline" is a low bar, and "auto-rate 0.75"
   is not automation — it's a rubber stamp.** The same run auto-handles **47%
   of the messages that needed a human** (§3.2, §4 F1). The only genuinely safe
   system in the table is always-escalate, which automates nothing. A number
   like "auto-rate 0.75" quoted without the missed-escalation rate next to it is
   the misleading framing.
2. **The escalation result is not portable across models.** Same code, same
   golden protocol: `gpt-oss-120b` misses 47%, Gemini misses 73%, llama misses
   36%. llama "wins" only because its noisier confidence trips a threshold by
   luck (§4 F1). Any missed-escalation rate is meaningless without the exact
   worker model and its calibration.
3. **Reply quality is confounded by the brand, and beaten by a constant
   string.** Canned "please DM us" scores judge pass-rate **0.93** vs the
   agent's 0.90 (§3.3). AppleSupport's real replies are deflections, so the
   rubric rewards deflection. The retrieval + drafting pipeline is near
   dead-weight on this brand — I should have run the canned baseline first and
   seen that immediately.
4. **The judge has no human anchor.** κ is unmeasured (§3.4). The n=150 primary
   run at least uses a different judge model from the worker; the reply-quality
   table is still from the n=30 run where they were the same model.
5. **n = 150, one annotator, and 203 of the 233 labels are model-assisted
   drafts** not yet human-reviewed (`SAMPLING_NOTES.md`). 95% CI on the
   missed-escalation rate at n=150 (53 positives) is roughly ±0.13. `gold_action`
   is *my* call on what's automatable, made with hindsight (`reference_resolution`
   visible) the live agent never has.
6. **The intent macro-F1 is partly penalising a bad taxonomy.** ~1/4 of the
   errors are the `software_update_issue`↔`device_hardware_battery` boundary
   (§4 F5), which the text often genuinely doesn't determine. Merging those two
   buckets would move macro-F1 up without the model getting any better.
7. **Majority-intent baseline is fit on the golden label distribution** (peeks
   at test priors), so "agent beats majority" understates the gap — the
   baseline column isn't a clean floor.
8. **Free-tier caps shaped the evidence.** Gemini (500 req/day) and Groq
   (200k tok/day) each capped a full 233-row judged run; n=150 is where the
   primary run stopped, not a chosen sample size, and there is no n=150 judge
   pass. Reproducing the exact numbers depends on the committed `.llm_cache/`;
   Gemini model ids also churned mid-session.
9. **Frozen 2017 world + opening-message only.** Grounding is in iOS-11-era
   precedent; real resolution takes the whole thread. Both inflate any
   "helpfulness" read.

**Bottom line.** The one solid, portable result is narrow: **a capable
classifier beats the keyword and majority baselines on intent (macro-F1 ~0.5–0.65
vs 0.49 / 0.08), and a weak local model does not.** The escalate/auto decision as
built is not deployable — it misses 36–73% of true escalations depending on the
model, because it gates on an uncalibrated confidence scalar and has no rules for
the categories that actually need a human. Reply quality can't be judged
meaningfully on this brand until there's a non-deflecting baseline and a
human-anchored judge.

---

## 6. One more week (in priority order)
1. **Redesign the escalation decision** (the actual broken thing). Stop gating
   on the raw confidence scalar. Minimum: (a) forced-escalate lexicon from §4 F2;
   (b) churn-risk boolean beside the intent (F3); (c) turn on the LLM
   escalation-veto with precedents in context and make it mandatory. Then a
   learned P(escalate) head — logistic regression on [intent one-hot,
   *calibrated* confidence, BM25 top-k, msg length, sentiment, lexicon hits] —
   once labels allow. Target missed-escalation < 0.05 at auto-rate ≥ 0.5.
2. **Finish the golden set** — label all 233 candidates (currently 30); second
   annotator on 100 rows for a real κ. Per-class F1 and the escalation slices
   are anecdotes until then.
3. **Judge trust** — `make judge` on a hand-scored worksheet; if κ < 0.6,
   2-shot the rubric with a worked pass + fail, use a *different* model family
   as judge (not the one being graded), add an adversarial "confident
   fabrication" probe (F4).
4. **Re-baseline reply quality honestly** — the current baselines are too weak
   for this brand. Add: (a) "same model, summarise the precedent, no
   anti-fabrication rules"; (b) "always send the canned line" as the *primary*
   comparison, since it already wins. If the agent can't clearly beat the canned
   line on a brand-appropriate metric, RAG isn't earning its place here.
5. **Collapse the fuzzy intents** (F5): merge software/hardware fault into one
   bucket (identical `default_action`); split `billing` into dispute-vs-howto.
6. **Second brand (SpotifyCares)** — 56% in-channel resolution vs Apple's ~35%,
   so the reply task isn't all "DM us". This is the real test of whether the
   pipeline adds value, or just this brand doesn't suit it.
7. **Retrieval A/B** — BM25 vs MiniLM vs BM25+cross-encoder + a brand-filler
   stoplist, scored by "did the drafter cite a precedent with the right intent".
8. **Cost/latency table** per backend (gemini-flash-lite / llama3.1:8b /
   gpt-4o-mini) at the chosen operating point.

---

## Appendix — reproduce
`README.md`. `make smoke` (no backend) → `make prep` → `make eval` →
`make worksheet` (hand-score) → `make judge`. All thresholds in `config.yaml`;
the non-obvious calls in `DECISION_LOG.md`.
