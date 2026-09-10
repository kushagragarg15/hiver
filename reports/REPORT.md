# AI support agent — report

**Brand: AppleSupport** · Dataset: *Customer Support on Twitter* (Kaggle,
thoughtvector), Oct–Dec 2017 slice · Agent: intent → retrieval → grounded draft
→ escalate/auto.

> **Status of the numbers.** The figures below are from a **real run on
> `ollama/llama3.1:8b`** (worker *and* judge), over the **30-row seed golden
> set**, 27 min wall-clock (`reports/results.json`, `meta.provider ==
> "ollama"`). They are a small-n, weak-model baseline. Two things are still
> open: (a) the golden set needs expanding to 150–250 (233 candidates are
> staged; `python -m eval.label_tool`), and (b) the judge is itself an 8B model
> — `make judge` (needs a hand-scored `judge_human_scores.csv`) will report the
> human-agreement κ. §5 is written around exactly these limitations.
>
> **Headline finding: on this brand + model the agent is not yet trustworthy.**
> Intent macro-F1 (0.32) is *below* the keyword baseline (0.34); missed-
> escalation rate is 0.36. The one clear win is reply quality — the grounded
> drafter's judge pass-rate (0.60) is 6× the retrieval-only baseline (0.10).

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

Golden set: **[N]** hand-labelled held-out messages (seed of 30 committed;
protocol + sampling in `data/golden/SAMPLING_NOTES.md`). `gold_action` = *should
a good bot have handled this unsupervised* — deliberately not "what Apple did".

---

## 3. Results vs baselines

`ollama/llama3.1:8b`, golden **n = 30** (seed; 19 auto / 11 escalate). Raw
tables + per-class breakdown + confusion matrix: `reports/results.md`.

### 3.1 Intent classification
| system | accuracy | macro-F1 |
|---|---|---|
| agent (LLM few-shot) | 0.367 | **0.315** |
| simple baseline — keyword weak-labeller | 0.500 | **0.341** |
| trivial baseline — majority intent (`software_update_issue`) | 0.600 | 0.094 |

**The 8B agent loses to the keyword baseline on macro-F1 and to the majority
baseline on accuracy.** With 60% of the seed set being `software_update_issue`,
"always guess the majority" is hard to beat on accuracy, and the model's habit
of splitting that class into `complaint_churn_risk` / `how_to_feature_question`
/ `device_hardware_battery` (see §4) tanks both its accuracy and the rare-class
recall that macro-F1 rewards. This is the single most important number to fix
(§6) and the strongest argument for a stronger worker model.

### 3.2 Escalate vs auto (positive class = escalate)
| system | esc-recall | **missed-esc-rate ↓** | unnec-esc-rate ↓ | auto-rate |
|---|---|---|---|---|
| agent (rule stack) | 0.636 | **0.364** | 0.053 | 0.733 |
| trivial — always-auto | 0.00 | 1.00 | 0.00 | 1.00 |
| trivial — always-escalate | 1.00 | **0.00** | 1.00 | 0.00 |
| simple — escalate by intent prior | 0.545 | 0.455 | 0.053 | 0.767 |

The agent beats the intent-prior baseline (0.36 vs 0.46 missed-escalation at a
similar auto-rate) but **0.36 missed-escalation is not production-safe** — 4 of
11 true escalations were auto-handled (§4 F1). The always-escalate baseline is
the only "safe" system here and it automates nothing; closing that gap is the
whole game.

### 3.3 Reply quality — LLM-as-judge (1–5, n = 30)
| system | groundedness | correctness/safety | relevance | completeness | pass-rate |
|---|---|---|---|---|---|
| agent (retrieval-grounded) | **3.27** | **4.47** | **4.13** | **3.37** | **0.60** |
| simple — retrieval-only (echo top precedent) | 2.23 | 3.10 | 3.00 | 2.17 | 0.10 |
| trivial — canned "please DM us" | 2.03 | 3.93 | 2.53 | 1.80 | 0.03 |

**This is the agent's clear win.** Grounding the drafter in retrieved precedents
lifts the judge pass-rate 6× over echoing the top precedent verbatim and ~18×
over the canned line, driven by groundedness and completeness. The lift is real
*despite* Apple's precedents being mostly "DM us + which iOS version" — the
model reuses the diagnostic questions and known workarounds embedded in those
threads rather than the DM redirect. Caveat: the judge is also `llama3.1:8b`
(§3.4).

### 3.4 Is the judge trustworthy?  *(open)*
Not yet measured — needs a hand-scored `data/golden/judge_human_scores.csv`
(`make worksheet` emits 36 rows to score, `make judge` computes it).
`reports/judge_agreement.json` will report per-dimension exact / within-1 /
Spearman and **Cohen's κ on `overall_pass`**. Bar for using the judge in the
headline: **κ ≥ 0.6 and correctness/safety within-1 ≥ 0.8**. Until then §3.3 is
**directional only** — an 8B model grading 8B-model output is the weakest link
in this report.

---

## 4. Top 5 failure modes

From `reports/agent_golden_preds.jsonl` (n=30). Counts are small — treat as
directions, not rates.

### F1 — "confident intent + high BM25 score" is treated as "safe to auto", but neither implies the issue is bot-resolvable
- **Mechanism.** The rule stack auto-handles when intent-confidence ≥ 0.60 and
  top BM25 ≥ 3.0 and the intent isn't on the always-escalate list. BM25 scores
  on AppleSupport run 20–50 because of shared filler ("@115858", "fix this",
  "ever since the update"), so the score floor is essentially always cleared,
  and the model is over-confident (most predictions 0.8–0.95). Nothing in the
  stack encodes *hardware safety*, *recall handling*, *prior support already
  failed*, or *the problem is in an image the agent can't see*.
- **Evidence (all 4 missed escalations):**
  - `gold_0015` "why is my iPhone X **over heating** … takes so long to charge" →
    auto (conf 0.80, BM25 42). Overheating + charging fault is a
    hardware-safety signal that should always get eyes on it.
  - `gold_0022` "#iMovie … videos will not become files on my external hard
    drives" → auto (conf 0.95, BM25 50). Apple historically routes pro-app
    workflow issues to specialists.
  - `gold_0013` "Faulty 3rd Gen Apple TV, **under recall** … phone support
    doesn't even acknowledge recall" → auto (conf 0.80, BM25 41). Recall +
    a failed prior support contact.
  - `gold_0014` "how can I fix this? [image only]" → auto (conf 0.90, BM25 21).
    The agent can't see the attachment.
- **Cheap fix.** Add soft-signal rules that force escalate: a
  hardware-safety lexicon (`overheat|swollen|burning|spark|smoke`); the tokens
  `recall|already (called|contacted)|case number`; and "opening is < N words
  AND contains a media URL". Down-weight BM25 by penalising matches that are
  only on a brand-filler stoplist.

### F2 — angry-but-concrete update complaints get pulled into `complaint_churn_risk`
- **Mechanism.** The taxonomy says "use `complaint_churn_risk` only when there
  is no concrete technical ask", but the model weights tone over content.
- **Evidence.** `gold_0009` (music deleted — "YOU SUCK SO MUCH"), `gold_0010`
  (sarcastic voicemail complaint), `gold_0023` ("SICK AND TIRED … DONE WITH
  YOU") all predicted `complaint_churn_risk`; gold intent is
  `software_update_issue`. This both hurts intent macro-F1 and — because
  `complaint_churn_risk` is always-escalate — caused the one *unnecessary*
  escalation (`gold_0010`, gold action `auto`).
- **Hypothesis.** "Primary need" is genuinely ambiguous when a real issue is
  wrapped in a churn threat; my golden labels resolve it toward the technical
  bucket, the model toward affect. A single annotator can't adjudicate this.
- **Cheap fix.** Make `intent` and `escalation-trigger` separate outputs: let
  the classifier emit `software_update_issue` + a `churn_risk` boolean flag;
  route on the flag without distorting the intent label.

### F3 — `software_update_issue` vs `device_hardware_battery` vs `how_to_feature_question` boundaries are too fuzzy for an 8B model
- **Evidence.** Battery-drain messages with no explicit "since the update"
  (`gold_0004`, `gold_0007`, `gold_0019`) → predicted `device_hardware_battery`,
  gold `software_update_issue` (2017 context: it *was* the iOS 11 bug).
  "I️" keyboard-bug messages phrased as "why does X / fix this"
  (`gold_0008`, `gold_0025`, `gold_0027`) → predicted `how_to_feature_question`.
- **Hypothesis.** These require either era knowledge (iOS 11 = known battery
  bug) or a firm "a fault report is not a how-to" rule the model doesn't hold.
- **Cheap fix.** Merge `software_update_issue` + `device_hardware_battery` into
  one `device_or_software_fault` bucket for v1 (the *action* is the same for
  both — auto with triage); keep the split only if a downstream routing target
  actually needs it. Add a decision rule: "describes something broken →
  fault bucket, never how-to".

### F4 — the drafter inserts support URLs that weren't in any precedent
- **Mechanism.** The prompt says "at most one link, only if a precedent used
  one", but the model pattern-matches "iPhone slow → link to a support
  article".
- **Evidence.** `gold_0003` reply: *"Try restarting your device and testing it:
  https://support.apple.com/en-us/HT201542"* — no retrieved precedent contained
  that URL. (It happens to be a real article, which is worse — it's plausible
  enough to pass a skim.)
- **Cheap fix.** Post-process: strip any URL from the draft that does not
  appear verbatim in a retrieved precedent; or hard-fail the draft and escalate
  if it contains an unseen link.

### F5 — `connectivity_sync` is a magnet for any Wi-Fi/Bluetooth token
- **Evidence.** `gold_0001` (AirPods won't pair), `gold_0006` (can't download
  apps on cellular), `gold_0011` (Bluetooth keyboard + failed OS install),
  `gold_0016` (typing lag) all predicted `connectivity_sync` on a single
  keyword; none is primarily a connectivity issue.
- **Cheap fix.** Add contrastive few-shots to the classifier prompt (AirPods
  pairing = hardware; "download without Wi-Fi" = how-to/settings); or a
  second-stage check "is connectivity the *cause* or just *mentioned*?".

---

## 5. What is misleading about my headline number? *(mandatory)*

The one number that looks like a win is **"grounded drafter pass-rate 0.60 vs
0.10 retrieval-only — 6× better."** Everything wrong with taking that at face
value:

1. **The judge is `llama3.1:8b` grading `llama3.1:8b` output, and its trust is
   unmeasured** (§3.4, κ not yet computed). Same-family judges reward fluent,
   confident, on-topic text — which is exactly what the same-family drafter
   produces. The 6× gap partly measures "the agent writes in the judge's
   preferred style", not "the agent is 6× more helpful". Until κ ≥ 0.6 this is
   not a headline.
2. **n = 30, one annotator, one pass.** No inter-annotator agreement yet. 95%
   CI on a 0.60 pass-rate at n=30 is roughly ±0.18. Per-intent slices (billing
   n=1, account n=1, connectivity n=1 in the seed set) are anecdotes, not
   rates. The intent macro-F1 gap vs the keyword baseline (0.315 vs 0.341) is
   inside the noise.
3. **`gold_action` encodes *my* opinion of what's automatable, not Apple's.**
   I labelled "known iOS 11 bug + standard workaround" as `auto`; Apple
   escalated those to DM (~65% of its real replies are "DM us"). So a model that
   *imitated the brand* scores badly against my labels, and an *over-eager*
   model scores well while being risky in production. The agent's 0.73 auto-rate
   only means something next to that definition — and the 0.36 missed-escalation
   rate says the over-eager failure is already happening (§4 F1).
4. **The reply win is inflated by weak baselines on this brand.** "Retrieval-
   only" echoes the first brand turn of the top precedent, which for Apple is
   often a bare "DM us" fragment — trivially easy to beat. A fairer baseline
   (retrieve, then have the *same model* summarise the precedent without the
   anti-fabrication rules) would close much of the gap.
5. **Majority-intent baseline is fit on the golden label distribution** — it
   peeks at test priors, so any "agent beats majority" framing *understates*
   the real gap (and here the agent loses to it on accuracy anyway).
6. **Frozen 2017 world.** "How the brand resolves things" = 2017 policies, iOS
   11, support URLs. On today's traffic groundedness would fall and nothing in
   this eval would catch it. F4 (invented `HT201542` link) is a preview.
7. **The agent is slow enough that the eval sample is self-limiting.** 27 min
   for 30 rows on CPU → the full 233-candidate set is ~3.5 h per run, so
   iteration pressure pushes toward small n, which pushes toward noisy numbers.
   A hosted model removes this and is the honest next step.
8. **Opening-message only.** Real resolution needs the whole thread; single-turn
   scoring overstates end-to-end usefulness.

**Bottom line:** the only defensible claim today is *"retrieval-grounding the
drafter helps reply quality on a small sample, judged by a model of unknown
reliability."* Intent and escalation are not yet at a bar I'd deploy.

---

## 6. One more week (in priority order)
1. **Finish the golden set** — label all 233 candidates (currently 30), so
   per-class F1 and the escalation slices stop being anecdotes; add a second
   annotator on 100 rows for a real κ.
2. **Re-run on `gpt-4o-mini`** as worker + `gpt-4o` as judge, keep the
   `llama3.1:8b` numbers as a *local-vs-hosted* comparison row. The intent
   result (agent < keyword baseline) is very likely a model-capacity problem,
   not a method problem — this settles it in ~5 min / ~$1.
3. **Judge trust** — `make judge` on a hand-scored worksheet; if κ < 0.6,
   2-shot the rubric with a worked pass + fail, ensemble two judge models, add
   an adversarial "confident fabrication" probe (F4-style).
4. **Fix the escalation stack** per §4 F1: hardware-safety lexicon, recall /
   "already contacted" tokens, image-only rule, BM25 filler-penalty. Target
   missed-escalation < 0.05 without dropping auto-rate below ~0.5.
5. **Collapse the fuzzy intents** (F3): merge software/hardware fault into one
   bucket for v1; split `billing` into dispute-vs-howto; move churn-risk to a
   boolean flag beside the intent (F2).
6. **Learned escalation head** once labels allow — logistic regression on
   [intent one-hot, confidence, BM25 top-k, msg length, sentiment, regex hits] →
   calibrated P(escalate); hard rules stay as overrides.
7. **Retrieval A/B** — BM25 vs MiniLM vs BM25+cross-encoder, scored by "did the
   drafter cite a precedent with the right intent"; add the filler stoplist.
8. **Second brand (SpotifyCares)** where in-channel resolution is real, so the
   reply task isn't dominated by "DM us"; compare what each brand's data lets
   you automate. Plus a cost/latency table per backend at the operating point.

---

## Appendix — reproduce
`README.md`. `make smoke` (no backend) → `make prep` → `make eval` →
`make worksheet` (hand-score) → `make judge`. All thresholds in `config.yaml`;
the non-obvious calls in `DECISION_LOG.md`.
