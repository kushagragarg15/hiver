# AI support agent — report

**Brand: AppleSupport** · Dataset: *Customer Support on Twitter* (Kaggle,
thoughtvector), Oct–Dec 2017 slice · Agent: intent → retrieval → grounded draft
→ escalate/auto.

> **Status of the numbers.** Two real runs over the **30-row seed golden set**,
> both committed: `gemini/gemini-3.1-flash-lite` (primary — `reports/results.json`,
> `results_gemini-flash-lite.*`) and `ollama/llama3.1:8b` (local baseline —
> `results_llama3.1-8b.*`). Same harness, same golden set, worker = judge in
> both. Still open: (a) expand the golden set to 150–250 (233 candidates staged,
> `python -m eval.label_tool`); (b) the judge is the same model it grades — κ vs
> a human is unmeasured (`make judge`). §5 is built around these.
>
> **Three findings, none of them the number you'd put on a slide:**
> 1. **A capable classifier beats the baselines; a weak one doesn't.** Gemini
>    intent macro-F1 **0.50** (acc 0.70) clears keyword (0.34) and majority
>    (0.60 acc); llama3.1:8b scored **0.32** — *below* the keyword baseline.
> 2. **The better classifier made escalation *worse*.** Gemini missed-escalation
>    **0.73** vs llama's 0.36 — and Gemini's decision is now *identical* to the
>    intent-prior baseline. The rule stack was silently relying on the weak
>    model's under-confidence to do its safety work (§4 F1).
> 3. **The reply-quality "win" is an artefact of the brand.** Gemini agent
>    reply pass-rate 0.90 — but the canned "please DM us" one-liner scores
>    **0.93**. AppleSupport's real resolutions *are* deflections, so a constant
>    string is "grounded and safe" by the rubric (§3.3, §5).

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
*Result: §4 F1 shows the confidence and BM25 floors never fire in practice, so
this stack currently reduces to the always-escalate-intents check.*

Golden set: **30** hand-labelled held-out messages (seed; 233 candidates staged;
protocol + sampling in `data/golden/SAMPLING_NOTES.md`). `gold_action` = *should
a good bot have handled this unsupervised* — deliberately not "what Apple did".
Backends run: `gemini-3.1-flash-lite` (primary) and `llama3.1:8b` (local
baseline); both committed with warm caches.

---

## 3. Results vs baselines

Golden **n = 30** (seed; 19 auto / 11 escalate). Primary =
`gemini-3.1-flash-lite`; `llama3.1:8b` alongside as a local-model baseline. Same
harness, worker = judge in both. Per-class breakdown + confusion matrix:
`reports/results_gemini-flash-lite.md`, `reports/results_llama3.1-8b.md`.

### 3.1 Intent classification
| system | accuracy | macro-F1 |
|---|---|---|
| **agent — gemini-3.1-flash-lite** | **0.70** | **0.50** |
| agent — llama3.1:8b | 0.37 | 0.32 |
| simple baseline — keyword weak-labeller | 0.50 | 0.34 |
| trivial baseline — majority intent | 0.60 | 0.09 |

Gemini clears both baselines; **llama3.1:8b does not** (below keyword on
macro-F1, below majority on accuracy). With 60% of the seed set being
`software_update_issue`, the weak model splits that class on *tone*
(→ `complaint_churn_risk`) or *phrasing* (→ `how_to_feature_question`) and
sinks. Gemini's 9/30 errors are mostly the genuinely fuzzy software-vs-hardware
battery boundary (§4 F5). Same method — model capacity is decisive.

### 3.2 Escalate vs auto (positive class = escalate)
| system | esc-recall | **missed-esc-rate ↓** | unnec-esc ↓ | auto-rate |
|---|---|---|---|---|
| **agent — gemini-3.1-flash-lite** | 0.27 | **0.73** | 0.00 | 0.90 |
| agent — llama3.1:8b | 0.64 | **0.36** | 0.05 | 0.73 |
| simple — escalate by intent prior (gemini) | 0.27 | 0.73 | 0.00 | 0.90 |
| trivial — always-auto | 0.00 | 1.00 | 0.00 | 1.00 |
| trivial — always-escalate | 1.00 | **0.00** | 1.00 | 0.00 |

**The headline failure of this project.** With Gemini the agent's escalation
decision is *bit-for-bit identical* to the intent-prior baseline — the rule
stack contributes nothing. All 8 missed escalations were predicted at
confidence 0.85–0.95, so the `min_intent_confidence: 0.60` floor never fires;
BM25 scores (21–50) always clear the `min_retrieval_score: 3.0` floor; none hit
an always-escalate intent. So the only things that escalate are the 3
always-escalate intent categories. llama scored **better** here (0.36) *only*
because it was under-confident enough to trip the floor ~5 extra times — the
safety was accidental, not designed (§4 F1/F2).

### 3.3 Reply quality — LLM-as-judge (1–5, n = 30)
| system | groundedness | corr./safety | relevance | completeness | pass-rate |
|---|---|---|---|---|---|
| agent — gemini-3.1-flash-lite | 4.80 | 4.83 | 4.53 | 4.50 | 0.90 |
| trivial — canned "please DM us" | **4.90** | **4.90** | 3.90 | 3.90 | **0.93** |
| simple — retrieval-only (echo top precedent) | 4.10 | 4.20 | 3.83 | 3.73 | 0.70 |

**The canned one-liner beats the agent on groundedness, safety and pass-rate.**
Every agent reply in the run is a variant of *"we'd like to help — please DM us
your iOS version"* (`agent_golden_preds_gemini-flash-lite.jsonl`), because
that's what ~65% of the retrieved precedents say. The rubric rewards a safe,
on-brand deflection and a fixed string is the purest one. The agent edges ahead
only on *relevance* / *completeness* (it names the symptom, asks one diagnostic
question). **On AppleSupport the retrieval + drafting pipeline is close to dead
weight over a constant string** — a fact about the brand's channel strategy,
measured directly by the baseline. (llama numbers: agent 0.60 pass-rate vs
canned 0.03 — the weak drafter *did* beat canned, but only because its replies
were worse in a way this same-family judge penalised; see §5.)

### 3.4 Is the judge trustworthy?  *(open — and load-bearing)*
Not measured. The judge is the *same model* grading its own output and it just
scored a canned deflection 4.9/5 on groundedness — plausibly right by the
rubric, plausibly leniency; can't tell without a human. `make worksheet` →
hand-score 36 rows → `make judge` → per-dimension exact/within-1/Spearman +
**Cohen's κ on `overall_pass`**. Bar: **κ ≥ 0.6 and safety within-1 ≥ 0.8**.
Until then §3.3 is directional only.

---

## 4. Top 5 failure modes

From `reports/agent_golden_preds_gemini-flash-lite.jsonl` (n=30). Counts are
small — directions, not rates.

### F1 — the escalation rule stack is hollow: it collapses to "escalate iff intent ∈ {account, billing, complaint}"
- **Mechanism.** Auto-handle requires confidence ≥ 0.60 **and** BM25 ≥ 3.0 **and**
  intent ∉ always-escalate. Gemini reports confidence 0.85–1.0 on essentially
  everything, so the confidence floor never fires. AppleSupport openings share
  heavy filler ("@115858", "fix this", "ever since the update") so BM25 runs
  21–50 and the score floor never fires. What's left is just the
  intent-category check → the decision is **bit-identical to the intent-prior
  baseline** (§3.2).
- **Evidence.** All 8 missed escalations: `gold_0002 0011 0013 0014 0015 0022
  0023 0026`, every one at confidence 0.85–0.95, BM25 21–50.
- **Why llama looked better.** llama's confidence was noisier and lower, tripping
  the 0.60 floor on ~5 more messages — so its 0.36 missed-rate was luck, not
  design. **A floor on a model's self-reported confidence is not a safety
  mechanism.**
- **Fix.** Escalation must not depend on the classifier's confidence scalar.
  Options: calibrate confidence against held-out labels before using it as a
  gate; replace it with a learned P(escalate) head (§6); or make the LLM
  escalation-veto (`use_llm_check`, currently off) mandatory and give it the
  precedents so it can judge "is there actually a safe answer here".

### F2 — no rule for hardware-safety / recall / "already contacted support" / image-only messages
- **Mechanism.** Even with a working confidence gate, nothing in the stack
  encodes these "needs a human regardless" cases.
- **Evidence.**
  - `gold_0015` "iPhone X **overheating** … takes so long to charge" → auto.
    Overheating is a safety signal.
  - `gold_0013` "Apple TV **under recall** … phone support doesn't even
    acknowledge the recall" → auto. Recall + a failed prior contact.
  - `gold_0022` "#iMovie … videos won't become files on my external drives" →
    auto. Pro-app workflow, historically specialist-routed.
  - `gold_0014` "how can I fix this? [image only]" → auto. The agent can't see
    the attachment.
  - `gold_0002` "I was sitting in a meeting … there are **witnesses** … I just
    want it **replaced**" → auto. A device-replacement claim.
- **Fix.** A short forced-escalate lexicon: `overheat|swollen|burning|smoke`;
  `recall|already (called|contacted|emailed)|case ?number|reference ?number`;
  `replace(d|ment)|refund`; and "opening < 8 words AND contains a media URL".

### F3 — churn risk rides along with a real technical intent and gets auto-handled
- **Mechanism.** Gemini (correctly, vs my labels) keeps the technical intent for
  angry-but-concrete messages — but then `software_update_issue` +
  high-confidence → auto, and the churn signal is lost.
- **Evidence.** `gold_0023` "SICK AND TIRED … **DONE WITH YOU** … want to switch
  to Samsung" → `software_update_issue` / auto. `gold_0026` "Don't ever restore
  your phone … **you suck**" → `software_update_issue` / auto.
- **Fix.** Emit `churn_risk` as a separate boolean beside the intent and force
  escalate on it, so recognising the battery issue and recognising the
  relationship risk aren't in tension.

### F4 — the drafter invents support URLs not in any precedent
- **Evidence.** `llama` run, `gold_0003`: *"…testing it:
  https://support.apple.com/en-us/HT201542"* — no retrieved precedent contained
  that URL (it's a real article, which is worse — it passes a skim). Gemini did
  this less but still emits `https://t.co/GDrqU22YpT` (Apple's real DM link)
  even on messages whose precedents didn't include it.
- **Fix.** Post-process: delete any URL from the draft that doesn't appear
  verbatim in a retrieved precedent; escalate if the model insists.

### F5 — the software-vs-hardware fault boundary is genuinely underdetermined
- **Evidence.** Gemini's 9 intent errors are mostly here: battery-drain
  messages with no explicit "since the update" split unpredictably between
  `software_update_issue` and `device_hardware_battery` (`gold_0004 0007
  0019`), and both my label and the model are guessing — the *text* doesn't
  say. AirPods pairing (`gold_0001`) landed in `connectivity_sync`.
- **Fix.** Merge the two fault buckets for v1 (their `default_action` is
  identical — auto with triage); only re-split if a downstream routing target
  needs it. The intent macro-F1 number is partly penalising a distinction the
  data doesn't support.

---

## 5. What is misleading about my headline number? *(mandatory)*

The number that looks like a win is **"Gemini agent: intent macro-F1 0.50 (beats
every baseline), reply pass-rate 0.90."** Why that is misleading:

1. **The reply pass-rate is beaten by a hard-coded string.** Canned "please DM
   us" scores 0.93 pass-rate and 4.9/5 groundedness (§3.3). AppleSupport's real
   resolutions *are* deflections, so the rubric — correctly — rates a safe
   deflection highly, and a constant is the safest deflection. Any reply-quality
   headline on this brand mostly measures "did you also say DM us". The retrieval
   + drafting pipeline is near-dead-weight here; I'd have caught this earlier by
   running the canned baseline first.
2. **The judge is the same model grading its own output**, κ vs a human
   unmeasured (§3.4). It scored a canned one-liner 4.9/5 on groundedness. Until
   `make judge` returns κ ≥ 0.6 none of §3.3 is load-bearing.
3. **"auto-rate 0.90" reads as good automation; it's actually a rubber stamp.**
   The same run misses **73%** of the messages that needed a human (§3.2). High
   auto-rate + low missed-escalation would be the win; high auto-rate alone is
   just the always-auto baseline with extra steps.
4. **The escalation number isn't portable across models.** Swapping llama →
   Gemini moved missed-escalation the *wrong way* (0.36 → 0.73), because the
   safety was leaning on the weak model's under-confidence (§4 F1). A
   missed-escalation rate quoted without the exact model + its calibration is
   meaningless.
5. **n = 30, one annotator, hindsight labels.** 95% CI on a 0.90 pass-rate at
   n=30 is roughly ±0.11; per-intent slices have support 1–3. `gold_action` is
   *my* call on what's automatable (Apple would have DM'd most of it), made with
   `reference_resolution` visible — the live agent has neither the hindsight nor,
   apparently, my caution.
6. **Majority-intent baseline is fit on the golden label distribution** — it
   peeks at test priors, so "agent beats majority" understates the gap; fine
   here, but it means the baseline column isn't a clean floor.
7. **`gemini-3.1-flash-lite` is a preview model that may not survive.** During
   this session `gemini-2.0-flash` and `gemini-2.5-flash` both returned "no
   longer available to new users". Reproducing these exact numbers depends on
   the committed `.llm_cache/gemini/`; a cold re-run may have to pick a
   different model and will move the numbers.
8. **Frozen 2017 world + opening-message only.** Grounding is in 2017 iOS-11-era
   precedent; real resolution needs the whole thread. Both inflate any
   "helpfulness" read.

**Bottom line.** The one solid, defensible result is *narrow*: **a capable
classifier (Gemini) beats the keyword and majority baselines on intent (macro-F1
0.50 vs 0.34 / 0.09), and a weak one (llama3.1:8b) does not.** Everything about
reply quality is confounded by the brand's deflect-to-DM behaviour, and the
escalation decision as built adds nothing over "escalate the 3 sensitive intent
categories" — it needs a redesign that doesn't trust a raw confidence scalar.

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
