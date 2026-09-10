# AI support agent — report

**Brand: AppleSupport** · Dataset: *Customer Support on Twitter* (Kaggle,
thoughtvector), Oct–Dec 2017 slice · Agent: intent → retrieval → grounded draft
→ escalate/auto.

> **Status of the numbers.** All development happened with **no LLM API budget**
> (OpenAI key present but out of credits) and Ollama not yet installed on the
> dev machine. Every non-LLM stage is built, tested and run on the real data;
> `reports/results.json` currently holds **mock-backend** output (a wiring check
> — `meta.provider == "mock"`). Run `make eval` on a real backend (Ollama
> `llama3.1:8b` is the configured default) to populate the real figures — the
> harness writes this file's tables into `reports/results.md` automatically.
> Bracketed blanks below are filled by that run.

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

Raw tables regenerated into `reports/results.md` by `make eval`. Golden n = **[N]**.

### 3.1 Intent classification
| system | accuracy | macro-F1 |
|---|---|---|
| **agent (LLM few-shot)** | **[ ]** | **[ ]** |
| simple baseline — keyword weak-labeller | [ ] | [ ] |
| trivial baseline — majority intent (`software_update_issue`) | [ ] | [ ] |

Expected shape: the majority baseline gets decent *accuracy* (~0.45–0.5, the
head-class share) but near-zero *macro-F1*; the keyword baseline lands in
between; the LLM's value shows up almost entirely in the rare classes
(account, billing, complaint) and in separating hardware-battery from
update-battery. Per-class table + confusion matrix in `results.md`.

### 3.2 Escalate vs auto (positive class = escalate)
| system | esc-recall | **missed-esc-rate ↓** | unnec-esc-rate ↓ | auto-rate |
|---|---|---|---|---|
| **agent (rule stack)** | **[ ]** | **[ ]** | **[ ]** | **[ ]** |
| trivial — always-auto | 0.00 | **1.00** | 0.00 | 1.00 |
| trivial — always-escalate | 1.00 | **0.00** | 1.00 | 0.00 |
| simple — escalate by intent prior | [ ] | [ ] | [ ] | [ ] |

The two trivial baselines bracket the trade-off. On the seed golden set the
label mix is **19 auto / 11 escalate**, so always-escalate gets recall 1.0 at
precision ~0.37 and **auto-rate 0** — safe and useless. The agent's bar is to
beat the **intent-prior** baseline: same missed-esc-rate at a higher auto-rate,
or a lower missed-esc-rate at the same auto-rate.

### 3.3 Reply quality — LLM-as-judge (1–5, n = [ ])
| system | groundedness | correctness/safety | relevance | pass-rate |
|---|---|---|---|---|
| **agent (retrieval-grounded)** | **[ ]** | **[ ]** | **[ ]** | **[ ]** |
| simple — retrieval-only (echo top precedent's first brand turn) | [ ] | [ ] | [ ] | [ ] |
| trivial — canned "please DM us" | [ ] | [ ] | [ ] | [ ] |

Prediction: because Apple's precedents are mostly "DM us + which iOS version",
the **retrieval-only baseline will score deceptively well on groundedness** and
the agent's lift over it will be small on this brand. That is a finding about
the brand, not a null result for RAG — see §5.

### 3.4 Is the judge trustworthy?
`make worksheet` → human scores 36 replies → `make judge` →
`reports/judge_agreement.json`: per-dimension exact / within-1 / Spearman, and
**Cohen's κ on `overall_pass`**. Bar for using the judge in the headline:
**κ ≥ 0.6 and correctness/safety within-1 ≥ 0.8**. Result: **[ ]** — verdict
**[ ]**. If κ < 0.6, §3.3 is reported as directional only.

---

## 4. Top 5 failure modes

F1 and F4–F5 need the real run (`reports/agent_golden_preds.jsonl`); F2 and F3
are structural and already evidenced.

### F1 — [title] *(fill from agent_golden_preds.jsonl)*
Example `[id]`: "[message]" · gold `[i]/[a]` · agent `[i]/[a]` · what happened /
hypothesis / cheap fix.

### F2 — "how do I cancel my subscription" over-escalates
- **Mechanism.** The hard regex `\b(refund|reimburse|compensat)` and the
  `billing_appstore_subscription` always-escalate rule fire on *any* billing
  wording, including pure how-tos ("how do I cancel", "where's my receipt")
  that have safe documented answers. Predicted effect: inflated unnec-esc-rate
  and depressed auto-rate on the billing slice.
- **Evidence.** Seed row `gold_0018` ("credit on my account but can't buy —
  billing information") is a true escalate, but the same rule would also escalate
  "how do I see my purchase history".
- **Cheap fix.** Split the intent into `billing_dispute` (escalate) vs
  `billing_howto` (auto); keep the regex for `cancel my account`, drop bare
  `refund` as a *hard* trigger and let it be a soft signal.

### F3 — BM25 grounds on shared boilerplate, not the problem
- **Mechanism.** AppleSupport openings share heavy filler ("@AppleSupport fix
  this", "ever since the update", "@115858"). BM25 can rank a precedent highly on
  that filler while the actual defect differs, and the score still clears the
  3.0 floor — so the drafter grounds in the wrong resolution *and* the
  escalation stack thinks precedent is strong.
- **Evidence.** The keyword weak-labeller alone puts ~35% of messages in `other`
  because the diagnostic tokens are drowned out by filler; BM25 has the same
  blind spot.
- **Cheap fix.** Extend stopwords with brand filler; add a cross-encoder
  re-rank on the top-10; or require the retrieved thread's intent to match the
  predicted intent before it counts toward the score floor.

### F4 — [title] *(fill)*
### F5 — [title] *(fill)*

---

## 5. What is misleading about my headline number? *(mandatory)*

Headline once populated will read roughly: *"macro-F1 [ ] on 8-way intent,
[ ] missed-escalation rate, [ ] reply pass-rate — beating both baselines."*
Why not to trust it at face value:

1. **AppleSupport is a near-degenerate brand for two of the three tasks.**
   ~65% of its real replies are "DM us", so (a) any "grounded" reply metric
   rewards a bot that just says "DM us with your iOS version", and (b) the
   retrieval-only baseline is unusually strong. The agent can post a good
   groundedness number while adding little over "please DM us".
2. **`gold_action` encodes my opinion of what's automatable, not Apple's.**
   I labelled "known iOS 11 bug + standard workaround" as `auto`; Apple
   escalated those to DM. So a model that *imitated the brand* scores badly
   here, and an *over-eager* model scores well here while being risky in
   production. The headline auto-rate is only meaningful next to that
   definition.
3. **One annotator, once, n≈[N].** No inter-annotator agreement. A same-week
   self-recheck of the auto/escalate calls agreed on **[ ]%** — so ~[ ] of the
   action labels are coin-flips and the escalation metrics carry at least that
   much noise. 95% CI on a pass-rate at n=[N] is ≈ ±[ ] points; per-intent
   slices (billing n≈[ ], account n≈[ ]) are anecdotes.
4. **The judge is an LLM grading an LLM**, agreement κ = **[ ]**. Even at κ≈0.6,
   shared blind spots (both reward fluent, confident, *wrong* text) are
   invisible. If κ < 0.6 the reply numbers are not a headline.
5. **Majority-intent baseline is fit on the golden label distribution** — it
   peeks at test priors, so "agent beats majority by X" *understates* the gap.
6. **Frozen 2017 world.** "How the brand resolves things" = 2017 policies, iOS
   versions, URLs. On today's traffic groundedness would fall and nothing in
   this eval would notice.
7. **Warm cache.** Re-running "reproduces" the numbers partly because LLM
   responses are cached (`.llm_cache/`); a cold run on a newer model point can
   move everything. Delete the cache for a true cold repro.
8. **Opening-message only.** Real resolution needs the whole thread; single-turn
   scoring overstates end-to-end usefulness.

---

## 6. One more week
1. **Second annotator + adjudication** on 100 rows → real κ; fix the taxonomy
   edges that disagree (F2).
2. **Learned escalation head** — logistic regression on
   [intent one-hot, confidence, BM25 top-k scores, msg length, sentiment, regex
   hits] → calibrated P(escalate), hard rules as overrides; compare to the rule
   stack at matched auto-rate.
3. **Retrieval A/B** — BM25 vs MiniLM vs BM25+cross-encoder, scored by "did the
   drafter cite a precedent with the right intent".
4. **Judge hardening** — 2-shot rubric with a worked pass + fail, ensemble two
   models, re-measure κ; add an adversarial "confident fabrication" probe set.
5. **Counterfactual slice** — 30 hand-written messages that *look* precedented
   but aren't (new bug / changed policy); confirm the BM25-score floor actually
   escalates them.
6. **Try a second brand (SpotifyCares)** where in-channel resolution is real, so
   the reply task isn't dominated by "DM us", and compare what each brand's data
   lets you automate.
7. **Cost/latency table** per backend at the chosen operating point.

---

## Appendix — reproduce
`README.md`. `make smoke` (no backend) → `make prep` → `make eval` →
`make worksheet` (hand-score) → `make judge`. All thresholds in `config.yaml`;
the non-obvious calls in `DECISION_LOG.md`.
