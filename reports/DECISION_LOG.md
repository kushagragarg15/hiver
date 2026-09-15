# Decision log

Non-obvious choices, and why. Ordered roughly by where they bite.

1. **One brand, chosen by data — then one manual override.**
   `scripts/pick_brand.py` ranks the top-12 brands by volume on four properties:
   % of threads the brand closed itself, median depth, **% of resolutions that
   don't immediately punt to DM/phone** (answerable in-channel), and intent
   spread (entropy). Full table: `reports/brand_selection.json`. The script's
   top two came out **AmazonHelp 0.750 vs AppleSupport 0.740** — a tie inside
   the noise of my hand-picked weights. I overrode to **AppleSupport** because:
   (a) Amazon's `intent_spread` is 1.33 vs Apple's 2.09 — Amazon traffic is
   ~2 intents (order status, refunds), which makes the classification task
   trivial and the report thin; (b) Apple resolves only ~35% in-channel, so it
   has *more real escalate cases* — and the escalate/auto decision is the
   assignment's stated "hard part". My score weighted in-channel resolvability
   highest (0.35), which perversely penalises the brand with the more
   interesting escalation problem. That's a flaw in the metric, logged not
   hidden.

2. **Chronological memory/eval split (`split_date`), not random.** The agent may
   retrieve from the brand's history. A random split lets it retrieve a
   near-duplicate of the thread it's being tested on, so the reply task becomes
   copy-paste and every number inflates. Everything before the date is memory;
   everything on/after is the eval pool. No thread is in both.

3. **Threads reconstructed by union-find on the reply graph**, rooting each tweet
   via `in_response_to_tweet_id` and keeping components that contain ≥1 brand
   tweet and ≥1 customer tweet. `response_tweet_id` is redundant with the parent
   pointer and noisier, so it's ignored. Pointer-jumping is vectorised — no
   per-row Python loop over 3M tweets.

4. **8 intents, closed set, with an `other` escape hatch.** Derived by
   TF-IDF+KMeans over ~800 inbound openings (`scripts/explore_intents.py`) then
   consolidated by hand. Fewer buckets = more label agreement and a usable
   confusion matrix; `other` keeps forced misclassification out of the metrics
   (it counts against coverage instead). Banking77 was *not* used — its intents
   are retail-banking specific and would pull the taxonomy off-brand.

5. **Two intents are "always escalate" by policy** regardless of model
   confidence: `account_access_security` and `billing_appstore_subscription`.
   These need identity/payment verification the bot cannot do; a confident wrong
   answer here is a security or money incident, not a bad tweet.
   `complaint_churn_risk` is also on the list — a templated bot reply to an angry
   "I'm leaving" tweet measurably makes it worse.

6. **Escalation is a rule stack, not a classifier.** Order: hard regex
   (legal/fraud/self-harm/"cancel my account") → always-escalate intents →
   intent-confidence floor (0.60) → retrieval-score floor (BM25 ≥ 3.0) →
   drafter's own "I used weak precedent" flag → optional LLM veto. Auto-handling
   needs *all* green. Rationale: it's auditable, safety-biased, and every branch
   yields a human-readable reason. The learned version is future work (needs
   more labels than 200).

7. **BM25, not embeddings, for retrieval.** No model download → the repro stays
   dependency-light and offline-capable, and it's deterministic. An embedding
   retriever is stubbed behind `retrieval.method` for later comparison.

8. **The drafter is told what *not* to do, explicitly.** No invented policy,
   prices, repair times, compensation; never claim to have looked at the
   account; never ask for personal info in public; ≤55 words; ≤1 link and only
   if a precedent used one. Most failure modes in this task are
   over-confident fabrication, so the prompt spends its budget on suppression.

9. **Retrieval-score floor doubles as a grounding gate.** If the top precedent
   is weak (BM25 < 3.0) the drafter is handed "no strong precedent found" and
   the escalation stack routes to a human. One threshold, two jobs.

10. **LLM-as-judge scores 5 dimensions but `overall_pass` only uses 3**
    (groundedness ≥3, correctness_safety ≥4, relevance ≥3). Tone and
    completeness are diagnostic, not gating — a polite, complete, *wrong* reply
    should still fail.

11. **Judge trust is measured, not assumed.** `make worksheet` → human scores
    36 replies → `make judge` computes exact/within-1/Spearman per dimension and
    Cohen's kappa on `overall_pass`. Bar for headlining the judge number:
    kappa ≥ 0.6 and safety within-1 ≥ 0.8. Below that the report says so.

12. **Golden `gold_action` is labelled with hindsight** (the labeller sees what
    the brand actually did). This makes it a harder target than a real-time
    human — deliberately — and is called out in the "misleading headline"
    section rather than hidden.

13. **Majority-intent baseline is fit on the *golden* labels**, which is
    optimistic (it peeks at test-set priors). Kept anyway because it's the
    honest floor for "predict the prior" and the caveat is cheaper than a
    train split on 200 rows. Flagged inline in `run_eval.py`.

14. **The LLM cache is committed; it is the reproducibility guarantee.**
    Keyed by (provider, model, messages, params). `.llm_cache/gemini/` holds
    exactly the 1,153 calls behind `reports/results.json` — nothing else — so
    `make eval` replays the headline keyless in ~2 min instead of ~50 min and
    ~1,150 live calls. A different backend/model misses cleanly. Gemini model
    ids are a moving target (`gemini-2.0/2.5-flash` went "not available to new
    users" mid-project), so a cold re-run may need a different id; the cache
    means the committed numbers don't depend on that.

15. **`mock` LLM provider ships in `llm.py`.** Lets `make smoke` and the test
    suite exercise the whole pipeline with no backend and no network. Its
    outputs are canned; never reported.

16. **Config is one `config.yaml`, read once.** Every threshold that could be
    argued about (confidence floor, BM25 floor, `k`, split date, always-escalate
    list) is there, not buried in code, so tuning is visible in a diff.

17. **Golden set: 30 hand-labelled from scratch, then 203 model-assisted drafts
    (233 total), 139 of them since human-reviewed.** The 30-row seed was
    labelled first, cold. The remaining 203 were drafted by applying the
    written protocol (`SAMPLING_NOTES.md`) and stamped `label_source:
    "model_assisted"`. A review pass (`label_tool.py --review`) then walked
    139 of them, changing 12 labels (10 intents, 4 actions) and flipping them
    to `human`; 64 remain `model_assisted`. The eval was re-run on the
    reviewed labels by replaying the committed cache — same model outputs,
    macro-F1 0.63 → 0.60, missed-escalation 0.53 → 0.55 — and REPORT.md
    reports the post-review numbers. Disclosed in `SAMPLING_NOTES.md` and
    next to every results table. Mix: 155 auto / 78 escalate (unchanged).

18. **One committed run, on the full golden set, rather than several partial
    ones.** Earlier iterations had three backends at n=30/n=150 with no
    complete judge pass, because every free tier (Gemini 500 req/day per key,
    Groq 200k tok/day) capped out first. Three half-runs invited
    cross-model comparisons the sample sizes couldn't support. The repo now
    carries exactly one run — `gemini-3.1-flash-lite`, n=233, judged — and
    the report is written around it. `src/llm.py` still treats every
    provider as an OpenAI-compatible endpoint (`_PRESETS`), so switching is a
    config line.

19. **Multiple API keys pooled in one env var.** `GEMINI_API_KEY="k1,k2,..."`;
    `src/llm.py` round-robins across the list, throttles RPM per key, and on a
    429/quota error marks that key exhausted and retries immediately on the
    next one instead of sleeping. This is what made n=233 with a judge pass
    reachable on the free tier (5 keys × 500/day > 1,132 calls). Chosen over
    concurrency: the eval loop is sequential and latency-bound (~2.7 s/call),
    and the binding constraint was the *daily* cap, not RPM.

20. **Gemini 3.x flash models are thinking models; `reasoning_effort: "none"`
    is in config.** Without it they spend the completion budget on hidden
    reasoning and truncate the JSON. Reasoning models also get a floor of 800
    completion tokens regardless of the caller's budget (the classifier asks
    for 120) — the cache key uses the effective value so this stays
    reproducible.

21. **Judge provider is a separate config key (`llm.judge_provider`) — and it
    was deliberately left unset for the committed run.** The wiring exists so
    the judge can be a different vendor from the worker (§3.4's ask). Only
    Gemini keys were available, so worker and judge are the same model and
    the report says so in three places rather than pretending otherwise. The
    honest version of "judge independence" is a config line away, not a
    claim. It was subsequently exercised: the 36-row judge-agreement check
    (`make judge`) was re-run with `LLM_JUDGE_PROVIDER=groq` (env override,
    not the committed config default) once a Groq key was available — κ went
    from 0.532 (self) to 0.566 (cross-vendor), still below the 0.6 bar
    (§3.4). `config.yaml`'s default stays unset on purpose, so `make eval`
    keeps replaying the committed §3.3 headline keyless from
    `.llm_cache/gemini`.
