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

14. **On-disk LLM cache keyed by (provider, model, messages, params); both
    runs' caches are committed.** `.llm_cache/gemini/` and `.llm_cache/ollama/`
    hold the exact calls behind `reports/results_*.json`, so `make eval`
    reproduces either headline in ~15 s instead of 18–27 min. A different
    backend/model misses cleanly. `rm -rf .llm_cache/` for a true cold run
    (which, for Gemini, may need a different model id — see #19).

15. **`mock` LLM provider ships in `llm.py`.** Lets `make smoke` and the test
    suite exercise the whole pipeline with no backend and no network — useful
    for CI and for a reviewer who wants to see the wiring before installing
    Ollama. Its outputs are canned; never reported.

16. **Config is one `config.yaml`, read once.** Every threshold that could be
    argued about (confidence floor, BM25 floor, `k`, split date, always-escalate
    list) is there, not buried in code, so tuning is visible in a diff.

17. **Committed a 30-row hand-labelled *seed* golden set, not the full 150–250.**
    The 233 candidates are staged; the seed is enough to (a) prove the harness
    on real data and (b) surface real failure modes for the report, without
    front-loading hours of labelling before the method was settled. Expanding it
    is the top "next week" item, and the report's numbers are all flagged n=30.

18. **Ran two backends and committed both: `gemini-3.1-flash-lite` (primary) and
    `llama3.1:8b` (local baseline).** The pair *is* a finding — Gemini clears
    the intent baselines, llama doesn't, and swapping them moves
    missed-escalation the wrong way (safety was leaning on llama's
    under-confidence). One backend would have hidden that. `src/llm.py` treats
    every provider as an OpenAI-compatible endpoint (`_PRESETS`), so adding
    gemini/groq/deepseek/openrouter was a table entry, not a code path.

19. **`gemini-3.1-flash-lite`, not a `-latest` alias or a bigger flash.** During
    this work `gemini-2.0-flash` and `gemini-2.5-flash` both started returning
    "no longer available to new users", and `gemini-3.5/3.7/3.8-flash` were
    intermittently 503 or returned non-standard envelopes. `3.1-flash-lite` was
    the most reliable id that also emitted clean JSON without burning output
    tokens on visible reasoning. Pinned (not `flash-latest`) for reproducibility,
    accepting that a cold re-run may need a different id — the committed cache is
    the real reproducibility guarantee (#14).

20. **Added a client-side RPM throttle (`llm.<provider>.rpm`) + 429-aware
    backoff.** Free tiers rate-limit hard (Gemini ~15/min); without the throttle
    a run trips limits and the retry loop wastes minutes. Set to 14 for Gemini.
