# AI support agent for one Twitter brand

Classify an incoming customer tweet → draft a reply grounded in how the brand
has actually resolved similar issues → decide **auto-handle vs escalate**, with a
stated reason. Built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset.

The point of this repo is the **evaluation**, not the agent. See
[`reports/REPORT.md`](reports/REPORT.md) for what the headline number hides.

---

## Reproduce the headline results in <15 minutes

### 0. Prereqs
- Python 3.10+ (developed on 3.13)
- `twcs.csv` from the Kaggle dataset. Put it at
  `Primary Customer Support on Twitter Dataset/twcs/twcs.csv`
  (or edit `paths.raw_csv` in `config.yaml`).
- An LLM backend — **pick one**:
  - **Ollama** (default, no API cost): install from <https://ollama.com>, then
    `ollama pull llama3.1:8b` and make sure `ollama serve` is running.
  - **OpenAI**: set `OPENAI_API_KEY` and change `llm.provider` to `openai` in
    `config.yaml` (uses `gpt-4o-mini` + `gpt-4o` judge; ~$1–2 for a full run).

```bash
pip install -r requirements.txt
```

### 1. Smoke test — no dataset, no LLM (30 s)
Proves the pipeline and eval harness are wired correctly using an offline mock
backend. **These numbers are meaningless** — it's a wiring check.

```bash
make smoke            # Windows: .\run.ps1 smoke
```

### 2. Build the brand's data + golden set (one-time, ~5 min)
```bash
make prep             # twcs.csv -> data/processed/<brand>_{history,eval_pool}.jsonl
```
`config.yaml` ships with `data_prep.brand: <BRAND>` already chosen by
`scripts/pick_brand.py` (rationale + table in `reports/brand_selection.json`).
The hand-labelled golden set is committed at `data/golden/golden_set.jsonl`
(150–250 rows). To rebuild candidates and re-label from scratch:
```bash
make candidates       # stratified sample -> data/golden/golden_candidates.jsonl
python -m eval.label_tool     # interactive; writes data/golden/golden_set.jsonl
```

### 3. Headline numbers
```bash
make eval-fast        # 60 golden rows + 45 judged, the <15-min path
# or
make eval             # full golden set
```
Writes `reports/results.json` (committed) and `reports/results.md` (the tables).
The first run populates `.llm_cache/`; re-runs are near-instant.

### 4. Is the LLM judge trustworthy?
```bash
make worksheet        # -> data/golden/judge_worksheet.csv  (score it by hand)
# save your scores as data/golden/judge_human_scores.csv
make judge            # -> reports/judge_agreement.json  (Cohen's kappa vs you)
```

---

## What each piece is

```
src/
  data_prep.py   twcs.csv -> per-brand reconstructed threads, chronological split
  intents.py     the 8-intent taxonomy (defined from data) + keyword weak-labeller
  llm.py         provider abstraction (ollama | openai | mock) + on-disk cache
  retrieve.py    BM25 over the brand's pre-split resolved threads
  classify.py    LLM intent classifier + majority/keyword baselines
  draft.py       grounded reply drafting (+ retrieval-only baseline)
  escalate.py    rule stack (safety-biased) + optional LLM veto -> action + reason
  agent.py       classify -> retrieve -> draft -> escalate
  pipeline.py    run the agent over a jsonl of messages
eval/
  metrics.py         intent + escalation metrics (explicit formulas)
  baselines.py       trivial + simple baselines for all three sub-tasks
  judge.py           LLM-as-judge rubric (5 dims, 1-5, overall_pass)
  judge_agreement.py judge vs human: exact/within-1/Spearman/kappa
  run_eval.py        the headline harness
  label_tool.py      terminal golden-set labeller (resumable)
scripts/
  pick_brand.py            data-driven brand choice
  explore_intents.py       cluster openings -> taxonomy evidence
  make_golden_candidates.py stratified candidate sampler
  make_judge_worksheet.py  emit replies for human judge-scoring
reports/
  REPORT.md          problem framing, results, failure analysis, caveats
  DECISION_LOG.md    the non-obvious decisions and why
```

## Design in one paragraph
Intent is a small closed set (8 buckets) derived by clustering inbound openings
and consolidating by hand. The drafter is retrieval-augmented: BM25 pulls the
`k` most similar *pre-split* threads and the model must ground its reply in how
the brand resolved them — it is told not to invent policy, prices, or
account-specific facts. The escalation decision is a transparent rule stack
(hard regex → always-escalate intents → confidence floor → retrieval-score floor
→ weak-precedent flag) with an optional LLM safety veto; **anything uncertain
escalates**. Every decision carries a human-readable `reason` and a `signals`
dict for failure analysis.

## Caveats that matter
- **No live API budget was available during development**, so the committed
  `reports/results.json` is from `<BACKEND>` — re-run `make eval` on your backend
  to regenerate.
- The dataset is from 2017; "how the brand resolves things" is frozen at that
  point.
- Single annotator for the golden set. See `data/golden/SAMPLING_NOTES.md`.

## Attribution
- Dataset: *Customer Support on Twitter*, thoughtvector, Kaggle (CC0).
- Libraries: `rank-bm25`, `scikit-learn`, `openai`, `pandas`, `scipy`.
- LLM coding assistant (Claude) was used to scaffold this repo; all design
  decisions and the evaluation are the author's and are defended in
  `reports/DECISION_LOG.md`.
