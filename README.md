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
- An LLM backend — set `llm.provider` in `config.yaml` (all are
  OpenAI-compatible; base URLs are built into `src/llm.py`):

  | provider | cost | how | notes |
  |---|---|---|---|
  | `gemini` | **free** | `GEMINI_API_KEY` from <https://aistudio.google.com/apikey> | `gemini-2.0-flash`, 15 rpm / 1500-day free tier — **recommended** |
  | `groq` | **free** | `GROQ_API_KEY` from <https://console.groq.com/keys> | `llama-3.3-70b`, ~30 rpm free tier |
  | `ollama` | free, local | install <https://ollama.com>, `ollama pull llama3.1:8b` | no key; ~1 min/message on CPU (slow) |
  | `openai` | paid | `OPENAI_API_KEY` | `gpt-4o-mini` + `gpt-4o` judge, ~$1–2/run |
  | `deepseek` | ~$0.20/run | `DEEPSEEK_API_KEY` | cheap, not free |

  The committed `reports/results.json` is an `ollama/llama3.1:8b` run; switching
  provider is one line and the cache misses cleanly (keyed by provider+model).

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
make eval             # full golden set; reuses the committed .llm_cache -> seconds
make eval-fast        # --limit 60 --judge-sample 45 (for a bigger golden set)
```
Writes `reports/results.json` (committed) + `reports/results.md` (tables) +
`reports/agent_golden_preds.jsonl` (per-example, for failure analysis).
A cold run (`rm -rf .llm_cache/`) on `llama3.1:8b` CPU is ~1 min/row; on
`gpt-4o-mini` the full seed set is ~1 min. Switch backend in `config.yaml`
(`llm.provider`).

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
- The committed `reports/results.json` is a **real `ollama/llama3.1:8b` run over
  a 30-row seed golden set** (27 min; `.llm_cache/` is committed so `make eval`
  reproduces it in seconds). On this brand + model the agent **loses to the
  keyword baseline on intent macro-F1** and misses 36% of escalations — the
  grounded drafter's reply quality is the only clear win. See `REPORT.md` §3/§5.
- The golden set is a **30-row seed**; 233 candidates are staged for labelling
  (`python -m eval.label_tool`). Single annotator. `data/golden/SAMPLING_NOTES.md`.
- The LLM judge is also `llama3.1:8b` and its human-agreement κ is **not yet
  measured** (`make worksheet` → hand-score → `make judge`).
- The dataset is from 2017; "how the brand resolves things" is frozen there.

## Attribution
- Dataset: *Customer Support on Twitter*, thoughtvector, Kaggle (CC0).
- Libraries: `rank-bm25`, `scikit-learn`, `openai`, `pandas`, `scipy`.
- LLM coding assistant (Claude) was used to scaffold this repo; all design
  decisions and the evaluation are the author's and are defended in
  `reports/DECISION_LOG.md`.
