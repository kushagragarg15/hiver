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
  | `groq` | **free** | `GROQ_API_KEY` from <https://console.groq.com/keys> | **primary run** used `openai/gpt-oss-120b` (worker) + `qwen/qwen3.8-27b` (judge). 200k tokens/day cap |
  | `gemini` | **free** | `GEMINI_API_KEY` from <https://aistudio.google.com/apikey> | reply-quality run used `gemini-3.1-flash-lite`. 500 req/day cap; model IDs churn — see below |
  | `ollama` | free, local | install <https://ollama.com>, `ollama pull llama3.1:8b` | no key; ~1 min/message on CPU (slow). Committed as the local baseline |
  | `openai` | paid | `OPENAI_API_KEY` | `gpt-4o-mini` + `gpt-4o` judge, ~$1–2/run |
  | `deepseek` | ~$0.20/run | `DEEPSEEK_API_KEY` | cheap, not free |

  `reports/results.json` is the `groq/gpt-oss-120b` run (n=150); `results_*`
  keep every backend side by side. `.llm_cache/<provider>/` is committed so
  `make eval` reproduces each in seconds; the cache misses cleanly on a new
  provider or model. Reasoning models (Gemini 3.x flash, gpt-oss, qwen3) need
  `llm.<provider>.reasoning_effort` set (already configured) or they truncate
  the JSON. **Every free tier has a daily cap** (Groq 200k tok, Gemini 500 req)
  that a full 233-row judged run exceeds — hence n=150. **Gemini model IDs are
  unstable** (`gemini-2.0-flash` / `gemini-2.5-flash` went "not available to new
  users" mid-project); list live ones with
  `curl -s https://generativelanguage.googleapis.com/v1beta/models -H "x-goog-api-key: $GEMINI_API_KEY"`.

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

## Caveats that matter (full version: `REPORT.md` §5)
- Runs, all committed with warm caches:
  | run | worker | judge | n | gives |
  |---|---|---|---|---|
  | primary | groq `gpt-oss-120b` | qwen3.8-27b | 150 | intent + escalation |
  | reply-quality | gemini `3.1-flash-lite` | self | 30 | LLM-as-judge table |
  | local baseline | ollama `llama3.1:8b` | self | 30 | model-capacity contrast |
- **Solid:** capable models beat the keyword + majority intent baselines
  (macro-F1 ~0.5–0.65 vs 0.49 / 0.08); `llama3.1:8b` doesn't (0.32).
- **Not deployable:** the escalate/auto stack misses **36–73%** of true
  escalations depending on the worker model — it gates on an uncalibrated
  confidence scalar and has no rule for hardware-safety / recall / image-only
  cases (`REPORT.md` §4 F1/F2).
- **Confounded:** canned "please DM us" *beats* the agent on reply-quality
  pass-rate (0.93 vs 0.90) — AppleSupport's real replies are deflections.
- Golden set: **233 rows**, but 203 are model-assisted drafts pending review
  (`python -m eval.label_tool --review`). Single annotator, hindsight labels.
- Judge has **no human anchor** yet (`make worksheet` → hand-score → `make judge`).
- Free-tier token/request caps (Gemini 500/day, Groq 200k tok/day) stopped the
  primary run at n=150 and blocked its judge pass.
- Dataset is 2017; "how the brand resolves things" is frozen there.

## Attribution
- Dataset: *Customer Support on Twitter*, thoughtvector, Kaggle (CC0).
- Libraries: `rank-bm25`, `scikit-learn`, `openai`, `pandas`, `scipy`.
- LLM coding assistant (Claude) was used to scaffold this repo; all design
  decisions and the evaluation are the author's and are defended in
  `reports/DECISION_LOG.md`.
