# AI support agent for one Twitter brand

Classify an incoming customer tweet → draft a reply grounded in how the brand
has actually resolved similar issues → decide **auto-handle vs escalate**, with a
stated reason. Built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset.

The point of this repo is the **evaluation**, not the agent. See
[`reports/REPORT.md`](reports/REPORT.md) for what the headline number hides.

---

## Reproduce the headline results in <15 minutes

The committed headline is **one run: `gemini-3.1-flash-lite` as worker and
judge, all 233 golden rows** (`reports/results.json`). Its LLM cache is
committed, so reproducing it needs **no API key and no raw dataset**:

```bash
pip install -r requirements.txt      # Python 3.10+ (developed on 3.13)
make smoke                           # 30 s  — tests + mock-backend wiring check (numbers meaningless)
make eval                            # ~2 min — replays the committed cache -> reports/results.{json,md}
```
Windows without `make`: `.\run.ps1 smoke`, `.\run.ps1 eval`.

`make eval` writes `reports/results.json` + `results.md` (the tables in
`REPORT.md` §3) + `reports/agent_golden_preds.jsonl` (per-row output behind
§4). `data/processed/AppleSupport_*.jsonl` (the BM25 memory + eval pool) is
committed for the same reason.

### Re-running live (optional)
```bash
export GEMINI_API_KEY="key1,key2,key3"   # comma-separate several keys to pool quota
make eval-cold                           # deletes .llm_cache/gemini, ~1,150 calls, ~50 min
```
Gemini's free tier is **500 requests/day per key**; a full judged run is
~1,150 calls, so `src/llm.py` round-robins across every key in the list and
skips a key for the rest of the run when it returns a quota error. Five keys
were used for the committed run. Other OpenAI-compatible backends are one
`config.yaml` line away (`llm.provider`: `groq` / `openai` / `deepseek` /
`ollama` / `mock`; base URLs in `src/llm.py::_PRESETS`); `llm.judge_provider`
lets the judge run on a different vendor than the worker. Reasoning models
need `reasoning_effort` set (already configured for gemini/groq) or they
truncate the JSON. **Gemini model IDs churn** — if `gemini-3.1-flash-lite`
disappears, list live ones with
`curl -s https://generativelanguage.googleapis.com/v1beta/models -H "x-goog-api-key: $GEMINI_API_KEY"`.

### Rebuilding the data from the raw CSV (optional, ~5 min)
Only needed to change brand or re-sample the golden set. Put `twcs.csv` from
the Kaggle dataset at `Primary Customer Support on Twitter Dataset/twcs/twcs.csv`
(or edit `paths.raw_csv`).
```bash
make pick             # brand ranking -> reports/brand_selection.json (won't overwrite the set brand; see DECISION_LOG #1)
make prep             # twcs.csv -> data/processed/<brand>_{history,eval_pool}.jsonl
make candidates       # stratified sample -> data/golden/golden_candidates.jsonl
python -m eval.label_tool            # interactive labeller -> data/golden/golden_set.jsonl
python -m eval.label_tool --review   # review the remaining 64 model-assisted rows (flips them to `human`)
```

### Is the LLM judge trustworthy?
```bash
make worksheet        # -> data/golden/judge_worksheet.csv  (score it by hand)
# save your scores as data/golden/judge_human_scores.csv
make judge            # -> reports/judge_agreement.json  (Cohen's kappa vs you)
# cross-vendor check (needs GROQ_API_KEY; doesn't touch the committed gemini headline):
LLM_JUDGE_PROVIDER=groq make judge
```
Already run once each way: κ=0.532 (gemini, self) / κ=0.566 (groq, cross-vendor)
— both below the κ≥0.6 trust bar (`REPORT.md` §3.4).

---

## What each piece is

```
src/
  data_prep.py   twcs.csv -> per-brand reconstructed threads, chronological split
  intents.py     the 8-intent taxonomy (defined from data) + keyword weak-labeller
  llm.py         OpenAI-compatible client for any provider, multi-key rotation, on-disk cache, mock backend
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

## Headline (n=233, `gemini-3.1-flash-lite`) and what it hides
| | agent | simple baseline | trivial baseline |
|---|---|---|---|
| intent macro-F1 / accuracy | **0.60 / 0.67** | keyword 0.43 / 0.42 | majority 0.08 / 0.46 |
| missed-escalation rate ↓ | **0.55** | intent-prior 0.56 | always-escalate 0.00 |
| reply judge pass-rate | **0.91** | retrieval-only 0.64 | canned "DM us" 0.81 |

Full version: `REPORT.md` §5.
- **Solid:** the classifier beats the keyword + majority baselines; the four
  largest classes F1 0.70–0.78.
- **Not deployable:** the escalate/auto stack auto-handles **55% of messages
  that needed a human** and is one row away from the intent-prior lookup —
  the confidence and retrieval floors never fire, and the taxonomy routes
  physical hardware faults (bent, water, logic board) to *auto*
  (`REPORT.md` §4 F1/F2).
- **Thin:** the agent beats canned on judge pass-rate only on *relevance*;
  groundedness is identical and 88% of agent replies are DM redirects.
- Golden set: **233 rows** — 169 human-labelled, 64 still model-assisted
  (`python -m eval.label_tool --review`). Single annotator, hindsight labels;
  the review pass changed ~9% of labels and moved macro-F1 0.63 → 0.60,
  missed-escalation 0.53 → 0.55 with the model outputs held fixed.
- Judge trust measured against a human on 36 rows, both self-judged (gemini,
  κ=0.532) and cross-vendor (groq/qwen3, κ=0.566) — both below the κ≥0.6 bar
  (`reports/judge_agreement.json`, `reports/REPORT.md` §3.4).
- Dataset is 2017; "how the brand resolves things" is frozen there.

## Attribution
- Dataset: *Customer Support on Twitter*, thoughtvector, Kaggle (CC0).
- Libraries: `rank-bm25`, `scikit-learn`, `openai`, `pandas`, `scipy`.
- LLM coding assistant (Claude) was used to scaffold this repo; all design
  decisions and the evaluation are the author's and are defended in
  `reports/DECISION_LOG.md`.
