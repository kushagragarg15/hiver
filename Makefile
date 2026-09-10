# Convenience targets. Windows users without `make`: see run.ps1 / README.
PY ?= python

.PHONY: setup smoke prep pick candidates worksheet eval eval-fast judge test clean

setup:
	$(PY) -m pip install -r requirements.txt

# End-to-end on the offline mock backend -- no dataset pass, no LLM. Proves wiring.
smoke:
	LLM_PROVIDER=mock $(PY) -m pytest -q
	LLM_PROVIDER=mock $(PY) -m eval.run_eval --limit 12 --judge-sample 8

pick:
	$(PY) scripts/pick_brand.py --top 12 --write

prep:
	$(PY) -m src.data_prep

candidates:
	$(PY) scripts/make_golden_candidates.py --target 240

worksheet:
	$(PY) scripts/make_judge_worksheet.py --n 12

# Reproduce the committed headline (groq/gpt-oss-120b, n=150, intent+escalation).
# Runs from the committed .llm_cache in ~1 min, no API key needed.
eval:
	$(PY) -m eval.run_eval --limit 150 --no-judge

# Full run incl. LLM-as-judge over all 233 rows. Needs a backend with quota
# left -- every free tier caps out before this finishes (see README).
eval-full:
	$(PY) -m eval.run_eval --judge-sample 90

judge:
	$(PY) -m eval.judge_agreement

test:
	LLM_PROVIDER=mock $(PY) -m pytest -q

clean:
	rm -rf .llm_cache data/processed reports/agent_golden_preds.jsonl
