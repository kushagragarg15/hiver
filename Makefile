# Convenience targets. Windows users without `make`: see run.ps1 / README.
PY ?= python

.PHONY: setup smoke prep pick candidates worksheet eval eval-cold judge test clean

setup:
	$(PY) -m pip install -r requirements.txt

# End-to-end on the offline mock backend -- no dataset pass, no LLM. Proves
# wiring. Writes to reports/smoke/ so the committed headline is untouched.
smoke:
	LLM_PROVIDER=mock $(PY) -m pytest -q
	LLM_PROVIDER=mock $(PY) -m eval.run_eval --limit 12 --judge-sample 8 --out-dir reports/smoke

pick:
	$(PY) scripts/pick_brand.py --top 12 --write

prep:
	$(PY) -m src.data_prep

candidates:
	$(PY) scripts/make_golden_candidates.py --target 240

worksheet:
	$(PY) scripts/make_judge_worksheet.py --n 12

# Reproduce the committed headline: gemini-3.1-flash-lite, all 233 rows,
# intent + escalation + LLM-as-judge. Replays the committed .llm_cache in
# ~2 min with NO API key.
eval:
	$(PY) -m eval.run_eval

# Same run, cold: ~1,150 live calls. Needs GEMINI_API_KEY (comma-separate
# several keys to pool their free-tier quota; see README).
eval-cold:
	@test -n "$$GEMINI_API_KEY" || { echo "eval-cold: GEMINI_API_KEY is not set -- refusing to delete the committed cache"; exit 1; }
	rm -rf .llm_cache/gemini
	$(PY) -m eval.run_eval

judge:
	$(PY) -m eval.judge_agreement

test:
	LLM_PROVIDER=mock $(PY) -m pytest -q

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__
