# Windows PowerShell runner. Usage:  .\run.ps1 <target>
# targets: setup | smoke | pick | prep | candidates | worksheet | eval | eval-cold | judge | test
param([string]$Target)
$ErrorActionPreference = "Stop"
if (-not $Target) {
    Write-Host "usage: .\run.ps1 <target>"
    Write-Host "targets: setup | smoke | pick | prep | candidates | worksheet | eval | eval-cold | judge | test"
    Write-Host "start here: .\run.ps1 smoke   then   .\run.ps1 eval   (no API key needed)"
    exit 1
}
$env:PYTHONIOENCODING = "utf-8"
$PY = if ($env:PY) { $env:PY } else { "python" }

switch ($Target) {
    "setup"      { & $PY -m pip install -r requirements.txt }
    "smoke" {
        $env:LLM_PROVIDER = "mock"
        & $PY -m pytest -q
        & $PY -m eval.run_eval --limit 12 --judge-sample 8 --out-dir reports/smoke
    }
    "pick"       { & $PY scripts/pick_brand.py --top 12 --write }
    "prep"       { & $PY -m src.data_prep }
    "candidates" { & $PY scripts/make_golden_candidates.py --target 240 }
    "worksheet"  { & $PY scripts/make_judge_worksheet.py --n 12 }
    "eval"       { & $PY -m eval.run_eval }                          # replays committed cache, no key, ~2 min
    "eval-cold" {                                                    # needs GEMINI_API_KEY
        if (-not $env:GEMINI_API_KEY) { Write-Error "eval-cold: GEMINI_API_KEY is not set -- refusing to delete the committed cache" }
        Remove-Item -Recurse -Force .llm_cache/gemini -ErrorAction SilentlyContinue
        & $PY -m eval.run_eval
    }
    "judge"      { & $PY -m eval.judge_agreement }
    "test"       { $env:LLM_PROVIDER = "mock"; & $PY -m pytest -q }
    default      { Write-Error "unknown target: $Target" }
}
