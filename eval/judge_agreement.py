"""How much do we trust the LLM judge? Compare it to a human on the same replies.

Inputs
  --worksheet data/golden/judge_human_scores.csv
      columns: id,message,reference,candidate,
               groundedness,relevance,correctness_safety,tone,completeness,overall_pass
      (the human fills the score columns; make_judge_worksheet.py emits the rest)

Outputs (reports/judge_agreement.json + stdout)
  per-dimension: exact-agreement %, within-1 %, MAE, Spearman rho
  overall_pass: accuracy, Cohen's kappa
  A kappa >= 0.6 on overall_pass is our bar for "the headline judge number
  means something".
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.config import load_config, rel
from src.llm import LLM
from .judge import DIMS, judge_reply

_BOOL = {"true": True, "1": True, "yes": True, "y": True,
         "false": False, "0": False, "no": False, "n": False}


def _to_bool(x) -> bool:
    return _BOOL.get(str(x).strip().lower(), False)


def run(worksheet: str, out_json: str | None = None) -> dict:
    rows = list(csv.DictReader(open(worksheet, encoding="utf-8")))
    rows = [r for r in rows if (r.get("groundedness") or "").strip() != ""]
    if not rows:
        raise SystemExit(f"{worksheet}: no human-scored rows (fill the score columns)")

    llm = LLM(judge=True)
    human, model = {d: [] for d in DIMS}, {d: [] for d in DIMS}
    hpass, mpass = [], []
    detail = []
    for r in rows:
        mj = judge_reply(r["message"], r.get("reference", ""), r["candidate"], llm=llm)
        for d in DIMS:
            human[d].append(int(float(r[d])))
            model[d].append(int(mj[d]))
        hpass.append(_to_bool(r["overall_pass"]))
        mpass.append(bool(mj["overall_pass"]))
        detail.append({"id": r.get("id", ""), "human": {d: int(float(r[d])) for d in DIMS},
                       "model": {d: int(mj[d]) for d in DIMS},
                       "human_pass": hpass[-1], "model_pass": mpass[-1]})

    per_dim = {}
    for d in DIMS:
        h, m = np.array(human[d]), np.array(model[d])
        rho = spearmanr(h, m).correlation if len(set(h.tolist())) > 1 and len(set(m.tolist())) > 1 else None
        per_dim[d] = {
            "exact_agreement": round(float(np.mean(h == m)), 3),
            "within_1": round(float(np.mean(np.abs(h - m) <= 1)), 3),
            "mae": round(float(np.mean(np.abs(h - m))), 3),
            "spearman_rho": None if rho is None or np.isnan(rho) else round(float(rho), 3),
            "human_mean": round(float(h.mean()), 2), "model_mean": round(float(m.mean()), 2),
        }

    pass_kappa = (round(float(cohen_kappa_score(hpass, mpass)), 3)
                  if len(set(hpass)) > 1 and len(set(mpass)) > 1 else None)
    result = {
        "n": len(rows),
        "per_dimension": per_dim,
        "overall_pass": {
            "accuracy": round(float(np.mean(np.array(hpass) == np.array(mpass))), 3),
            "cohen_kappa": pass_kappa,
            "human_pass_rate": round(float(np.mean(hpass)), 3),
            "model_pass_rate": round(float(np.mean(mpass)), 3),
        },
        "verdict": _verdict(pass_kappa, per_dim),
        "detail": detail,
    }
    out_json = out_json or str(rel(load_config()["paths"]["reports_dir"]) / "judge_agreement.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "detail"}, indent=2))
    print(f"-> {out_json}")
    return result


def _verdict(kappa, per_dim) -> str:
    if kappa is None:
        return "inconclusive (need both pass/fail classes in the human labels)"
    safety = per_dim["correctness_safety"]["within_1"]
    if kappa >= 0.6 and safety >= 0.8:
        return f"judge is trustworthy for headline use (pass kappa={kappa}, safety within-1={safety})"
    if kappa >= 0.4:
        return f"judge is directional only (pass kappa={kappa}); report with a caveat"
    return f"judge disagrees with the human too often (pass kappa={kappa}); do not headline it"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet",
                    default=str(rel(load_config()["eval"]["judge_human_scores"])))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.worksheet, a.out)


if __name__ == "__main__":
    main()
