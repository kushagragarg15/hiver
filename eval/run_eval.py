"""Headline evaluation harness.

Runs the agent + baselines over the golden set and writes:
  reports/results.json   machine-readable, committed as the headline record
  reports/results.md     the tables that go in the report

    python -m eval.run_eval                 # full golden set
    python -m eval.run_eval --limit 60      # fast path for the README repro
    python -m eval.run_eval --no-judge      # skip LLM-as-judge (metrics only)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.agent import SupportAgent
from src.config import llm_cfg, load_config, rel
from src.data_prep import read_jsonl
from src.intents import SLUGS
from src.llm import LLM
from src.retrieve import Retriever

from . import baselines as B
from .judge import DIMS, judge_reply
from .metrics import escalation_metrics, intent_metrics, judge_rollup


def _load_golden() -> list[dict]:
    cfg = load_config()
    path = rel(cfg["eval"]["golden_set"])
    if not path.exists():
        raise SystemExit(f"{path} missing -- build it with eval/label_tool.py "
                         f"(or use data/golden/golden_set.sample.jsonl for a smoke run)")
    rows = read_jsonl(path)
    bad = [r for r in rows if r.get("gold_intent") not in SLUGS
           or r.get("gold_action") not in ("auto", "escalate")]
    if bad:
        raise SystemExit(f"{len(bad)} golden rows have an invalid gold_intent/gold_action")
    return rows


def run(limit: int | None = None, do_judge: bool = True,
        judge_sample: int | None = None, out_dir: Path | None = None) -> dict:
    cfg = load_config()
    lc = llm_cfg()
    golden = _load_golden()
    if limit:
        golden = golden[:limit]
    n = len(golden)
    t0 = time.time()

    retriever = Retriever.from_config()
    agent = SupportAgent(retriever=retriever)
    llm_judge = LLM(judge=True) if do_judge else None

    y_intent = [g["gold_intent"] for g in golden]
    y_action = [g["gold_action"] for g in golden]
    messages = [g["message"] for g in golden]
    references = [g.get("reference_resolution", "") for g in golden]

    # ---- agent forward pass -------------------------------------------------
    agent_rows = []
    for g in tqdm(golden, desc="agent"):
        r = agent.handle(g["message"], thread_context=g.get("thread_context", ""))
        agent_rows.append(r)

    agent_intent = [r.intent for r in agent_rows]
    agent_action = [r.action for r in agent_rows]
    agent_reply = [r.draft_reply for r in agent_rows]
    ret_scores = [r.retrieved[0]["score"] if r.retrieved else 0.0 for r in agent_rows]

    # ---- baselines --------------------------------------------------------
    maj = B.intent_trivial(y_intent)          # NOTE: fit on gold => optimistic; see report caveat
    kw = B.intent_simple()
    b_intent_majority = [maj(m) for m in messages]
    b_intent_keyword = [kw(m) for m in messages]

    prior = B.escalation_by_intent_prior()
    b_act_always_auto = ["auto"] * n
    b_act_always_esc = ["escalate"] * n
    b_act_prior = [prior(intent=i) for i in agent_intent]   # uses agent's intent

    reply_simple = B.reply_simple(retriever, k=cfg["retrieval"]["k"])
    b_reply_retrieval = [reply_simple(m) for m in messages]
    b_reply_canned = [B.CANNED_REPLY] * n

    result = {
        "meta": {
            "brand": cfg["data_prep"]["brand"],
            "provider": lc["provider"], "model": lc["model"],
            "judge_provider": llm_judge.provider if llm_judge else None,
            "judge_model": llm_judge.model if llm_judge else None,
            "n_golden": n, "limit": limit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "intent": {
            "agent": intent_metrics(y_intent, agent_intent, SLUGS),
            "baseline_majority_trivial": intent_metrics(y_intent, b_intent_majority, SLUGS),
            "baseline_keyword_simple": intent_metrics(y_intent, b_intent_keyword, SLUGS),
        },
        "escalation": {
            "agent": escalation_metrics(y_action, agent_action),
            "baseline_always_auto": escalation_metrics(y_action, b_act_always_auto),
            "baseline_always_escalate": escalation_metrics(y_action, b_act_always_esc),
            "baseline_intent_prior": escalation_metrics(y_action, b_act_prior),
        },
    }

    # ---- reply quality (LLM-as-judge) -----------------------------------
    if do_judge:
        js = judge_sample or cfg["eval"].get("judge_sample") or n
        idx = list(range(min(js, n)))
        def _judge_all(cands):
            return [judge_reply(messages[i], references[i], cands[i], llm=llm_judge)
                    for i in tqdm(idx, desc="judge", leave=False)]
        result["reply_quality"] = {
            "judge_n": len(idx),
            "agent": judge_rollup(_judge_all(agent_reply), DIMS),
            "baseline_retrieval_only": judge_rollup(_judge_all(b_reply_retrieval), DIMS),
            "baseline_canned": judge_rollup(_judge_all(b_reply_canned), DIMS),
        }

    result["meta"]["seconds"] = round(time.time() - t0, 1)
    result["meta"]["llm_usage_agent"] = agent.llm.usage.as_dict()
    if llm_judge:
        result["meta"]["llm_usage_judge"] = llm_judge.usage.as_dict()

    result["headline"] = {
        "intent_macro_f1": result["intent"]["agent"]["macro_f1"],
        "intent_accuracy": result["intent"]["agent"]["accuracy"],
        "missed_escalation_rate": result["escalation"]["agent"]["missed_escalation_rate"],
        "unnecessary_escalation_rate": result["escalation"]["agent"]["unnecessary_escalation_rate"],
        "auto_rate": result["escalation"]["agent"]["auto_rate"],
        "reply_pass_rate": (result.get("reply_quality", {}).get("agent", {}) or {}).get("pass_rate"),
    }

    out_json = rel(cfg["eval"]["results_json"])
    out_md = rel(cfg["eval"]["results_md"])
    dbg = rel(cfg["paths"]["reports_dir"]) / "agent_golden_preds.jsonl"
    if out_dir is not None:
        out_json, out_md, dbg = (out_dir / p.name for p in (out_json, out_md, dbg))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    _write_md(result, out_md)

    # dump per-example agent output for failure analysis
    with open(dbg, "w", encoding="utf-8") as fh:
        for g, r in zip(golden, agent_rows):
            fh.write(json.dumps({**g, "pred": r.to_dict()}, ensure_ascii=False) + "\n")

    print(json.dumps(result["headline"], indent=2))
    print(f"\n-> {out_json}\n-> {out_md}\n-> {dbg}")
    return result


def _row(name, m):
    return (f"| {name} | {m.get('accuracy','-')} | {m.get('macro_f1','-')} | "
            f"{m.get('weighted_f1','-')} |")


def _write_md(r: dict, path: Path) -> None:
    warn = ([f"> ⚠️ **provider = `{r['meta']['provider']}` — these numbers are a "
             f"wiring check, not a quality signal.** `make eval` replays the "
             f"committed gemini run.", ""]
            if r["meta"]["provider"] == "mock" else [])
    L = warn + [f"# Results -- {r['meta']['brand']}", "",
         f"_{r['meta']['provider']}/{r['meta']['model']}, judge "
         f"{r['meta'].get('judge_model')}, n={r['meta']['n_golden']}, "
         f"{r['meta'].get('seconds','?')}s, generated {r['meta']['timestamp']}_", "",
         "## Intent classification", "",
         "| system | accuracy | macro-F1 | weighted-F1 |",
         "|---|---|---|---|",
         _row("agent (LLM)", r["intent"]["agent"]),
         _row("baseline: keyword (simple)", r["intent"]["baseline_keyword_simple"]),
         _row("baseline: majority (trivial)", r["intent"]["baseline_majority_trivial"]),
         "", "## Escalation decision (positive class = escalate)", "",
         "| system | esc-precision | esc-recall | esc-F1 | missed-esc-rate | unnec-esc-rate | auto-rate |",
         "|---|---|---|---|---|---|---|"]
    for name, key in [("agent", "agent"),
                      ("baseline: always-auto", "baseline_always_auto"),
                      ("baseline: always-escalate", "baseline_always_escalate"),
                      ("baseline: intent-prior", "baseline_intent_prior")]:
        m = r["escalation"][key]
        L.append(f"| {name} | {m['escalate_precision']} | {m['escalate_recall']} | "
                 f"{m['escalate_f1']} | {m['missed_escalation_rate']} | "
                 f"{m['unnecessary_escalation_rate']} | {m['auto_rate']} |")
    if "reply_quality" in r:
        rq = r["reply_quality"]
        L += ["", f"## Reply quality -- LLM-as-judge (n={rq['judge_n']})", "",
              "| system | " + " | ".join(DIMS) + " | pass-rate |",
              "|---|" + "|".join(["---"] * (len(DIMS) + 1)) + "|"]
        for name, key in [("agent", "agent"),
                          ("baseline: retrieval-only", "baseline_retrieval_only"),
                          ("baseline: canned", "baseline_canned")]:
            m = rq[key]
            cells = [f"{(m.get(d) or {}).get('mean','-')}" for d in DIMS]
            L.append(f"| {name} | " + " | ".join(cells) + f" | {m.get('pass_rate','-')} |")
    L += ["", "## Headline", "", "```json", json.dumps(r["headline"], indent=2), "```",
          "", "See REPORT.md for what these numbers hide."]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-sample", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write results here instead of the committed reports/ paths")
    a = ap.parse_args()
    run(limit=a.limit, do_judge=not a.no_judge, judge_sample=a.judge_sample,
        out_dir=a.out_dir)


if __name__ == "__main__":
    main()
