"""Emit a CSV of replies for a human to score, so we can measure judge trust.

Pulls candidate replies from three sources (agent, retrieval-only, canned) over
a sample of the golden set, shuffles, and writes an unscored worksheet. The
human fills the score columns; eval/judge_agreement.py then re-scores the same
rows with the LLM judge and reports agreement.

    python scripts/make_judge_worksheet.py --n 12   # -> 36 rows (3 sources x 12)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, rel                       # noqa: E402
from src.data_prep import read_jsonl                          # noqa: E402
from src.agent import SupportAgent                            # noqa: E402
from src.retrieve import Retriever                            # noqa: E402
from eval import baselines as B                               # noqa: E402

COLS = ["id", "source", "message", "reference", "candidate",
        "groundedness", "relevance", "correctness_safety", "tone",
        "completeness", "overall_pass"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    cfg = load_config()
    golden = read_jsonl(rel(cfg["eval"]["golden_set"]))
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(golden), size=min(a.n, len(golden)), replace=False)
    sample = [golden[int(i)] for i in pick]

    retr = Retriever.from_config()
    agent = SupportAgent(retriever=retr)
    rsimple = B.reply_simple(retr, k=cfg["retrieval"]["k"])

    rows = []
    for g in sample:
        m, ref = g["message"], g.get("reference_resolution", "")
        ar = agent.handle(m, thread_context=g.get("thread_context", ""))
        for src, cand in [("agent", ar.draft_reply),
                          ("retrieval_only", rsimple(m)),
                          ("canned", B.CANNED_REPLY)]:
            rows.append({"id": g["id"], "source": src, "message": m,
                         "reference": ref, "candidate": cand,
                         **{c: "" for c in COLS[5:]}})
    rng.shuffle(rows)

    out = rel(cfg["paths"]["golden_dir"]) / "judge_worksheet.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")
    print("Fill groundedness/relevance/correctness_safety/tone/completeness (1-5) "
          "and overall_pass (true/false), then save as "
          f"{rel(cfg['eval']['judge_human_scores'])}")


if __name__ == "__main__":
    main()
