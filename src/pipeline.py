"""Run the agent over a jsonl of messages -> predictions jsonl.

Input jsonl rows need at least {"message": "..."} (extra keys are passed through,
e.g. "id", "gold_intent"). Output adds the agent fields.

    python -m src.pipeline --in data/golden/golden_set.jsonl \
                           --out reports/preds_agent.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .agent import SupportAgent
from .data_prep import read_jsonl
from .llm import LLM


def run(in_path: str, out_path: str, limit: int | None = None) -> dict:
    rows = read_jsonl(in_path)
    if limit:
        rows = rows[:limit]
    agent = SupportAgent()
    out = []
    t0 = time.time()
    for r in tqdm(rows, desc="agent"):
        msg = r.get("message") or r.get("text") or r.get("customer_opening", "")
        res = agent.handle(msg, thread_context=r.get("thread_context", ""))
        out.append({**r, **res.to_dict()})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for o in out:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    meta = {"n": len(out), "seconds": round(time.time() - t0, 1),
            "llm_usage": agent.llm.usage.as_dict(), "out": out_path}
    print(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.in_path, a.out_path, a.limit)


if __name__ == "__main__":
    main()
