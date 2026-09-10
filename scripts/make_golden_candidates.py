"""Sample candidate rows for the golden set from the held-out eval pool.

Sampling design (documented in data/golden/SAMPLING_NOTES.md):
  - Source: {brand}_eval_pool.jsonl only (post-split threads; never in retrieval).
  - Stratify by a rough intent guess (keyword weak-labeller) so rare intents
    (security, billing, complaint) are not swamped by the head.
  - Within each stratum, stratify again by conversation length
    (1 turn / 2-3 / 4+) so we get both one-shot questions and messy threads.
  - Light quality gate: drop < 4-word openings and pure link/emoji tweets.
  - Emit `target` rows with heuristic PRE-LABELS (keyword intent + intent's
    default action). These are NOT gold -- eval/label_tool.py is where a human
    confirms or overrides every one.

    python scripts/make_golden_candidates.py --target 240
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, rel                       # noqa: E402
from src.data_prep import read_jsonl                          # noqa: E402
from src.intents import BY_SLUG, keyword_classify             # noqa: E402

_LINK_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$")


def _len_bucket(n: int) -> str:
    return "1turn" if n <= 1 else ("2_3turn" if n <= 3 else "4plus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=240)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = load_config()
    brand = cfg["data_prep"]["brand"]
    pool = read_jsonl(rel(cfg["paths"]["processed_dir"]) / f"{brand}_eval_pool.jsonl")
    rng = np.random.default_rng(args.seed)

    # quality gate
    def ok(r):
        t = r["customer_opening"]
        return (len(t.split()) >= 4) and not _LINK_ONLY.match(t) and len(t) <= 500
    pool = [r for r in pool if ok(r)]

    # stratify: intent-guess x length-bucket
    strata: dict[tuple, list] = defaultdict(list)
    for r in pool:
        n_cust = sum(1 for tn in r["turns"] if tn["role"] == "customer")
        strata[(keyword_classify(r["customer_opening"]), _len_bucket(n_cust))].append(r)

    # proportional-ish allocation with a floor per non-empty stratum
    keys = list(strata)
    floor = 4
    alloc = {k: min(len(strata[k]), floor) for k in keys}
    remaining = args.target - sum(alloc.values())
    weights = np.array([max(0, len(strata[k]) - floor) for k in keys], float)
    if remaining > 0 and weights.sum() > 0:
        extra = np.floor(weights / weights.sum() * remaining).astype(int)
        for k, e in zip(keys, extra):
            alloc[k] = min(len(strata[k]), alloc[k] + int(e))

    picked = []
    for k in keys:
        pool_k = strata[k]
        idx = rng.choice(len(pool_k), size=min(alloc[k], len(pool_k)), replace=False)
        for i in idx:
            picked.append((k, pool_k[int(i)]))
    rng.shuffle(picked)

    out_path = rel(cfg["paths"]["golden_dir"]) / "golden_candidates.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for n, ((guess, lb), r) in enumerate(picked, 1):
            ctx = "\n".join(f'{t["role"]}: {t["text"]}' for t in r["turns"])
            row = {
                "id": f"gold_{n:04d}",
                "thread_id": r["thread_id"],
                "message": r["customer_opening"],
                "thread_context": ctx,
                "reference_resolution": r["brand_resolution"],
                "n_turns": r["n_turns"],
                "stratum": f"{guess}|{lb}",
                # heuristic pre-labels -- REVIEW REQUIRED
                "gold_intent": guess,
                "gold_action": BY_SLUG[guess].default_action if guess in BY_SLUG else "escalate",
                "gold_action_reason": "",
                "notes": "",
                "label_source": "heuristic_prelabel",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dist = defaultdict(int)
    for (g, lb), _ in [((k[0], k[1]), None) for k, _ in picked]:
        dist[g] += 1
    print(f"[candidates] {len(picked)} rows -> {out_path}")
    for g, c in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {g:<32} {c}")
    print("\nNext: python -m eval.label_tool   (review every pre-label)")


if __name__ == "__main__":
    main()
