"""Choose the brand to build the agent for -- data-driven, not vibes.

Two passes over twcs.csv:
  1. outbound-tweet volume per brand  -> shortlist the top N
  2. reconstruct the shortlist's threads and score each brand on properties
     that actually matter for an *auto-reply* agent:
       - n_threads                threads with >=1 customer + >=1 brand turn
       - pct_brand_closed         % of threads whose last turn is the brand's
       - median_turns             conversation depth
       - pct_in_channel_resolved  % of brand resolutions that DON'T immediately
                                  punt to DM / phone / email (i.e. the brand
                                  actually answered in-channel -> automatable)
       - intent_spread            entropy of the keyword weak-labeller over
                                  openings (higher = richer intent problem)

Score = 0.35*pct_in_channel_resolved + 0.25*pct_brand_closed
      + 0.20*norm(n_threads) + 0.20*norm(intent_spread)

Prints a ranked table and (with --write) sets data_prep.brand in config.yaml.

    python scripts/pick_brand.py --top 12 --write
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, rel                       # noqa: E402
from src.data_prep import TWEET_DATE_FMT, clean_text, _roots  # noqa: E402
from src.intents import SLUGS, keyword_classify               # noqa: E402

_PUNT = re.compile(r"\b(dm|d\.m\.|direct message|private message|pm us|"
                   r"call us|give us a call|email us|ring us|phone)\b", re.I)


def _norm(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def _entropy(counts: Counter) -> float:
    tot = sum(counts.values()) or 1
    ps = [c / tot for c in counts.values() if c]
    return -sum(p * math.log(p, 2) for p in ps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--write", action="store_true",
                    help="set data_prep.brand in config.yaml -- only if it is currently null")
    ap.add_argument("--force", action="store_true",
                    help="with --write: overwrite an already-set brand (breaks the committed "
                         "AppleSupport eval until you re-run prep/candidates/labelling)")
    args = ap.parse_args()

    cfg = load_config()
    csv_path = rel(cfg["paths"]["raw_csv"])
    chunk = 400_000

    print("[pick_brand] pass 1: outbound volume per brand ...")
    vol: Counter = Counter()
    for c in pd.read_csv(csv_path, usecols=["author_id", "inbound"],
                         chunksize=chunk, dtype={"author_id": "string"}):
        out = c[c["inbound"].astype("string").str.lower() == "false"]
        vol.update(out["author_id"].dropna().tolist())
    shortlist = [b for b, _ in vol.most_common(args.top)]
    print("  shortlist:", ", ".join(f"{b}({vol[b]:,})" for b in shortlist))

    print("[pick_brand] pass 2: reconstructing shortlist threads ...")
    g_cols = ["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"]
    G = []
    for c in pd.read_csv(csv_path, usecols=g_cols, chunksize=chunk,
                         dtype={"tweet_id": "int64", "author_id": "string"}):
        c["inbound"] = c["inbound"].astype("string").str.lower().eq("true")
        c["parent"] = pd.to_numeric(c["in_response_to_tweet_id"], errors="coerce")
        G.append(c[["tweet_id", "author_id", "inbound", "parent"]])
    G = pd.concat(G, ignore_index=True)
    tid = G["tweet_id"].to_numpy()
    root = tid[_roots(tid, G["parent"].to_numpy("float64"))]
    root_of = dict(zip(tid.tolist(), root.tolist()))
    author_of = dict(zip(tid.tolist(), G["author_id"].tolist()))

    # roots that involve each shortlisted brand
    brand_roots: dict[str, set] = defaultdict(set)
    for t, a in author_of.items():
        if a in shortlist:
            brand_roots[a].add(root_of[t])

    want_ids = set()
    for rs in brand_roots.values():
        want_ids |= rs
    # gather full rows for those threads
    rows_by_root: dict[int, list] = defaultdict(list)
    for c in pd.read_csv(csv_path, chunksize=chunk,
                         dtype={"tweet_id": "int64", "author_id": "string"}):
        c = c[c["tweet_id"].map(lambda x: root_of.get(x) in want_ids)]
        if c.empty:
            continue
        c["ts"] = pd.to_datetime(c["created_at"], format=TWEET_DATE_FMT,
                                 errors="coerce", utc=True)
        c["inbound"] = c["inbound"].astype("string").str.lower().eq("true")
        for r in c.itertuples(index=False):
            rows_by_root[root_of[int(r.tweet_id)]].append(r)

    metrics = []
    for brand in shortlist:
        n_threads = pct_closed = 0
        turns_list, kw = [], Counter()
        in_channel = 0
        for rt in brand_roots[brand]:
            rows = sorted(rows_by_root.get(rt, []),
                          key=lambda r: (pd.isna(r.ts), r.ts))
            if len(rows) < 2:
                continue
            cust = [r for r in rows if r.inbound]
            bt = [r for r in rows if (not r.inbound) and str(r.author_id) == brand]
            if not cust or not bt:
                continue
            n_threads += 1
            turns_list.append(len(rows))
            if not rows[-1].inbound and str(rows[-1].author_id) == brand:
                pct_closed += 1
            res = " ".join(clean_text(r.text) for r in bt)
            if not _PUNT.search(res):
                in_channel += 1
            kw[keyword_classify(clean_text(cust[0].text))] += 1
        if n_threads < 50:
            continue
        metrics.append({
            "brand": brand, "n_threads": n_threads,
            "pct_brand_closed": pct_closed / n_threads,
            "median_turns": float(np.median(turns_list)),
            "pct_in_channel_resolved": in_channel / n_threads,
            "intent_spread": _entropy(kw),
        })

    if not metrics:
        sys.exit("no brand cleared the 50-thread floor; widen --top")

    nt = _norm([m["n_threads"] for m in metrics])
    isp = _norm([m["intent_spread"] for m in metrics])
    for m, a, b in zip(metrics, nt, isp):
        m["score"] = round(0.35 * m["pct_in_channel_resolved"]
                           + 0.25 * m["pct_brand_closed"]
                           + 0.20 * a + 0.20 * b, 4)
    metrics.sort(key=lambda m: m["score"], reverse=True)

    hdr = f'{"brand":<16}{"threads":>9}{"brand_closed":>13}{"med_turns":>11}{"in_chan":>9}{"spread":>8}{"score":>8}'
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for m in metrics:
        print(f'{m["brand"]:<16}{m["n_threads"]:>9}{m["pct_brand_closed"]*100:>12.1f}%'
              f'{m["median_turns"]:>11.1f}{m["pct_in_channel_resolved"]*100:>8.1f}%'
              f'{m["intent_spread"]:>8.2f}{m["score"]:>8.3f}')

    winner = metrics[0]["brand"]
    print(f"\n[pick_brand] winner: {winner}")
    current = cfg["data_prep"].get("brand")
    if args.write and current and not args.force:
        # The committed repo is built around one brand (data, golden set, cache).
        # Silently switching it makes every downstream target fail, so refuse.
        print(f"[pick_brand] config.yaml already has data_prep.brand = {current!r}; "
              f"NOT overwriting (pass --force to switch brand -- then re-run prep, "
              f"candidates and labelling).")
    elif args.write:
        p = rel("config.yaml")
        txt = p.read_text(encoding="utf-8")
        txt = re.sub(r'(\n\s*brand:\s*)(?:"[^"]*"|null|~)', rf'\g<1>"{winner}"', txt, count=1)
        p.write_text(txt, encoding="utf-8")
        print(f"[pick_brand] wrote data_prep.brand = {winner} to config.yaml")
    # dump full table for the report
    out = rel(cfg["paths"]["reports_dir"]) / "brand_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[pick_brand] table -> {out}")


if __name__ == "__main__":
    main()
