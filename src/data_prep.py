"""twcs.csv  ->  per-brand reconstructed threads + a chronological split.

Why chronological split?
    The agent is allowed to "remember" how the brand resolved past issues.
    If we let it retrieve from the same threads we score it on, the reply
    task collapses to copy-paste and the numbers are meaningless. So every
    thread whose FIRST message is before `split_date` goes into the agent's
    memory (`history`), and everything on/after is the pool we sample the
    golden set from (`eval_pool`). No thread is in both.

Output (data/processed/):
    {brand}_history.jsonl     retrieval memory (customer opening -> brand resolution)
    {brand}_eval_pool.jsonl   held-out cases to sample the golden set from
    {brand}_stats.json        counts, date ranges, turn distribution

Run:
    python -m src.data_prep                 # uses config.yaml brand
    python -m src.data_prep --brand Xbox    # override
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, rel

TWEET_DATE_FMT = "%a %b %d %H:%M:%S %z %Y"
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str, drop_leading_mentions: bool = True) -> str:
    """Light normalisation. We keep URLs and emoji (they carry signal); we
    unescape HTML entities and drop the @handle noise that every tweet carries."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    if drop_leading_mentions:
        # strip a run of @mentions only at the very start ("@AppleSupport @123 hi" -> "hi")
        t = re.sub(r"^(?:\s*@\w+)+\s*", "", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _load_graph_columns(csv_path: Path, chunksize: int) -> pd.DataFrame:
    """Pass 1: cheap columns only, to reconstruct the reply graph."""
    frames = []
    cols = ["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"]
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize,
                             dtype={"tweet_id": "int64", "author_id": "string"}):
        chunk["inbound"] = chunk["inbound"].astype("string").str.lower().eq("true")
        chunk["parent"] = pd.to_numeric(chunk["in_response_to_tweet_id"],
                                        errors="coerce")
        frames.append(chunk[["tweet_id", "author_id", "inbound", "parent"]])
    return pd.concat(frames, ignore_index=True)


def _roots(tweet_ids: np.ndarray, parent_ids: np.ndarray) -> np.ndarray:
    """Follow parent pointers to the thread root via vectorised pointer-jumping.

    Nodes with no (in-dataset) parent point at themselves, so repeatedly
    applying `parent = parent[parent]` converges to the root in O(log depth)
    passes -- no Python-level per-row loop over 3M tweets.
    """
    n = len(tweet_ids)
    idx_of = {int(t): i for i, t in enumerate(tweet_ids)}
    parent_idx = np.arange(n, dtype=np.int64)          # self-loop = root
    for i in range(n):
        p = parent_ids[i]
        if not np.isnan(p):
            j = idx_of.get(int(p))
            if j is not None:
                parent_idx[i] = j
    for _ in range(64):                                # 2**64 depth ceiling
        nxt = parent_idx[parent_idx]
        if np.array_equal(nxt, parent_idx):
            break
        parent_idx = nxt
    return parent_idx


def build_threads(brand: str, cfg: dict) -> dict:
    dp = cfg["data_prep"]
    csv_path = rel(cfg["paths"]["raw_csv"])
    if not csv_path.exists():
        sys.exit(f"raw csv not found at {csv_path} -- fix paths.raw_csv in config.yaml")

    print(f"[data_prep] pass 1: reply graph from {csv_path.name} ...")
    g = _load_graph_columns(csv_path, chunksize=400_000)
    tweet_ids = g["tweet_id"].to_numpy()
    root_idx = _roots(tweet_ids, g["parent"].to_numpy(dtype="float64"))
    root_id = tweet_ids[root_idx]

    is_brand = (g["author_id"].str.lower() == brand.lower()).to_numpy()
    is_inbound = g["inbound"].to_numpy()

    # roots that contain at least one brand tweet AND one customer tweet
    by_root_brand: dict[int, bool] = defaultdict(bool)
    by_root_cust: dict[int, bool] = defaultdict(bool)
    for r, b, inb in zip(root_id, is_brand, is_inbound):
        r = int(r)
        if b:
            by_root_brand[r] = True
        if inb:
            by_root_cust[r] = True
    keep_roots = {r for r in by_root_brand if by_root_cust.get(r)}
    keep_ids = {int(t): int(r) for t, r in zip(tweet_ids, root_id) if int(r) in keep_roots}
    print(f"[data_prep] {len(keep_roots):,} candidate threads, {len(keep_ids):,} tweets")

    print("[data_prep] pass 2: materialising thread text ...")
    rows_by_root: dict[int, list] = defaultdict(list)
    for chunk in pd.read_csv(csv_path, chunksize=400_000,
                             dtype={"tweet_id": "int64", "author_id": "string"}):
        chunk = chunk[chunk["tweet_id"].isin(keep_ids)]
        if chunk.empty:
            continue
        chunk["ts"] = pd.to_datetime(chunk["created_at"], format=TWEET_DATE_FMT,
                                     errors="coerce", utc=True)
        chunk["inbound"] = chunk["inbound"].astype("string").str.lower().eq("true")
        for row in chunk.itertuples(index=False):
            rows_by_root[keep_ids[int(row.tweet_id)]].append(row)

    threads = []
    for root, rows in rows_by_root.items():
        rows = sorted(rows, key=lambda r: (r.ts is pd.NaT, r.ts))
        turns = [{
            "tweet_id": int(r.tweet_id),
            "role": "customer" if r.inbound else "agent",
            "author": str(r.author_id),
            "ts": None if pd.isna(r.ts) else r.ts.isoformat(),
            "text": clean_text(r.text, drop_leading_mentions=True),
        } for r in rows]
        cust = [t for t in turns if t["role"] == "customer"]
        agent = [t for t in turns if t["role"] == "agent"
                 and t["author"].lower() == brand.lower()]
        if len(turns) < dp["min_thread_turns"] or not cust or not agent:
            continue
        ts_vals = [t["ts"] for t in turns if t["ts"]]
        threads.append({
            "thread_id": int(root),
            "brand": brand,
            "n_turns": len(turns),
            "first_ts": min(ts_vals) if ts_vals else None,
            "last_ts": max(ts_vals) if ts_vals else None,
            "customer_opening": cust[0]["text"],
            "customer_all": " || ".join(c["text"] for c in cust),
            "brand_resolution": " || ".join(a["text"] for a in agent),
            "turns": turns,
        })

    threads = [t for t in threads if t["first_ts"]]
    threads.sort(key=lambda t: t["first_ts"])
    split = pd.Timestamp(dp["split_date"], tz="UTC").isoformat()
    history = [t for t in threads if t["first_ts"] < split]
    eval_pool = [t for t in threads if t["first_ts"] >= split]

    rng = np.random.default_rng(dp["random_seed"])
    if dp.get("max_history_threads") and len(history) > dp["max_history_threads"]:
        # sample ACROSS the whole pre-split period, not just the tail -- a
        # 3-day retrieval memory would miss most precedent.
        keep = rng.choice(len(history), dp["max_history_threads"], replace=False)
        history = [history[i] for i in sorted(keep)]
    if dp.get("eval_pool_size") and len(eval_pool) > dp["eval_pool_size"]:
        pick = rng.choice(len(eval_pool), dp["eval_pool_size"], replace=False)
        eval_pool = [eval_pool[i] for i in sorted(pick)]

    out = rel(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / f"{brand}_history.jsonl", history)
    _write_jsonl(out / f"{brand}_eval_pool.jsonl", eval_pool)

    stats = {
        "brand": brand,
        "threads_total": len(threads),
        "history_threads": len(history),
        "eval_pool_threads": len(eval_pool),
        "split_date": dp["split_date"],
        "date_min": threads[0]["first_ts"] if threads else None,
        "date_max": threads[-1]["first_ts"] if threads else None,
        "turn_dist": _pct({t["n_turns"] for t in threads}, [t["n_turns"] for t in threads]),
    }
    (out / f"{brand}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[data_prep] history={len(history):,}  eval_pool={len(eval_pool):,}  -> {out}")
    return stats


def _pct(_keys, values):
    s = pd.Series(values)
    return {
        "min": int(s.min()), "p50": int(s.median()),
        "p90": int(s.quantile(0.9)), "max": int(s.max()),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default=cfg["data_prep"]["brand"])
    args = ap.parse_args()
    build_threads(args.brand, cfg)


if __name__ == "__main__":
    main()
