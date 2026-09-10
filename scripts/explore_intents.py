"""Evidence for the intent taxonomy: cluster inbound openings and print exemplars.

Uses TF-IDF + KMeans by default (no model download). With --embed it uses
sentence-transformers/all-MiniLM-L6-v2 if installed. The output is a starting
point for a human to name buckets -- the named result lives in src/intents.py.

    python scripts/explore_intents.py --k 12 --sample 800
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, rel                       # noqa: E402
from src.data_prep import read_jsonl                          # noqa: E402
from src.intents import keyword_classify                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    cfg = load_config()
    brand = cfg["data_prep"]["brand"]
    pool = read_jsonl(rel(cfg["paths"]["processed_dir"]) / f"{brand}_history.jsonl")
    rng = np.random.default_rng(a.seed)
    texts = [r["customer_opening"] for r in pool if len(r["customer_opening"].split()) >= 4]
    if len(texts) > a.sample:
        texts = [texts[i] for i in rng.choice(len(texts), a.sample, replace=False)]

    if a.embed:
        from sentence_transformers import SentenceTransformer
        X = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 2),
                            stop_words="english").fit_transform(texts)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=a.k, random_state=a.seed, n_init=10).fit(X)
    labels = km.labels_

    print(f"\n{brand}: {len(texts)} openings -> {a.k} clusters "
          f"({'embeddings' if a.embed else 'tf-idf'})\n" + "=" * 70)
    for c in range(a.k):
        idx = np.where(labels == c)[0]
        kw = Counter(keyword_classify(texts[i]) for i in idx).most_common(2)
        print(f"\n--- cluster {c}  (n={len(idx)})  keyword-guess≈{kw}")
        for i in idx[:6]:
            print("   ", re.sub(r"\s+", " ", texts[i])[:150])
    print("\n" + "=" * 70)
    print("keyword-labeller distribution over the whole sample:")
    for slug, n in Counter(keyword_classify(t) for t in texts).most_common():
        print(f"  {slug:<32} {n:>4}  ({n/len(texts):.0%})")


if __name__ == "__main__":
    main()
