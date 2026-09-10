"""Retrieval over the brand's historical resolved threads.

The index key is the *customer side* of a past thread (opening + all customer
turns). The payload is that thread's brand resolution -- the text the drafter
grounds its reply in. BM25 (bag-of-words, no embedding model to download) keeps
the repro dependency-light and fast; a TF-IDF cosine fallback is included.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import load_config, rel
from .data_prep import read_jsonl

_TOK = re.compile(r"[a-z0-9']+")


def _tok(s: str) -> list[str]:
    return _TOK.findall((s or "").lower())


@dataclass
class Exemplar:
    thread_id: int
    score: float
    customer: str
    resolution: str
    n_turns: int


class Retriever:
    def __init__(self, records: list[dict], method: str = "bm25"):
        self.records = records
        self.method = method
        self._corpus_txt = [
            f"{r.get('customer_opening','')} {r.get('customer_all','')}"
            for r in records
        ]
        if method == "bm25":
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_tok(t) for t in self._corpus_txt])
        elif method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2),
                                        sublinear_tf=True)
            self._mat = self._vec.fit_transform(self._corpus_txt)
        else:
            raise ValueError(method)

    @classmethod
    def from_config(cls, brand: str | None = None) -> "Retriever":
        cfg = load_config()
        brand = brand or cfg["data_prep"]["brand"]
        path = rel(cfg["paths"]["processed_dir"]) / f"{brand}_history.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run: python -m src.data_prep")
        return cls(read_jsonl(path), method=cfg["retrieval"]["method"])

    def search(self, query: str, k: int = 4) -> list[Exemplar]:
        if self.method == "bm25":
            scores = self._bm25.get_scores(_tok(query))
        else:
            from sklearn.metrics.pairwise import linear_kernel
            qv = self._vec.transform([query])
            scores = linear_kernel(qv, self._mat).ravel()
        order = scores.argsort()[::-1][:k]
        out = []
        for i in order:
            r = self.records[int(i)]
            out.append(Exemplar(
                thread_id=r.get("thread_id", -1),
                score=round(float(scores[int(i)]), 3),
                customer=r.get("customer_opening", ""),
                resolution=r.get("brand_resolution", ""),
                n_turns=r.get("n_turns", 0),
            ))
        return out
