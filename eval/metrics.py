"""Automated metrics: intent classification + the escalate/auto decision.

Kept deliberately explicit (no hidden sklearn magic in the headline numbers) so
every figure in the report can be traced to a formula.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


# --------------------------- intent classification ------------------------- #
def intent_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    n = len(y_true)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / n if n else 0.0
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    per_class = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": round(prec, 3), "recall": round(rec, 3),
                          "f1": round(f1, 3), "support": sum(t == lab for t in y_true)}

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "n": n, "accuracy": round(acc, 3),
        "macro_f1": round(float(macro_f1), 3),
        "weighted_f1": round(float(weighted_f1), 3),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "pred_distribution": dict(Counter(y_pred)),
    }


# --------------------------- escalation decision -------------------------- #
def escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Positive class = 'escalate'. We care most about *missed* escalations
    (false auto) -- those are the ones that reach a customer unsupervised."""
    assert set(y_true) | set(y_pred) <= {"auto", "escalate"}
    tp = sum(t == "escalate" and p == "escalate" for t, p in zip(y_true, y_pred))
    fp = sum(t == "auto" and p == "escalate" for t, p in zip(y_true, y_pred))
    fn = sum(t == "escalate" and p == "auto" for t, p in zip(y_true, y_pred))
    tn = sum(t == "auto" and p == "auto" for t, p in zip(y_true, y_pred))
    n = len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "escalate_precision": round(prec, 3),
        "escalate_recall": round(rec, 3),
        "escalate_f1": round(f1, 3),
        "accuracy": round((tp + tn) / n, 3) if n else 0.0,
        "auto_rate": round((tn + fn) / n, 3) if n else 0.0,          # share sent unsupervised
        # the safety-critical number:
        "missed_escalation_rate": round(fn / (tp + fn), 3) if tp + fn else 0.0,
        "unnecessary_escalation_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 3) if n and len(set(y_true)) > 1 else None,
    }


# --------------------------- reply-quality rollup ------------------------- #
def judge_rollup(scores: list[dict], dims: list[str]) -> dict:
    if not scores:
        return {"n": 0}
    out = {"n": len(scores)}
    for d in dims:
        vals = [s[d] for s in scores if isinstance(s.get(d), (int, float))]
        out[d] = {"mean": round(float(np.mean(vals)), 3),
                  "p25": round(float(np.percentile(vals, 25)), 2),
                  "min": min(vals)} if vals else None
    passes = [bool(s.get("overall_pass")) for s in scores if "overall_pass" in s]
    out["pass_rate"] = round(sum(passes) / len(passes), 3) if passes else None
    return out
