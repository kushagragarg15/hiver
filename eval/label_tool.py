"""Terminal labelling tool for the golden set.

Reads  data/golden/golden_candidates.jsonl
Writes data/golden/golden_set.jsonl        (append-only, resumable)

For each candidate it shows the message, the full thread, what the brand
actually did, and the heuristic pre-label. You press Enter to accept the
pre-label or type a correction. Every row you touch is stamped
label_source="human".

    python -m eval.label_tool                 # resume where you left off
    python -m eval.label_tool --limit 200
    python -m eval.label_tool --review-actions # 2nd pass: only re-check actions

Keys:
  intent:  1-8 (menu shown) or the slug; Enter = keep pre-label
  action:  a = auto, e = escalate; Enter = keep
  reason:  free text (why auto/escalate) -- required when you CHANGE the action
  notes:   free text, optional
  s = skip (don't write), q = save & quit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, rel                       # noqa: E402
from src.data_prep import read_jsonl                          # noqa: E402
from src.intents import INTENTS, SLUGS                        # noqa: E402

MENU = {str(i + 1): s.slug for i, s in enumerate(INTENTS)}


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except EOFError:
        return "q"


def _show(row: dict) -> None:
    print("\n" + "=" * 78)
    print(f'{row["id"]}   thread {row["thread_id"]}   stratum={row.get("stratum","?")}')
    print("-" * 78)
    print("THREAD:")
    for line in row.get("thread_context", "").splitlines():
        print("  " + line)
    print("-" * 78)
    print("MESSAGE TO LABEL:\n  " + row["message"])
    print("-" * 78)
    print("BRAND ACTUALLY DID:\n  " + (row.get("reference_resolution", "") or "(nothing on record)"))
    print("-" * 78)
    print("intent menu: " + "  ".join(f"{k}={v}" for k, v in MENU.items()))
    print(f'pre-label -> intent={row.get("gold_intent")}  action={row.get("gold_action")}')


def label_one(row: dict) -> dict | None:
    _show(row)
    raw = _prompt("intent [Enter=keep / 1-8 / slug / s / q]: ")
    if raw == "q":
        return "quit"          # type: ignore
    if raw == "s":
        return None
    intent = row.get("gold_intent")
    if raw in MENU:
        intent = MENU[raw]
    elif raw in SLUGS:
        intent = raw
    elif raw:
        print("  ! unknown intent, keeping pre-label")

    act_raw = _prompt(f"action [Enter=keep {row.get('gold_action')} / a / e]: ").lower()
    action = row.get("gold_action")
    if act_raw in ("a", "auto"):
        action = "auto"
    elif act_raw in ("e", "esc", "escalate"):
        action = "escalate"

    changed_action = action != row.get("gold_action")
    reason = row.get("gold_action_reason", "")
    if changed_action or not reason:
        reason = _prompt("reason (why this action): ") or reason or f"labeller judgement: {action}"
    notes = _prompt("notes (optional): ") or row.get("notes", "")

    return {**row, "gold_intent": intent, "gold_action": action,
            "gold_action_reason": reason, "notes": notes, "label_source": "human"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config()
    cand_path = Path(a.candidates or rel(cfg["paths"]["golden_dir"]) / "golden_candidates.jsonl")
    out_path = Path(a.out or rel(cfg["eval"]["golden_set"]))
    if not cand_path.exists():
        raise SystemExit(f"{cand_path} missing -- run scripts/make_golden_candidates.py first")

    cands = read_jsonl(cand_path)
    done_ids = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in cands if r["id"] not in done_ids]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(done_ids)} already labelled, {len(todo)} to go "
          f"({len(cands)} candidates total)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for row in todo:
            res = label_one(row)
            if res == "quit":
                break
            if res is None:
                continue
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
    print(f"\nwrote {written} new labels -> {out_path}")
    total = len(read_jsonl(out_path))
    print(f"golden set now has {total} rows "
          f"({'OK' if 150 <= total <= 260 else 'aim for 150-250'})")


if __name__ == "__main__":
    main()
