#!/usr/bin/env python3
"""Build the base G2 inclusion table from the representative pairs.

Amendment 2, G2-F4 step 1: every representative pair is auto-included; the
65-pair audit and the frozen six-rule screen then subtract invalid pairs.
G2-F5: same schema as the G1 inclusion table.

Output
  e2_validation/G2_inclusion.csv   (481 rows)
"""
import csv
from collections import Counter
from pathlib import Path

from e2_prepare_inclusion import load_latest, representative_pairs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
GEN = ROOT / "e2_generation" / "wave1_qwen.jsonl"
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
EXPECT = 481

FIELDS = ["seed_id", "condition", "candidate", "blind_id", "source",
          "coder_A_valid", "coder_B_valid", "coder_A_notes", "coder_B_notes",
          "included", "inclusion_basis"]


def main() -> None:
    reps = representative_pairs(load_latest(GEN))
    rows = []
    for (seed, cond), row in sorted(reps.items()):
        rows.append({
            "seed_id": seed,
            "condition": cond,
            "candidate": str(int(row["candidate"])),
            "blind_id": "",
            "source": "auto_hard_validation",
            "coder_A_valid": "",
            "coder_B_valid": "",
            "coder_A_notes": "",
            "coder_B_notes": "",
            "included": "1",
            "inclusion_basis": "auto_hard_validation",
        })
    assert len(rows) == EXPECT, f"{len(rows)} representative pairs, expected {EXPECT}"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "G2_inclusion.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    per = Counter(r["condition"] for r in rows)
    print(f"wrote {path}  pairs={len(rows)}  per_condition={ {c: per[c] for c in CONDS} }")


if __name__ == "__main__":
    main()
