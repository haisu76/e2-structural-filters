#!/usr/bin/env python3
"""Build the G2 representative-pair frame and the 65-pair blind audit pack.

Implements amendment 2, clause G2-F4: 13 pairs per condition (all if fewer),
drawn from the G2 representative frame by a fixed hash order, with blind IDs
that hide seed and condition from coder C.

Dedupe rule: the last logged record per (seed, condition, candidate), matching
the G1 pipeline and the 2026-11-18 amendment snapshot (G2-F1/F2).

Outputs
  e2_validation/G2_representative_pairs.csv        (PI use; 481 pairs expected)
  e2_validation/audit_G2/G2_audit_65_pairs.csv     (blind sheet for coder C)
  e2_validation/audit_G2/README_FIRST.md
  e2_validation/audit_G2/HOW_TO_CODE_G2.md
  e2_validation/PI_only/G2_audit_65_key.csv
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from e2_prepare_inclusion import load_latest, representative_pairs

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "e2_generation" / "wave1_qwen.jsonl"
OUT = ROOT / "e2_validation"
AUDIT_DIR = OUT / "audit_G2"
KEY_DIR = OUT / "PI_only"
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
PER_COND = 13


def h(*parts: str) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def main() -> None:
    latest = load_latest(GEN)
    reps = representative_pairs(latest)
    print(f"representative pairs: {len(reps)}")
    per = Counter(cond for _seed, cond in reps)
    print({c: per.get(c, 0) for c in CONDS})

    rows = []
    for seed, cond in reps:
        row = reps[(seed, cond)]
        rows.append({
            "seed_id": seed,
            "condition": cond,
            "candidate": int(row["candidate"]),
            "pair_id": row.get("pair_id", ""),
            "length_ratio": row.get("length_ratio", ""),
            "subject_original": row.get("subject_original", ""),
            "body_original": row.get("original", ""),
            "subject_rewrite": row.get("parsed_subject", ""),
            "body_rewrite": row.get("parsed_body", ""),
        })
    rows.sort(key=lambda r: (CONDS.index(r["condition"]), h("audit_g2", r["seed_id"], r["condition"])))

    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    KEY_DIR.mkdir(parents=True, exist_ok=True)

    rep_path = OUT / "G2_representative_pairs.csv"
    with rep_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Audit sample: first PER_COND per condition in the frozen hash order.
    seen = Counter()
    audit = []
    for row in rows:
        cond = row["condition"]
        if seen[cond] >= PER_COND:
            continue
        seen[cond] += 1
        audit.append(row)

    blind_rows = []
    key_rows = []
    used = set()
    for row in audit:
        blind = h("blind", "G2", row["seed_id"], row["condition"])[:12]
        while blind in used:
            blind = h("blind", "G2", row["seed_id"], row["condition"], "x")[:12]
        used.add(blind)
        blind_rows.append({
            "blind_id": blind,
            "subject_original": row["subject_original"],
            "body_original": row["body_original"],
            "subject_rewrite": row["subject_rewrite"],
            "body_rewrite": row["body_rewrite"],
            "valid": "",
            "notes": "",
        })
        key_rows.append({
            "blind_id": blind,
            "seed_id": row["seed_id"],
            "condition": row["condition"],
            "candidate": row["candidate"],
        })

    blind_rows.sort(key=lambda r: r["blind_id"])
    sheet = AUDIT_DIR / "G2_audit_65_pairs.csv"
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(blind_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blind_rows)

    key_path = KEY_DIR / "G2_audit_65_key.csv"
    with key_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"audit pairs: {len(blind_rows)}  {dict(Counter(r['condition'] for r in audit))}")
    print(f"wrote {rep_path}")
    print(f"wrote {sheet}")
    print(f"wrote {key_path}")


if __name__ == "__main__":
    main()
