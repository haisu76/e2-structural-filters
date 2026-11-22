#!/usr/bin/env python3
"""Merge a targeted rerating sheet into the first-pass rating file.

The rerating sheet contains only rater_id/sheet_id/block/order/calib_id/
subject/body plus the flagged item columns. This script validates the return,
replaces only those columns in the full 23-column file, and keeps every other
column from the first pass.

Usage
  python e2_merge_reratings.py --rater R2 \
    --original e2_ratings/returns_v1/e2_600_R2_filled.csv \
    --rerate e2_ratings/packages/rerate_R2/e2_600_R2_rerate_filled.csv \
    --out e2_ratings/e2_600_R2_filled_v2.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "e2_ratings/e2_annotation_600_template.csv"
TARGETS = {
    "R1": ["V1_negative_tone"],
    "R2": ["V1_negative_tone", "D1_institutional", "D2_impersonal", "D3_procedural"],
    "R3": ["V1_negative_tone", "W1_care", "W2_friendly", "W3_person_not_ticket"],
}
IDS = ["rater_id", "sheet_id", "block", "order", "calib_id", "subject", "body"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", required=True, choices=sorted(TARGETS))
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--rerate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    template = pd.read_csv(TEMPLATE, encoding="utf-8-sig")
    original = pd.read_csv(args.original, encoding="utf-8-sig")
    rerate = pd.read_csv(args.rerate, encoding="utf-8-sig")
    targets = TARGETS[args.rater]

    assert original.shape == (640, 23), original.shape
    assert rerate.shape[0] == 640, rerate.shape
    assert set(rerate.columns) == set(IDS + targets), set(rerate.columns) ^ set(IDS + targets)
    assert original["sheet_id"].tolist() == rerate["sheet_id"].tolist() == template["sheet_id"].tolist()
    assert (original["rater_id"] == args.rater).all()
    assert (rerate["rater_id"] == args.rater).all()
    for column in ("subject", "body"):
        assert original[column].fillna("").tolist() == rerate[column].fillna("").tolist()

    for column in targets:
        values = pd.to_numeric(rerate[column], errors="coerce")
        if values.isna().any():
            raise SystemExit(f"{column}: missing or non-numeric values")
        if not values.between(1, 7).all() or (values % 1 != 0).any():
            raise SystemExit(f"{column}: values must be integers 1-7")
        rerate[column] = values.astype(int)

    merged = original.copy()
    changed = {}
    for column in targets:
        changed[column] = int((merged[column] != rerate[column]).sum())
        merged[column] = rerate[column].to_numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"{args.rater}: merged {targets} -> {args.out}")
    print(f"changed cells: {changed}")
    for column in targets:
        s = merged[column]
        print(f"  {column}: mean={s.mean():.2f} sd={s.std():.2f} "
              f"distinct={s.nunique()} mode_share={s.value_counts(normalize=True).iloc[0]:.3f}")


if __name__ == "__main__":
    main()
