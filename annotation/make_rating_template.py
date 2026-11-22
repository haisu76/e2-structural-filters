#!/usr/bin/env python3
"""Build a rating template from a blinded sheet (50 calibration or 400 formal)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rating_schema import ITEMS, TEMPLATE_COLUMNS, THREAT_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an E1 rating template.")
    parser.add_argument("--items", default="samples/e1_calibration_50.csv")
    parser.add_argument("--out", default="annotation/e1_calibration_50_ratings_template.csv")
    parser.add_argument("--id-column", default="calib_id",
                        help="column copied into the template's calib_id field")
    parser.add_argument("--expected-rows", type=int, default=50)
    args = parser.parse_args()

    sheet = pd.read_csv(args.items, encoding="utf-8-sig")
    required = ["order", args.id_column, "subject", "body"]
    missing = [c for c in required if c not in sheet.columns]
    if missing:
        raise SystemExit(f"missing columns in {args.items}: {missing}")
    if args.expected_rows is not None and len(sheet) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, got {len(sheet)}")
    if sheet[args.id_column].duplicated().any():
        raise SystemExit(f"duplicate {args.id_column} in {args.items}")

    out = sheet[required].copy()
    if args.id_column != "calib_id":
        out = out.rename(columns={args.id_column: "calib_id"})
    out.insert(0, "rater_id", "")
    for column in ITEMS + THREAT_COLUMNS + ["confidence", "notes"]:
        out[column] = ""
    out = out[TEMPLATE_COLUMNS]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path} ({len(out)} rows, {len(out.columns)} columns)")
    print("rater workflow: copy this file per rater, fill rater_id (e.g. R1/R2/R3), "
          "then combine the filled copies for analysis.")


if __name__ == "__main__":
    main()
