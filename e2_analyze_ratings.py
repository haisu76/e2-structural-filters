#!/usr/bin/env python3
"""Manipulation check for the E2 600-item temperature subsample.

Loads the three filled rating sheets, drops hidden anchors via the QC map,
z-scores each rater x dimension across the 600 unique emails, and compares
each condition with the original within seed. Bootstrap resamples seeds.

Output
  e2_ratings/e2_manipulation_check.csv
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "annotation"))
import analyze_calibration as ac  # noqa: E402
from relative_scores import dimension_scores, zscore_within_rater  # noqa: E402

CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
DIMS = ["valence", "urgency", "warmth", "distance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", nargs="+", default=[
        "e2_ratings/e2_600_R1_filled.csv",
        "e2_ratings/e2_600_R2_filled.csv",
        "e2_ratings/e2_600_R3_filled.csv",
    ])
    ap.add_argument("--map", default="e2_ratings/e2_annotation_600_map.csv")
    ap.add_argument("--out", default="e2_ratings/e2_manipulation_check.csv")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mapping = pd.read_csv(args.map, encoding="utf-8-sig")
    meta = mapping.loc[mapping["is_anchor"] == 0, ["calib_id", "seed_id", "condition"]]
    df = ac.load_ratings([Path(p) for p in args.ratings], Path(args.map))
    df = df.merge(meta, on="calib_id", how="left", validate="many_to_one")
    if df["seed_id"].isna().any():
        raise SystemExit("some rated calib_id has no seed/condition in the map")

    raters = sorted(df["rater_id"].unique())
    counts = df.groupby("calib_id")["rater_id"].nunique()
    if not (counts == len(raters)).all():
        raise SystemExit("not every item was rated by every rater")
    if df.groupby("seed_id")["condition"].nunique().ne(6).any():
        raise SystemExit("a seed does not have exactly six versions")

    scored = dimension_scores(df)
    for dimension in DIMS:
        scored = zscore_within_rater(scored, dimension)

    rng = np.random.default_rng(args.seed)
    rows = []
    for dimension in DIMS:
        col = f"{dimension}_z"
        pivot = scored.pivot_table(index="seed_id", columns=["condition", "rater_id"],
                                   values=col, aggfunc="mean")
        for cond in CONDS:
            delta = pd.DataFrame({
                r: pivot[(cond, r)] - pivot[("original", r)] for r in raters
            }).dropna()
            if delta.empty:
                continue
            per_rater = delta.mean().to_dict()
            values = delta.mean(axis=1).to_numpy()
            means = [values[rng.integers(0, len(values), len(values))].mean()
                     for _ in range(args.bootstrap)]
            rows.append({
                "condition": cond,
                "dimension": dimension,
                "n_seeds": len(values),
                "mean_delta": values.mean(),
                "ci_low": np.percentile(means, 2.5),
                "ci_high": np.percentile(means, 97.5),
                **{f"mean_{r}": per_rater.get(r, np.nan) for r in raters},
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
