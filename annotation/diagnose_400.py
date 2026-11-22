#!/usr/bin/env python
"""Diagnostics for the E1 400-item annotation batch.

Reads the three normalized 400-item rating files and the three 50-item
calibration files, then writes the tables used in `e1_400_findings.md`:

- scale_use.csv             per rater x item mean/sd/mode share, plus the
                            difference from the same rater's calibration mean
- dimension_agreement.csv   absolute agreement vs rank-order consistency
- dimension_pairwise_icc.csv pairwise ICC(2,2) and ICC(3,2)
- item_pairwise_icc.csv     per-item pairwise ICC(2,2) and ICC(3,2) (v2.0)
- within_cell_icc.csv       ICC(2,3) inside each of the four cells
- threat_agreement.csv      threat prevalence and pairwise Cohen's kappa

Usage:
    python annotation/diagnose_400.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_calibration as ac
from rating_schema import DIMENSION_ITEMS, ITEMS


def scale_use(ratings: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in ITEMS:
        cal_means = calibration.groupby("rater_id")[item].mean()
        for rater, sub in ratings.groupby("rater_id"):
            counts = sub[item].value_counts()
            mode_share = float(counts.iloc[0] / counts.sum()) if len(counts) else np.nan
            rows.append({
                "item": item,
                "rater_id": rater,
                "mean": float(sub[item].mean()),
                "sd": float(sub[item].std(ddof=1)),
                "mode": int(counts.index[0]) if len(counts) else -1,
                "mode_share": mode_share,
                "calibration_mean": float(cal_means.get(rater, np.nan)),
                "diff_400_minus_cal": float(sub[item].mean() - cal_means.get(rater, np.nan)),
            })
    return pd.DataFrame(rows)


def consistency_icc(matrix: np.ndarray) -> float:
    """ICC(3,k): two-way mixed, average raters, consistency."""
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ssr = k * np.sum((row_means - grand) ** 2)
    sse = np.sum((matrix - row_means[:, None] - col_means[None, :] + grand) ** 2)
    msr = ssr / (n - 1)
    mse = sse / ((n - 1) * (k - 1))
    if msr <= 0:
        return float("nan")
    return float((msr - mse) / msr)


def dimension_scores(ratings: pd.DataFrame) -> pd.DataFrame:
    out = ratings.copy()
    for dimension, items in DIMENSION_ITEMS.items():
        scored = [f"{item}__scored" for item in items]
        out[f"{dimension}_score"] = out[scored].mean(axis=1)
    return out


def wide_scale(df: pd.DataFrame, dimension: str, subset=None) -> pd.DataFrame:
    data = df if subset is None else subset
    return (data.pivot_table(index="calib_id", columns="rater_id",
                             values=f"{dimension}_score")
            .dropna())


def dimension_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSION_ITEMS:
        wide = wide_scale(df, dimension)
        rows.append({
            "dimension": dimension,
            "n": len(wide),
            "icc_absolute": ac.icc_2k(wide.to_numpy()),
            "icc_consistency": consistency_icc(wide.to_numpy()),
        })
    return pd.DataFrame(rows)


def pairwise_icc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSION_ITEMS:
        wide = wide_scale(df, dimension)
        for a, b in (("R1", "R2"), ("R1", "R3"), ("R2", "R3")):
            matrix = wide[[a, b]].to_numpy()
            rows.append({
                "dimension": dimension,
                "pair": f"{a}-{b}",
                "icc_absolute": ac.icc_2k(matrix),
                "icc_consistency": consistency_icc(matrix),
            })
    return pd.DataFrame(rows)


def item_pairwise_icc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in ITEMS:
        wide = (df.pivot_table(index="calib_id", columns="rater_id", values=item)
                .dropna())
        for a, b in (("R1", "R2"), ("R1", "R3"), ("R2", "R3")):
            matrix = wide[[a, b]].to_numpy()
            rows.append({
                "item": item,
                "pair": f"{a}-{b}",
                "icc_absolute": ac.icc_2k(matrix),
                "icc_consistency": consistency_icc(matrix),
            })
    return pd.DataFrame(rows)


def within_cell_icc(df: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    rows = []
    for cell in cells:
        subset = df[df["cell"] == cell]
        for dimension in DIMENSION_ITEMS:
            wide = wide_scale(df, dimension, subset)
            rows.append({
                "cell": cell,
                "dimension": dimension,
                "n": len(wide),
                "icc_absolute": ac.icc_2k(wide.to_numpy()) if len(wide) >= 2 else np.nan,
            })
    return pd.DataFrame(rows)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    observed = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    expected = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - expected) < 1e-12:
        return float("nan")
    return (observed - expected) / (1 - expected)


def threat_agreement(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="calib_id", columns="rater_id", values="threat_binary").dropna()
    rows = [
        {"measure": f"prevalence_{rater}", "value": float(wide[rater].mean())}
        for rater in ("R1", "R2", "R3")
    ]
    for a, b in (("R1", "R2"), ("R1", "R3"), ("R2", "R3")):
        rows.append({"measure": f"agreement_{a}_{b}",
                     "value": float((wide[a] == wide[b]).mean())})
        rows.append({"measure": f"kappa_{a}_{b}",
                     "value": cohen_kappa(wide[a], wide[b])})
    rows.append({"measure": "all_three_agree",
                 "value": float(((wide.R1 == wide.R2) & (wide.R2 == wide.R3)).mean())})
    rows.append({"measure": "all_three_positive",
                 "value": float(((wide.R1 == 1) & (wide.R2 == 1) & (wide.R3 == 1)).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", nargs=3, type=Path,
                        default=[Path("annotation/e1_400_R1.csv"),
                                 Path("annotation/e1_400_R2.csv"),
                                 Path("annotation/e1_400_R3.csv")])
    parser.add_argument("--calibration", nargs=3, type=Path,
                        default=[Path("annotation/calibration_R1-v1.1.csv"),
                                 Path("annotation/calibration_R2-v1.1.csv"),
                                 Path("annotation/calibration_R3-v1.1.csv")])
    parser.add_argument("--items", type=Path,
                        default=Path("samples/e1_items.csv"))
    parser.add_argument("--map", type=Path, default=None,
                        help="QC map with sheet_id/is_anchor; anchor rows are dropped")
    parser.add_argument("--out", type=Path,
                        default=Path("annotation/e1_400_results/diagnostics"))
    args = parser.parse_args()

    ratings = ac.load_ratings(args.ratings, args.map)
    calibration = ac.load_ratings(args.calibration)
    meta = pd.read_csv(args.items)[["sample_id", "cell"]]
    meta = meta.rename(columns={"sample_id": "calib_id"})
    ratings = ratings.merge(meta, on="calib_id", how="left")
    ratings = dimension_scores(ratings)

    args.out.mkdir(parents=True, exist_ok=True)
    tables = {
        "scale_use.csv": scale_use(ratings, calibration),
        "dimension_agreement.csv": dimension_agreement(ratings),
        "dimension_pairwise_icc.csv": pairwise_icc(ratings),
        "item_pairwise_icc.csv": item_pairwise_icc(ratings),
        "within_cell_icc.csv": within_cell_icc(ratings, sorted(meta["cell"].unique())),
        "threat_agreement.csv": threat_agreement(ratings),
    }
    for name, table in tables.items():
        table.to_csv(args.out / name, index=False)
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
