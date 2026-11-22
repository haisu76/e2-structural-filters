#!/usr/bin/env python3
"""H4 (exploratory): relative temperature change vs detection flip.

Implements PREREGISTRATION_E2_amendment_2 §7.1 exactly:
  - sample: 600-item subsample, anchors dropped, pairs present in
    detector_pair_scores.csv
  - delta z: per (seed, condition), mean across the three raters of the
    within-rater z difference (condition - original) on the TARGET dimension
    (warmer/colder -> warmth; more_urgent/less_urgent -> urgency). neutral
    excluded.
  - flip: detect_original == 1 and detect_rewrite == 0
  - statistic: point-biserial Pearson r per detector x condition (20 cells),
    n, flip count, r, raw p; BH-FDR within detector across the 4 conditions.
  - cells with flip count < 10 are descriptive only.
  - optional pooled within-condition partial correlation per detector, only
    computed when at least one main-table cell has |r| >= .20.

Output
  e2_analysis/h4_temperature_flip.csv
  e2_analysis/h4_pooled.csv (only if triggered)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "annotation"))
import analyze_calibration as ac  # noqa: E402
from relative_scores import dimension_scores, zscore_within_rater  # noqa: E402

RATINGS = ["e2_ratings/e2_600_R1_filled.csv",
           "e2_ratings/e2_600_R2_filled.csv",
           "e2_ratings/e2_600_R3_filled.csv"]
MAP = "e2_ratings/e2_annotation_600_map.csv"
PAIRS = "e2_analysis/detector_pair_scores.csv"
OUT = "e2_analysis/h4_temperature_flip.csv"
OUT_POOLED = "e2_analysis/h4_pooled.csv"
TARGET_DIM = {"warmer": "warmth", "colder": "warmth",
              "more_urgent": "urgency", "less_urgent": "urgency"}
CONDS = list(TARGET_DIM)


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[ok] = np.clip(adj[np.argsort(order)], 0, 1)
    return q


def main():
    mapping = pd.read_csv(MAP, encoding="utf-8-sig")
    meta = mapping.loc[mapping["is_anchor"] == 0, ["calib_id", "seed_id", "condition"]]

    df = ac.load_ratings([Path(p) for p in RATINGS], Path(MAP))
    df = df.merge(meta, on="calib_id", how="left", validate="many_to_one")
    if df["seed_id"].isna().any():
        raise SystemExit("some rated calib_id has no seed/condition in the map")

    raters = sorted(df["rater_id"].unique())
    scored = dimension_scores(df)
    for dimension in ("warmth", "urgency"):
        scored = zscore_within_rater(scored, dimension)

    # delta z per seed x condition: mean over raters of z(cond) - z(original)
    deltas = {}
    for cond, dimension in TARGET_DIM.items():
        col = f"{dimension}_z"
        pivot = scored.pivot_table(index="seed_id", columns=["condition", "rater_id"],
                                   values=col, aggfunc="mean")
        delta = pd.DataFrame({
            r: pivot[(cond, r)] - pivot[("original", r)] for r in raters
        }).dropna()
        delta = delta.mean(axis=1).rename("delta_z").reset_index()
        delta["condition"] = cond
        delta["dimension"] = dimension
        deltas[cond] = delta
    dz = pd.concat(deltas.values(), ignore_index=True)
    print(f"rated seeds with delta z: {dz.groupby('condition')['seed_id'].nunique().to_dict()}")

    pairs = pd.read_csv(PAIRS, encoding="utf-8-sig")
    pairs["flip"] = ((pairs["detect_original"] == 1) & (pairs["detect_rewrite"] == 0))
    merged = pairs.merge(dz, on=["seed_id", "condition"], how="inner",
                         validate="many_to_one")

    rows = []
    for detector in sorted(merged["detector"].unique()):
        sub = merged[merged["detector"] == detector]
        det_rows = []
        for cond in CONDS:
            cell = sub[sub["condition"] == cond]
            n = len(cell)
            flips = int(cell["flip"].sum())
            r = p = np.nan
            if 0 < flips < n and n >= 3 and cell["delta_z"].std() > 0:
                r, p = stats.pearsonr(cell["delta_z"], cell["flip"].astype(float))
            det_rows.append({
                "detector": detector,
                "condition": cond,
                "dimension": TARGET_DIM[cond],
                "n_pairs": n,
                "flip_count": flips,
                "flip_rate": flips / n if n else np.nan,
                "pearson_r": r,
                "p_value": p,
                "note": "事件过少（flip < 10）" if flips < 10 else "",
            })
        q = bh_fdr([row["p_value"] for row in det_rows])
        for row, qi in zip(det_rows, q):
            row["q_value"] = qi
        rows.extend(det_rows)

    out = pd.DataFrame(rows)[["detector", "condition", "dimension", "n_pairs",
                              "flip_count", "flip_rate", "pearson_r", "p_value",
                              "q_value", "note"]]
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))
    print(f"wrote {OUT}")

    # optional pooled within-condition partial correlation
    triggered = (out["pearson_r"].abs() >= 0.20).any()
    print(f"\npooled sensitivity triggered (|r| >= .20): {triggered}")
    if triggered:
        pooled = []
        for detector in sorted(merged["detector"].unique()):
            sub = merged[merged["detector"] == detector].copy()
            sub["dz_res"] = sub["delta_z"] - sub.groupby("condition")["delta_z"].transform("mean")
            sub["flip_res"] = (sub["flip"].astype(float)
                               - sub.groupby("condition")["flip"].transform("mean"))
            r, p = stats.pearsonr(sub["dz_res"], sub["flip_res"])
            pooled.append({"detector": detector, "n_pairs": len(sub),
                           "flip_count": int(sub["flip"].sum()),
                           "partial_r": r, "p_value": p,
                           "method": "pooled within-condition (4 conditions, fixed effects)"})
        pout = pd.DataFrame(pooled)
        pout.to_csv(OUT_POOLED, index=False, encoding="utf-8-sig")
        print(pout.round(3).to_string(index=False))
        print(f"wrote {OUT_POOLED}")


if __name__ == "__main__":
    main()
