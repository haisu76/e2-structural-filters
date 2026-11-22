#!/usr/bin/env python3
"""Figure 4: exploratory H4 scatter, D3 under less urgent rewrites.

Each point is one seed with an included pair (n = 100). x is the mean across
the three raters of the within-rater z difference in urgency (rewrite minus
original); y is whether the D3 decision flipped from spam to ham (detected ->
undetected). The 34 flip points are highlighted; a logistic fit and binned
flip rates show the negative association (r = -.515, q < .001).

Inputs
  e2_ratings/e2_600_R{1,2,3}_filled.csv
  e2_ratings/e2_annotation_600_map.csv
  e2_analysis/detector_pair_scores.csv
Output
  paper/figures/fig4_h4_scatter.{png,pdf}
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "annotation"))
import analyze_calibration as ac  # noqa: E402
from relative_scores import dimension_scores, zscore_within_rater  # noqa: E402

DETECTOR = "D3"
CONDITION = "less_urgent"
DIMENSION = "urgency"
COLOR_FLIP = "#c0392b"
COLOR_NONE = "#9aa5b1"


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def load_delta_z():
    ratings = [ROOT / f"e2_ratings/e2_600_R{i}_filled.csv" for i in (1, 2, 3)]
    mapping_path = ROOT / "e2_ratings/e2_annotation_600_map.csv"
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    meta = mapping.loc[mapping["is_anchor"] == 0, ["calib_id", "seed_id", "condition"]]

    df = ac.load_ratings(ratings, mapping_path)
    df = df.merge(meta, on="calib_id", how="left", validate="many_to_one")
    if df["seed_id"].isna().any():
        raise SystemExit("some rated calib_id has no seed/condition in the map")

    raters = sorted(df["rater_id"].unique())
    scored = dimension_scores(df)
    scored = zscore_within_rater(scored, DIMENSION)
    pivot = scored.pivot_table(index="seed_id", columns=["condition", "rater_id"],
                               values=f"{DIMENSION}_z", aggfunc="mean")
    delta = pd.DataFrame({
        r: pivot[(CONDITION, r)] - pivot[("original", r)] for r in raters
    }).dropna()
    return delta.mean(axis=1).rename("delta_z").reset_index()


def main():
    dz = load_delta_z()
    pairs = pd.read_csv(ROOT / "e2_analysis/detector_pair_scores.csv",
                        encoding="utf-8-sig")
    sub = pairs[(pairs["detector"] == DETECTOR)
                & (pairs["condition"] == CONDITION)].copy()
    sub["flip"] = ((sub["detect_original"] == 1) & (sub["detect_rewrite"] == 0)).astype(int)
    data = sub.merge(dz, on="seed_id", how="inner", validate="many_to_one")
    n = len(data)
    flips = int(data["flip"].sum())
    r, p = stats.pearsonr(data["delta_z"], data["flip"])
    print(f"n={n} flips={flips} r={r:.3f} p={p:.2e}")

    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.045, 0.045, n)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    none = data["flip"] == 0
    ax.scatter(data.loc[none, "delta_z"], jitter[none.to_numpy()],
               s=34, facecolor="none", edgecolor=COLOR_NONE, linewidth=0.8,
               label=f"no flip (n = {n - flips})", zorder=3)
    ax.scatter(data.loc[~none, "delta_z"], 1 + jitter[~none.to_numpy()],
               s=42, facecolor=COLOR_FLIP, edgecolor="white", linewidth=0.6,
               label=f"flip: detected → undetected (n = {flips})", zorder=4)

    # logistic fit
    x = data["delta_z"].to_numpy().reshape(-1, 1)
    y = data["flip"].to_numpy()
    model = LogisticRegression().fit(x, y)
    grid = np.linspace(data["delta_z"].min() - 0.05, data["delta_z"].max() + 0.05, 200)
    prob = model.predict_proba(grid.reshape(-1, 1))[:, 1]
    ax.plot(grid, prob, color="#2c3e50", linewidth=1.6, zorder=5,
            label="logistic fit")

    # binned flip rates (quintiles)
    data["bin"] = pd.qcut(data["delta_z"], 5, labels=False)
    bx, by, lo, hi = [], [], [], []
    for b in range(5):
        cell = data[data["bin"] == b]
        bx.append(cell["delta_z"].mean())
        rate = cell["flip"].mean()
        by.append(rate)
        l, h = wilson(int(cell["flip"].sum()), len(cell))
        lo.append(max(l, 0))
        hi.append(min(h, 1))
    ax.errorbar(bx, by, yerr=[np.array(by) - np.array(lo), np.array(hi) - np.array(by)],
                fmt="s", color="#2c3e50", markersize=5, linewidth=1.1,
                capsize=3, zorder=6, label="quintile flip rate (95% CI)")

    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.axhline(1, color="black", linewidth=0.6, zorder=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_ylim(-0.22, 1.55)
    ax.set_yticks([0, 1], ["no flip", "flip"])
    ax.set_xlabel("Urgency change, rewrite − original (within-rater z, SD units)",
                  fontsize=10)
    ax.set_ylabel("D3 decision", fontsize=10)
    ax.set_title("H4 (exploratory): urgency drop and D3 flips under less urgent rewrites",
                 fontsize=10.5, pad=8)
    ax.text(0.02, 0.97, f"n = {n}, flips = {flips}\nr = {r:.3f}, q < .001",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6))
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_h4_scatter.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote {OUT / 'fig4_h4_scatter.png'}")
    print(f"wrote {OUT / 'fig4_h4_scatter.pdf'}")


if __name__ == "__main__":
    main()
