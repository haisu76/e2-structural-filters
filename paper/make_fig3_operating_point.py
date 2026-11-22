#!/usr/bin/env python3
"""Figure 3: paired detection loss vs standardized score drift (G1).

Every point is one detector x condition cell. The figure shows that the family
with the lowest absolute evasion rate (D3) has the largest paired detection
loss and score drift under colder / less_urgent rewrites.

Input  e2_analysis/detector_metrics.csv
Output paper/figures/fig3_operating_point.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DETECTORS = ["D1", "D2", "D3", "D4", "D5"]
NAMES = {"D1": "D1 lexical", "D2": "D2 surface style", "D3": "D3 structural",
         "D4": "D4 affect", "D5": "D5 semantic"}
COLORS = {"D1": "#1b9e77", "D2": "#d95f02", "D3": "#7570b3",
          "D4": "#e7298a", "D5": "#666666"}
CONDITIONS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
MARKERS = {"neutral": "o", "warmer": "^", "colder": "v",
           "more_urgent": "s", "less_urgent": "D"}
LABELS = {"neutral": "neutral", "warmer": "warmer", "colder": "colder",
          "more_urgent": "more urgent", "less_urgent": "less urgent"}


def main():
    df = pd.read_csv(ROOT / "e2_analysis/detector_metrics.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for detector in DETECTORS:
        for condition in CONDITIONS:
            row = df[(df["detector"] == detector) & (df["condition"] == condition)]
            if row.empty:
                continue
            x = row["detection_delta"].iloc[0] * 100
            y = row["score_drift_z"].iloc[0]
            ax.scatter(x, y, s=55, color=COLORS[detector], marker=MARKERS[condition],
                       edgecolor="white", linewidth=0.6, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.axvline(0, color="black", linewidth=0.8, zorder=1)

    # label the two cells that carry the argument
    ax.annotate(
        "D3 colder: −30.5 pp, z = −1.18\nD3 less urgent: −33.8 pp, z = −1.16",
        xy=(-32.0, -1.17), xytext=(-25.5, -0.72),
        fontsize=9, color=COLORS["D3"], ha="left",
        arrowprops=dict(arrowstyle="->", color=COLORS["D3"], linewidth=0.9),
    )
    ax.annotate(
        "",
        xy=(-33.8, -1.162), xytext=(-25.0, -0.86),
        arrowprops=dict(arrowstyle="->", color=COLORS["D3"], linewidth=0.9,
                        connectionstyle="arc3,rad=-0.15"),
    )

    ax.set_xlabel("Paired detection change after rewriting (percentage points)", fontsize=10)
    ax.set_ylabel("Standardized score drift (rewrite − original, SD units)", fontsize=10)
    ax.set_title("G1: structural features lose the most detection and drift the most",
                 fontsize=11, pad=8)

    detector_handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[d],
                                   markersize=7, label=NAMES[d]) for d in DETECTORS]
    condition_handles = [plt.Line2D([], [], marker=MARKERS[c], linestyle="",
                                    color="black", markerfacecolor="none",
                                    markersize=7, label=LABELS[c]) for c in CONDITIONS]
    first = ax.legend(handles=detector_handles, title="Detector family",
                      loc="upper left", fontsize=8, title_fontsize=8, frameon=False)
    ax.add_artist(first)
    ax.legend(handles=condition_handles, title="Condition", loc="lower right",
              fontsize=8, title_fontsize=8, frameon=False)

    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_operating_point.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote {OUT / 'fig3_operating_point.png'}")
    print(f"wrote {OUT / 'fig3_operating_point.pdf'}")


if __name__ == "__main__":
    main()
