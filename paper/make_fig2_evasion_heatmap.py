#!/usr/bin/env python3
"""Figure 2: evasion-rate heatmap, detector family x condition (G1).

Input  e2_analysis/detector_metrics.csv
Output paper/figures/fig2_evasion_heatmap.{png,pdf}

Defaults reproduce Fig. 2; --metrics/--out/--title switch to G2 (Fig. 5).
"""
import argparse
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
CONDITIONS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
LABELS = ["neutral", "warmer", "colder", "more urgent", "less urgent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "e2_analysis/detector_metrics.csv"))
    ap.add_argument("--out", default="fig2_evasion_heatmap")
    ap.add_argument("--title", default="G1 (DeepSeek): evasion by detector family and condition")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics, encoding="utf-8-sig")
    mat = (df.pivot(index="detector", columns="condition", values="evasion_rate")
             .loc[DETECTORS, CONDITIONS].to_numpy())
    fpr = df.groupby("detector")["test_fpr"].first()

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > 0.30 else "black", fontsize=10)

    ax.set_xticks(np.arange(len(CONDITIONS)), LABELS, fontsize=10)
    ax.set_yticks(
        np.arange(len(DETECTORS)),
        [f"{NAMES[d]}\n(test FPR = {fpr[d]:.3f})" for d in DETECTORS],
        fontsize=9,
    )
    ax.set_xlabel("Rewrite condition", fontsize=10)
    ax.set_ylabel("Detector family", fontsize=10)
    ax.set_xticks(np.arange(-.5, len(CONDITIONS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(DETECTORS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Evasion rate (1 − detection rate)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(args.title, fontsize=11, pad=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{args.out}.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote {OUT / (args.out + '.png')}")
    print(f"wrote {OUT / (args.out + '.pdf')}")


if __name__ == "__main__":
    main()
