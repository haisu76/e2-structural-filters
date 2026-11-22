#!/usr/bin/env python3
"""Figure 1: E2 design and analysis pipeline (G1 numbers; G2 marked pending).

Every count in the figure is taken from e2_analysis/G1_VERDICT.md:
  seeds 200; requests 3,000; validation 2,630/3,000 (87.7%); 965 pairs;
  inclusion 857 pairs / 199 seeds / 115 complete seeds; corpus 6,148 rows
  (1,835 spam / 4,313 ham); split 3,557 / 1,186 / 1,405; test FPR per family.

Output  paper/figures/fig1_design_flow.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FILL = "#eef3f8"
EDGE = "#4a6fa5"
ARROW = "#5a5a5a"
CHIP_COLORS = {"D1": "#1b9e77", "D2": "#d95f02", "D3": "#7570b3",
               "D4": "#e7298a", "D5": "#666666"}
DETECTOR_LABELS = {"D1": "D1 lexical", "D2": "D2 surface", "D3": "D3 structural",
                   "D4": "D4 affect", "D5": "D5 semantic"}
TEST_FPR = {"D1": ".005", "D2": ".008", "D3": ".134", "D4": ".493", "D5": ".005"}


def add_stage(ax, top, height, number, title, lines):
    left, width = 0.95, 8.55
    box = FancyBboxPatch((left, top - height), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=1.1, edgecolor=EDGE, facecolor=FILL, zorder=2)
    ax.add_patch(box)
    ax.add_patch(Circle((0.62, top - 0.30), 0.215, facecolor=EDGE,
                        edgecolor="white", linewidth=1.0, zorder=3))
    ax.text(0.62, top - 0.30, str(number), ha="center", va="center",
            color="white", fontsize=10, fontweight="bold", zorder=4)
    ax.text(1.16, top - 0.24, title, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color="#1f3350", zorder=4)
    ax.text(1.16, top - 0.55, "\n".join(lines), ha="left", va="top",
            fontsize=8.2, color="#222222", linespacing=1.42, zorder=4)


def arrow(ax, y_from, y_to):
    ax.annotate("", xy=(5.2, y_to), xytext=(5.2, y_from),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, linewidth=1.5,
                                mutation_scale=14), zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(8.4, 9.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    stages = [
        (1.00, "Seed pool",
         ["200 English spam seeds from the SpamAssassin corpus, with protected",
          "spans (links, amounts, contact details) that every rewrite must keep."]),
        (1.20, "Controlled rewriting",
         ["5 conditions (neutral, warmer, colder, more urgent, less urgent) ×",
          "3 candidates = 3,000 requests per generator.  G1: DeepSeek-V3;",
          "G2: Qwen2.5-7B-Instruct (generation in progress)."]),
        (1.00, "Hard validation",
         ["2,630 / 3,000 rewrites pass the rule-based validator (87.7%; no parse",
          "failures); 965 valid candidate pairs enter inclusion sampling."]),
        (1.45, "Inclusion coding",
         ["Four double-coding rounds fail the κ ≥ .60 bar (−.021, .503, .078, .177).",
          "PI adjudication + independent audits: 65-pair check 16.9% invalid",
          "[9.7–27.8]; 211-pair screen review 24.2% [18.9–30.4].",
          "Final main set: 857 pairs / 199 seeds / 115 complete seeds."]),
        (1.78, "Detector families and operating point",
         ["Corpus 6,148 rows (1,835 spam / 4,313 ham); grouped 60/20/20 split →",
          "3,557 train / 1,186 validation / 1,405 test.  One threshold per detector:",
          "10th percentile of validation spam scores (TPR ≈ .90)."]),
        (1.20, "Outcomes (H1–H4)",
         ["Absolute evasion by condition; paired detection change on the same",
          "seeds (10,000 clustered bootstrap); decision flips; standardized",
          "score drift.  BH-FDR within each outcome family."]),
    ]

    top = 11.60
    tops = []
    for i, (height, title, lines) in enumerate(stages, start=1):
        tops.append(top)
        add_stage(ax, top, height, i, title, lines)
        top = top - height - 0.42

    for i in range(len(stages) - 1):
        arrow(ax, tops[i] - stages[i][0], tops[i + 1])

    # detector chips inside stage 5
    stage5_top = tops[4]
    stage5_bottom = stage5_top - stages[4][0]
    chip_y = stage5_bottom + 0.16
    chip_w, chip_h, gap = 1.52, 0.36, 0.10
    x = 1.16
    for detector in ["D1", "D2", "D3", "D4", "D5"]:
        ax.add_patch(FancyBboxPatch(
            (x, chip_y), chip_w, chip_h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=CHIP_COLORS[detector], edgecolor="white", linewidth=0.8,
            alpha=0.92, zorder=4))
        ax.text(x + chip_w / 2, chip_y + chip_h / 2,
                f"{DETECTOR_LABELS[detector]}\nFPR {TEST_FPR[detector]}",
                ha="center", va="center", fontsize=7.2, color="white",
                linespacing=1.25, zorder=5)
        x += chip_w + gap

    ax.text(5.2, 1.15,
            "Analyses are run separately per generator; no pooling.  "
            "G1 results are reported in §4; G2 follows the frozen procedure in §5.",
            ha="center", va="center", fontsize=8.2, color="#555555",
            style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_design_flow.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote {OUT / 'fig1_design_flow.png'}")
    print(f"wrote {OUT / 'fig1_design_flow.pdf'}")


if __name__ == "__main__":
    main()
