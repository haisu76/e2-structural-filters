#!/usr/bin/env python3
"""Route A: per-rater relative (z-score) reliability on the v1.3 400 batch.

Implements the analysis frozen in `annotation/PREREGISTRATION_route_A.md`:

  - the primary absolute-agreement verdict stays REVISE; this script does not
    produce a new PASS/FAIL verdict;
  - each rater x dimension is z-scored across the 400 unique items, then
    ICC(2,3) is computed on the z-scores. With rater means at 0 the rater
    (column) effect is removed, so this is a rank/consistency ICC;
  - `relative_mean_z` (mean of the three rater z-scores) is written per item
    for within-rater relative drift analyses only;
  - distance is exploratory (v1.3 consistency ICC .698 < .70), and any
    dimension whose z ICC is below .70 is not used for drift conclusions.

Usage:
  python annotation/relative_scores.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_calibration as ac
from diagnose_400 import consistency_icc
from rating_schema import DIMENSION_ITEMS, SCHEMA_VERSION

DEFAULT_RATINGS = [
    "annotation/e1_400_v1.3_R1.csv",
    "annotation/e1_400_v1.3_R2.csv",
    "annotation/e1_400_v1.3_R3.csv",
]
DEFAULT_MAP = "annotation/e1_annotation_400_v1.3_qc_map.csv"


def dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for dimension, items in DIMENSION_ITEMS.items():
        scored = [f"{item}__scored" for item in items]
        out[f"{dimension}_score"] = out[scored].mean(axis=1)
    return out


def wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (df.pivot(index="calib_id", columns="rater_id", values=column)
            .dropna())


def zscore_within_rater(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """z_{r,d}(i) = (score - mean_i) / sd_i within each rater x dimension."""
    out = df.copy()
    for rater, sub in out.groupby("rater_id"):
        values = sub[f"{dimension}_score"]
        sd = values.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            raise SystemExit(f"{rater}/{dimension}: zero or undefined SD")
        out.loc[sub.index, f"{dimension}_z"] = (values - values.mean()) / sd
    return out


def reliability_table(df: pd.DataFrame, n_boot: int, seed: int,
                      min_relative_icc: float) -> pd.DataFrame:
    rows = []
    for index, dimension in enumerate(DIMENSION_ITEMS):
        absolute = wide(df, f"{dimension}_score").to_numpy(dtype=float)
        zwide = wide(df, f"{dimension}_z").to_numpy(dtype=float)

        icc_abs = ac.icc_2k(absolute)
        abs_low, abs_high, _ = ac.bootstrap_icc(absolute, n_boot, seed + 100 + index)
        alpha = ac.krippendorff_alpha(absolute, level="interval")

        icc_z = ac.icc_2k(zwide)
        z_low, z_high, _ = ac.bootstrap_icc(zwide, n_boot, seed + 200 + index)
        icc_consistency = consistency_icc(absolute)

        if dimension == "distance":
            drift_use = "exploratory"
        elif np.isfinite(icc_z) and icc_z < min_relative_icc:
            drift_use = "not usable (<.70)"
        else:
            drift_use = "primary"

        rows.append({
            "dimension": dimension,
            "n_items": len(DIMENSION_ITEMS[dimension]),
            "n_cases": len(absolute),
            "mean_score": float(absolute.mean()),
            "icc_absolute": icc_abs,
            "abs_ci_low": abs_low,
            "abs_ci_high": abs_high,
            "alpha_interval": alpha,
            "icc_z": icc_z,
            "z_ci_low": z_low,
            "z_ci_high": z_high,
            "icc_consistency": icc_consistency,
            "drift_use": drift_use,
        })
    return pd.DataFrame(rows)


def relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dimension in DIMENSION_ITEMS:
        scores = wide(df, f"{dimension}_score")
        scores.columns = [f"{dimension}_score_{rater}" for rater in scores.columns]
        zs = wide(df, f"{dimension}_z")
        zs.columns = [f"{dimension}_z_{rater}" for rater in zs.columns]
        merged = scores.join(zs)
        merged[f"{dimension}_relative_mean_z"] = zs.mean(axis=1)
        frames.append(merged)
    result = pd.concat(frames, axis=1).reset_index()
    block = df.drop_duplicates("calib_id").set_index("calib_id")["block"]
    result.insert(1, "block", result["calib_id"].map(block))
    return result


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    data = df[columns]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in data.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f"{float(value):.3f}" if isinstance(value, (float, np.floating))
                         else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path: Path, scored: pd.DataFrame, scores: pd.DataFrame,
                 table: pd.DataFrame, args: argparse.Namespace) -> None:
    rater_use = (scored.groupby("rater_id")
                 .agg(**{f"{d}_mean": (f"{d}_score", "mean") for d in DIMENSION_ITEMS})
                 .reset_index())
    distance = table.set_index("dimension").loc["distance"]
    lines = [
        "# E1 400 条 v1.3 路线 A：相对分数信度报告",
        "",
        "- 协议：`annotation/PREREGISTRATION_route_A.md`（2026-11-10，在看到任何 E2/检测器结果之前冻结）",
        "- 数据：v1.3 400 条唯一邮件，三人评分（锚题已按 QC map 剔除）",
        "- 主判定：**绝对一致性 REVISE**（见 `calibration_findings_v2.1.md` 与 `e1_400_v1.3_findings.md`）",
        "",
        "> 本报告不改变主判定，也不构成新的 PASS/FAIL。它只回答：同一评分员内部，",
        "> 邮件之间的相对温度差异是否可可靠排序。不能用于跨评分员的绝对温度比较。",
        "",
        "## 1. 数据与计分",
        "",
        "- schema v1.3（`rating_schema.py` 默认）：valence = mean(8−V1, V2)；"
        "urgency = mean(A1, A2, A3)；warmth = mean(W1, W2, W3)；distance = mean(D1, D2, D3)。",
        f"- 评分行数：{args.n_rows}；邮件数：{len(scores)}；评分员数：3；"
        f"bootstrap 次数：{args.bootstrap}；随机种子：{args.seed}。",
        "",
        "## 2. 绝对一致性（主判定，不变）",
        "",
        md_table(table, ["dimension", "n_cases", "mean_score", "icc_absolute",
                         "abs_ci_low", "abs_ci_high", "alpha_interval"]),
        "",
        "四个维度均未达到预注册的绝对一致性门槛（ICC ≥ .75、CI 下限 ≥ .70、interval α ≥ .67），"
        "总体判定维持 **REVISE**。",
        "",
        "## 3. 相对信度（评分员 × 维度 z 分）",
        "",
        md_table(table, ["dimension", "icc_z", "z_ci_low", "z_ci_high",
                         "icc_consistency", "drift_use"]),
        "",
        "- `icc_z`：每名评分员在每个维度内对 400 条邮件做 z 分后计算的 ICC(2,3)。"
        "z 分的评分员均值为 0、方差为 1，列效应被移除，因此该值等价于排序/一致性 ICC，"
        "而不是新的绝对一致性证据。",
        "- `icc_consistency`：原始分数上的 ICC(3,3) 一致性 ICC，仅作交叉核对"
        "（与 `e1_400_results_v1.3/diagnostics/dimension_agreement.csv` 一致）。",
        "- `drift_use`：distance 一律标为探索性；相对 ICC 低于 .70 的维度不用于漂移结论。",
        "",
        "## 4. 评分员量表使用",
        "",
        md_table(rater_use, list(rater_use.columns)),
        "",
        "z 分移除了这些评分员水平差，因此相对分析不比较评分员之间的绝对分数。",
        "",
        "## 5. distance 的探索性标记",
        "",
        "v1.3 400 条的 distance 排序一致性 ICC 低于 .70，且 D1 的 R1–R2 条目级排序一致性很低。"
        "本报告中 distance 的结果标为**探索性**，不用于检测器漂移的主要结论；"
        "若报告，必须同时报告绝对与相对信度。",
        "",
        f"- 本批 distance 的 z 分 ICC = {distance['icc_z']:.3f}"
        f"（95% CI [{distance['z_ci_low']:.3f}, {distance['z_ci_high']:.3f}]），"
        "CI 下限低于 .70，因此仍只作探索性报告。",
        "",
        "## 6. `relative_mean_z` 的允许用途",
        "",
        "`relative_scores.csv` 中每条邮件的各维度 `relative_mean_z` = 三名评分员 z 分的均值。"
        "允许：同一评分员内比较不同语料/改写条件/时间批次的相对温度变化，"
        "以及用 `relative_mean_z` 做探索性漂移描述。",
        "",
        "不允许：把相对分析写成 PASS 或“信度已解决”；跨评分员比较绝对温度分数；"
        "在信度未解决时做 EFA/CFA、测量不变性或因果性检测器结论；"
        "在看到检测器结果后修改本协议的分析或门槛。",
        "",
        "## 7. 复现",
        "",
        "```bash",
        "python annotation/relative_scores.py \\",
        "  --ratings annotation/e1_400_v1.3_R1.csv annotation/e1_400_v1.3_R2.csv \\",
        "            annotation/e1_400_v1.3_R3.csv \\",
        "  --map annotation/e1_annotation_400_v1.3_qc_map.csv \\",
        f"  --out {str(args.out).replace(chr(92), '/')}"
        f" --bootstrap {args.bootstrap} --seed {args.seed}",
        "```",
        "",
        "## 8. 输出",
        "",
        "- `relative_reliability.csv`：各维度绝对 ICC、z 分 ICC、一致性与 α；",
        "- `relative_scores.csv`：每条邮件的维度原始分、z 分与 `relative_mean_z`；",
        "- `relative_report.md`：本报告。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route A relative-score analysis (v1.3 400 batch).")
    parser.add_argument("--ratings", nargs="+", default=DEFAULT_RATINGS)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--out", default="annotation/e1_400_route_A")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-relative-icc", type=float, default=0.70)
    args = parser.parse_args()

    if SCHEMA_VERSION != "1.3":
        raise SystemExit(f"route A requires the frozen v1.3 schema, got {SCHEMA_VERSION}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ac.load_ratings([Path(p) for p in args.ratings], Path(args.map))
    raters = sorted(df["rater_id"].unique())
    counts = df.groupby("calib_id")["rater_id"].nunique()
    df = df[df["calib_id"].isin(counts[counts == len(raters)].index)].copy()

    scored = dimension_scores(df)
    for dimension in DIMENSION_ITEMS:
        scored = zscore_within_rater(scored, dimension)

    table = reliability_table(scored, args.bootstrap, args.seed,
                              args.min_relative_icc)
    scores = relative_scores(scored)

    table.to_csv(out_dir / "relative_reliability.csv", index=False,
                 encoding="utf-8-sig")
    scores.to_csv(out_dir / "relative_scores.csv", index=False,
                  encoding="utf-8-sig")

    args.n_rows = len(df)
    args.out = out_dir
    write_report(out_dir / "relative_report.md", scored, scores, table, args)

    print(f"wrote {out_dir}/relative_reliability.csv")
    print(f"wrote {out_dir}/relative_scores.csv")
    print(f"wrote {out_dir}/relative_report.md")
    print(table[["dimension", "icc_absolute", "icc_z", "z_ci_low",
                 "z_ci_high", "icc_consistency", "drift_use"]].to_string(index=False))


if __name__ == "__main__":
    main()
