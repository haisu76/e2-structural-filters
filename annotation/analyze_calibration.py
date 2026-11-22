#!/usr/bin/env python3
"""Analyze the 50-item E1 calibration ratings.

Primary output:
  - dimension-level ICC(2,k) with bootstrap 95% CI (absolute agreement)
  - item-level ICC(2,k), Krippendorff ordinal alpha, corrected item-total r
  - rater severity summary and item-level disagreement list

The script also has a --simulate mode that writes plausible ratings and runs the
full pipeline. Simulated results are for pipeline testing only.

Examples:
  python annotation/analyze_calibration.py --self-test
  python annotation/analyze_calibration.py --simulate --out annotation/simulated_results
  python annotation/analyze_calibration.py \
      --ratings annotation/calibration_R1.csv annotation/calibration_R2.csv \
                annotation/calibration_R3.csv \
      --out annotation/calibration_results
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rating_schema import (
    DIMENSION_ITEMS,
    ITEMS,
    ITEM_DIMENSION,
    ITEM_ROLE,
    REVERSE_ITEMS,
    SCALE_MAX,
    SCALE_MIN,
    TEMPLATE_COLUMNS,
    THREAT_COLUMNS,
)


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fisher_mean(rs: list[float]) -> float:
    vals = [float(r) for r in rs if np.isfinite(r) and -1 < r < 1]
    if not vals:
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(np.clip(vals, -0.999999, 0.999999)))))


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    data = df.head(max_rows) if max_rows else df
    cols = [str(c) for c in data.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in data.iterrows():
        cells = []
        for col in data.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                text = f"{float(value):.3f}"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# reliability estimators
# ---------------------------------------------------------------------------
def icc_2k(matrix: np.ndarray) -> float:
    """ICC(2,k): two-way random effects, average raters, absolute agreement."""
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ssr = k * np.sum((row_means - grand) ** 2)
    ssc = n * np.sum((col_means - grand) ** 2)
    sse = np.sum((matrix - row_means[:, None] - col_means[None, :] + grand) ** 2)
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denominator = msr + (msc - mse) / n
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((msr - mse) / denominator)


def bootstrap_icc(matrix: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, int]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 3:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = matrix.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        estimate = icc_2k(matrix[idx])
        if np.isfinite(estimate):
            values.append(estimate)
    if len(values) < max(100, n_boot // 10):
        return float("nan"), float("nan"), len(values)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), len(values)


def krippendorff_alpha(matrix: np.ndarray, level: str = "ordinal") -> float:
    """Krippendorff's alpha for complete units x raters.

    level='ordinal' uses the ordinal difference function (item ratings);
    level='interval' uses squared numeric differences (dimension scores).
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return float("nan")
    values = sorted(set(float(v) for v in matrix.flatten()))
    if len(values) < 2:
        return 1.0  # all ratings identical
    index = {value: i for i, value in enumerate(values)}
    size = len(values)
    observed = np.zeros((size, size), dtype=float)
    m = matrix.shape[1]

    for row in matrix:
        ratings = [float(v) for v in row]
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    observed[index[ratings[i]], index[ratings[j]]] += 1.0 / (m - 1)

    total = observed.sum()
    if total <= 1:
        return float("nan")
    marginals = observed.sum(axis=1)
    expected = np.outer(marginals, marginals) / (total - 1)
    for i in range(size):
        expected[i, i] = marginals[i] * (marginals[i] - 1) / (total - 1)

    delta = np.zeros((size, size), dtype=float)
    for i, value_i in enumerate(values):
        for j, value_j in enumerate(values):
            if level == "interval":
                delta[i, j] = (value_i - value_j) ** 2
            else:
                lo, hi = (i, j) if i <= j else (j, i)
                difference = marginals[lo:hi + 1].sum() - (marginals[lo] + marginals[hi]) / 2.0
                delta[i, j] = difference ** 2

    observed_disagreement = float((observed * delta).sum())
    expected_disagreement = float((expected * delta).sum())
    if expected_disagreement == 0:
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


# ---------------------------------------------------------------------------
# loading / scoring
# ---------------------------------------------------------------------------
def load_ratings(paths: list[Path], map_path: Path | None = None) -> pd.DataFrame:
    mapping = None
    if map_path is not None:
        mapping = pd.read_csv(map_path, encoding="utf-8-sig")
        missing = [c for c in ("sheet_id", "is_anchor") if c not in mapping.columns]
        if missing:
            raise SystemExit(f"QC map is missing columns: {missing}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.columns = [str(c).strip() for c in frame.columns]
        if mapping is not None:
            if "sheet_id" not in frame.columns:
                raise SystemExit(f"{path}: --map requires a sheet_id column")
            frame = frame.merge(mapping[["sheet_id", "is_anchor"]],
                                on="sheet_id", how="left")
            frame = frame[frame["is_anchor"].fillna(0).astype(int) == 0]
            frame = frame.drop(columns=["is_anchor"])
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return validate_ratings(df)


def validate_ratings(df: pd.DataFrame) -> pd.DataFrame:
    required = ["rater_id", "calib_id"] + ITEMS + THREAT_COLUMNS + ["confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    df = df.copy()
    df["rater_id"] = df["rater_id"].astype(str).str.strip()
    df["calib_id"] = df["calib_id"].astype(str).str.strip()
    if df["rater_id"].eq("").any() or df["rater_id"].eq("nan").any():
        raise SystemExit("blank rater_id found")
    if df["calib_id"].eq("").any() or df["calib_id"].eq("nan").any():
        raise SystemExit("blank calib_id found")
    if df.duplicated(["rater_id", "calib_id"]).any():
        duplicates = df[df.duplicated(["rater_id", "calib_id"], keep=False)]
        raise SystemExit(f"duplicate rater_id/calib_id rows: {duplicates.head().to_dict('records')}")

    for item in ITEMS:
        values = pd.to_numeric(df[item], errors="coerce")
        bad = values.isna() | (values < SCALE_MIN) | (values > SCALE_MAX) | (values % 1 != 0)
        if bad.any():
            examples = df.loc[bad, ["rater_id", "calib_id", item]].head(5).to_dict("records")
            raise SystemExit(f"{item}: invalid ratings (must be integers {SCALE_MIN}-{SCALE_MAX}): {examples}")
        df[item] = values.astype(int)
        df[f"{item}__scored"] = np.where(
            item in REVERSE_ITEMS, (SCALE_MIN + SCALE_MAX) - df[item], df[item]
        )

    binary = pd.to_numeric(df["threat_binary"], errors="coerce")
    if binary.isna().any() or not binary.isin([0, 1]).all():
        raise SystemExit("threat_binary must be 0/1 for every row")
    df["threat_binary"] = binary.astype(int)

    severity = pd.to_numeric(df["threat_severity"], errors="coerce")
    missing_severity = severity.isna()
    if (missing_severity & (df["threat_binary"] == 1)).any():
        raise SystemExit("threat_severity is missing while threat_binary=1")
    severity = severity.fillna(0)
    if not severity.isin([0, 1, 2, 3]).all():
        raise SystemExit("threat_severity must be 0-3")
    df["threat_severity"] = severity.astype(int)
    inconsistent = (df["threat_binary"] == 0) & (df["threat_severity"] > 0)
    if inconsistent.any():
        print(f"warning: {int(inconsistent.sum())} rows have threat_binary=0 but severity>0; "
              "severity set to 0")
        df.loc[inconsistent, "threat_severity"] = 0

    confidence = pd.to_numeric(df["confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.isin([1, 2, 3]).all():
        raise SystemExit("confidence must be 1/2/3 for every row")
    df["confidence"] = confidence.astype(int)

    return df


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def corrected_item_total(df: pd.DataFrame, item: str) -> float:
    dimension = ITEM_DIMENSION[item]
    others = [i for i in DIMENSION_ITEMS[dimension] if i != item]
    rs = []
    for _, sub in df.groupby("rater_id"):
        x = sub[f"{item}__scored"].to_numpy()
        y = sub[[f"{i}__scored" for i in others]].mean(axis=1).to_numpy()
        r = pearson(x, y)
        if np.isfinite(r):
            rs.append(r)
    return fisher_mean(rs)


def mean_inter_item_r(df: pd.DataFrame, dimension: str) -> float:
    items = DIMENSION_ITEMS[dimension]
    rs = []
    for _, sub in df.groupby("rater_id"):
        corr = sub[[f"{i}__scored" for i in items]].corr()
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                r = corr.iloc[a, b]
                if np.isfinite(r):
                    rs.append(float(r))
    return fisher_mean(rs)


def item_reliability(df: pd.DataFrame, n_boot: int, seed: int,
                     min_item_icc: float = 0.50,
                     min_item_total: float = 0.30,
                     min_distinct: int = 0) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(ITEMS):
        role = ITEM_ROLE[item]
        wide = df.pivot(index="calib_id", columns="rater_id",
                        values=f"{item}__scored").dropna()
        matrix = wide.to_numpy(dtype=float)
        icc = icc_2k(matrix)
        low, high, n_boot_valid = bootstrap_icc(matrix, n_boot, seed + index)
        alpha = krippendorff_alpha(matrix, level="ordinal")
        ranges = matrix.max(axis=1) - matrix.min(axis=1) if len(matrix) else np.array([])
        item_total = corrected_item_total(df, item)
        distinct_by_rater = df.groupby("rater_id")[f"{item}__scored"].nunique()
        min_distinct_rater = int(distinct_by_rater.min()) if len(distinct_by_rater) else 0

        flags: list[str] = []
        warnings: list[str] = []
        if role == "scored":
            if np.isfinite(icc) and icc < min_item_icc:
                flags.append(f"icc<{min_item_icc:.2f}")
            if np.isfinite(item_total) and item_total < min_item_total:
                flags.append(f"item_total<{min_item_total:.2f}")
            if min_distinct > 0 and min_distinct_rater < min_distinct:
                flags.append(f"distinct<{min_distinct}")
        if np.isfinite(alpha) and alpha < 0.40:
            warnings.append("alpha<.40")
        if role == "diagnostic":
            warnings.append("diagnostic; excluded from dimension score and PASS gate")

        rows.append({
            "item": item,
            "dimension": ITEM_DIMENSION[item],
            "role": role,
            "n_cases": len(matrix),
            "mean": float(matrix.mean()) if len(matrix) else float("nan"),
            "sd": float(matrix.std(ddof=1)) if len(matrix) > 1 else float("nan"),
            "mean_range": float(ranges.mean()) if len(ranges) else float("nan"),
            "icc": icc,
            "ci_low": low,
            "ci_high": high,
            "alpha": alpha,
            "item_total_r": item_total,
            "min_distinct": min_distinct_rater,
            "boot_valid": n_boot_valid,
            "warning": ";".join(warnings),
            "flag": ";".join(flags),
        })
    return pd.DataFrame(rows)


def dimension_verdict(icc: float, ci_low: float, alpha: float, target: float,
                      min_icc: float, min_alpha: float) -> str:
    if not np.isfinite(icc):
        return "FAIL"
    if icc < 0.60:
        return "FAIL"
    if (icc >= target and np.isfinite(ci_low) and ci_low >= min_icc
            and np.isfinite(alpha) and alpha >= min_alpha):
        return "PASS"
    return "REVISE"


def dimension_reliability(df: pd.DataFrame, n_boot: int, seed: int,
                          target: float, min_icc: float, min_alpha: float) -> pd.DataFrame:
    rows = []
    for index, (dimension, items) in enumerate(DIMENSION_ITEMS.items()):
        scored = [f"{i}__scored" for i in items]
        scale = (df.groupby(["rater_id", "calib_id"])[scored]
                 .mean().mean(axis=1).rename("score").reset_index())
        wide = scale.pivot(index="calib_id", columns="rater_id",
                           values="score").dropna()
        matrix = wide.to_numpy(dtype=float)
        icc = icc_2k(matrix)
        low, high, n_boot_valid = bootstrap_icc(matrix, n_boot, seed + 100 + index)
        alpha = krippendorff_alpha(matrix, level="interval")
        rows.append({
            "dimension": dimension,
            "label": dimension,
            "n_items": len(items),
            "n_cases": len(matrix),
            "mean_score": float(matrix.mean()) if len(matrix) else float("nan"),
            "mean_inter_item_r": mean_inter_item_r(df, dimension),
            "icc": icc,
            "ci_low": low,
            "ci_high": high,
            "alpha": alpha,
            "boot_valid": n_boot_valid,
            "verdict": dimension_verdict(icc, low, alpha, target, min_icc, min_alpha),
        })
    return pd.DataFrame(rows)


def rater_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rater, sub in df.groupby("rater_id"):
        dimension_scores = []
        for dimension, items in DIMENSION_ITEMS.items():
            score = sub[[f"{i}__scored" for i in items]].mean(axis=1)
            dimension_scores.append(score)
            rows.append({
                "rater_id": rater,
                "dimension": dimension,
                "n": int(score.shape[0]),
                "mean": float(score.mean()),
                "sd": float(score.std(ddof=1)) if len(score) > 1 else float("nan"),
                "min": float(score.min()),
                "max": float(score.max()),
            })
        overall = pd.concat(dimension_scores, axis=1).mean(axis=1)
        rows.append({
            "rater_id": rater,
            "dimension": "overall",
            "n": int(overall.shape[0]),
            "mean": float(overall.mean()),
            "sd": float(overall.std(ddof=1)) if len(overall) > 1 else float("nan"),
            "min": float(overall.min()),
            "max": float(overall.max()),
        })
    return pd.DataFrame(rows)


def disagreement_table(df: pd.DataFrame) -> pd.DataFrame:
    subject_map = df.drop_duplicates("calib_id").set_index("calib_id").get("subject", pd.Series(dtype=str))
    rows = []
    for item in ITEMS:
        wide = df.pivot(index="calib_id", columns="rater_id", values=f"{item}__scored")
        for calib_id, row in wide.iterrows():
            values = row.dropna()
            if len(values) < 2:
                continue
            value_range = float(values.max() - values.min())
            value_sd = float(values.std(ddof=1))
            if value_range >= 3 or value_sd >= 1.5:
                ratings = "; ".join(f"{rater}={int(value)}" for rater, value in values.items())
                subject = subject_map.get(calib_id, "")
                if pd.isna(subject):
                    subject = ""
                rows.append({
                    "calib_id": calib_id,
                    "item": item,
                    "dimension": ITEM_DIMENSION[item],
                    "ratings": ratings,
                    "range": value_range,
                    "sd": value_sd,
                    "subject": str(subject)[:120],
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["range", "sd"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def simulate_ratings(template_path: Path, meta_path: Path, raters: int,
                     seed: int) -> pd.DataFrame:
    template = pd.read_csv(template_path, encoding="utf-8-sig")
    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    required_meta = ["calib_id", "label", "urgency_proxy", "warmth_proxy"]
    missing = [c for c in required_meta if c not in meta.columns]
    if missing:
        raise SystemExit(f"missing meta columns: {missing}")

    base = template[["order", "calib_id", "subject", "body"]].merge(
        meta[required_meta], on="calib_id", how="left"
    )
    if base[["urgency_proxy", "warmth_proxy"]].isna().any().any():
        raise SystemExit("calibration ids missing from meta file")

    def zscore(values: pd.Series) -> np.ndarray:
        array = values.to_numpy(dtype=float)
        std = array.std(ddof=0)
        return (array - array.mean()) / (std if std else 1.0)

    rng = np.random.default_rng(seed)
    n = len(base)
    is_spam = base["label"].to_numpy(dtype=float)
    urgency = 4 + 0.9 * zscore(base["urgency_proxy"]) + rng.normal(0, 0.45, n)
    warmth = 4 + 0.9 * zscore(base["warmth_proxy"]) + rng.normal(0, 0.45, n)
    valence = (4 + 0.55 * zscore(base["warmth_proxy"])
               - 0.35 * zscore(base["urgency_proxy"]) + rng.normal(0, 0.55, n))
    distance = (4 + 0.60 * zscore(base["urgency_proxy"])
                + 0.50 * (is_spam - 0.5) + rng.normal(0, 0.45, n))
    latents = {"valence": valence, "urgency": urgency,
               "warmth": warmth, "distance": distance}
    item_offsets = {item: rng.normal(0, 0.25) for item in ITEMS}
    rater_ids = [f"R{i}" for i in range(1, raters + 1)]
    rater_shift = {rater: rng.normal(0, 0.35) for rater in rater_ids}

    records = []
    for rater in rater_ids:
        for idx, row in base.iterrows():
            record = row.to_dict()
            record["rater_id"] = rater
            for dimension, items in DIMENSION_ITEMS.items():
                score = latents[dimension][idx]
                for item in items:
                    raw = score + item_offsets[item] + rater_shift[rater] + rng.normal(0, 0.75)
                    if item in REVERSE_ITEMS:
                        raw = (SCALE_MIN + SCALE_MAX) - raw
                    record[item] = int(np.clip(np.round(raw), SCALE_MIN, SCALE_MAX))
            # Diagnostic item: rare by construction, never part of a dimension score.
            record["V3_hostility"] = int(rng.choice([1, 2, 3], p=[0.92, 0.06, 0.02]))
            logit = 1.1 * (urgency[idx] - 4) + 0.8 * (is_spam[idx] - 0.5) - 0.9
            probability = 1.0 / (1.0 + np.exp(-logit))
            binary = int(rng.random() < probability)
            record["threat_binary"] = binary
            record["threat_severity"] = 0 if binary == 0 else int(np.clip(
                np.round(1.5 + 0.5 * (urgency[idx] - 4) + rng.normal(0, 0.4)), 1, 3
            ))
            record["confidence"] = int(rng.choice([1, 2, 3], p=[0.15, 0.45, 0.40]))
            record["notes"] = ""
            records.append(record)

    simulated = pd.DataFrame(records)
    return simulated[TEMPLATE_COLUMNS]


# ---------------------------------------------------------------------------
# report / manifest
# ---------------------------------------------------------------------------
def write_report(path: Path, item_df: pd.DataFrame, dim_df: pd.DataFrame,
                 rater_df: pd.DataFrame, dis_df: pd.DataFrame, args: argparse.Namespace,
                 n_ratings: int, n_raters: int, n_calib: int,
                 incomplete: pd.Series, overall: str, alpha_median: float,
                 simulated: bool) -> None:
    flagged = item_df[(item_df["flag"] != "") & (item_df["role"] == "scored")].sort_values("icc")
    warned = item_df[(item_df["warning"] != "") & (item_df["role"] == "scored")]
    lines = [
        f"# {args.label}分析报告",
        "",
    ]
    if simulated:
        lines += [
            "> **本报告由模拟评分生成，仅用于验证 ICC/α 计算流程；不可作为真实信度证据。**",
            "",
        ]
    lines += [
        "## 1. 输入",
        "",
        f"- 评分行数：{n_ratings}",
        f"- 评分员数：{n_raters}",
        f"- 条目数：{n_calib}",
        f"- bootstrap 次数：{args.bootstrap}",
        f"- 随机种子：{args.seed}",
        "",
    ]
    if len(incomplete):
        lines += [
            f"- **不完整条目：{len(incomplete)} 个**（至少一个评分员缺失）；"
            "这些条目已从信度计算中排除。",
            "",
        ]
    lines += [
        "## 2. 门槛",
        "",
        f"- 维度 ICC(2,k)：点估计 ≥ {args.target_icc:.2f}，bootstrap 95% CI 下限 ≥ {args.min_icc:.2f}；"
        "点估计 < 0.60 判为 FAIL。",
        f"- 维度 Krippendorff interval α：每个维度 ≥ {args.min_alpha:.2f}。",
        f"- 计分条目：ICC < {args.min_item_icc:.2f} 或校正题总相关 < {args.min_item_total:.2f} 会被标记并阻断 PASS。",
        (f"- 计分条目：任一评分员的 distinct levels < {args.min_distinct} 会被标记并阻断 PASS。"
         if args.min_distinct > 0 else
         "- 粗量化门槛未启用（`--min-distinct 0`）。"),
        "- 条目 ordinal α < 0.40 作为诊断警告（单条目 3 人评分时噪声较大），不阻断 PASS。",
        "- 诊断条目（`V3_hostility`）不进入 valence 合成，也不参与 PASS 门槛，只报告敌意出现率。",
        "",
        "## 3. 维度信度",
        "",
        md_table(dim_df[["dimension", "n_items", "n_cases", "mean_score",
                         "mean_inter_item_r", "icc", "ci_low", "ci_high",
                         "alpha", "verdict"]]),
        "",
        "## 4. 条目信度",
        "",
        md_table(item_df[["item", "dimension", "role", "mean", "sd", "mean_range",
                          "min_distinct", "icc", "ci_low", "ci_high", "alpha",
                          "item_total_r", "warning", "flag"]]),
        "",
        "## 5. 评分员使用量表情况",
        "",
        md_table(rater_df),
        "",
    ]
    if len(dis_df):
        lines += [
            "## 6. 分歧最大的条目（range ≥ 3 或 SD ≥ 1.5）",
            "",
            md_table(dis_df[["calib_id", "item", "dimension", "ratings",
                             "range", "sd"]], max_rows=20),
            "",
        ]
    else:
        lines += [
            "## 6. 分歧",
            "",
            "没有条目的评分极差 ≥ 3 或 SD ≥ 1.5。",
            "",
        ]

    lines += [
        "## 7. 校准结论",
        "",
        f"- 总体判定：**{overall}**",
        f"- 维度 interval α 最小值：{dim_df['alpha'].min():.3f}",
        f"- 条目 ordinal α 中位数（诊断）：{alpha_median:.3f}",
    ]
    if len(flagged):
        lines.append(f"- 需要复核的计分条目：{', '.join(flagged['item'].tolist())}")
    else:
        lines.append("- 没有计分条目触发阻断性复核标记。")
    if len(warned):
        lines.append(f"- 诊断警告（不阻断 PASS）：{', '.join(warned['item'].tolist())}")
    diagnostic = item_df[item_df["role"] == "diagnostic"]
    if len(diagnostic):
        lines.append("- 诊断条目（不进入维度分）：" +
                     ", ".join(diagnostic["item"].tolist()))
    lines += [
        "",
        "建议动作：",
        "",
        "- **PASS**：维度门槛与条目检查全部通过；冻结评分手册与条目，生成 400 条正式评分模板。",
        "- **REVISE**：维度门槛通过但有条目被标记，或 CI 下限未达标；先讨论分歧、修订锚点/例子，必要时重跑 50 条校准。",
        "- **FAIL**：任一维度 ICC < 0.60；重新检查维度定义与条目措辞，不要直接进入 400 条正式标注。",
        "",
        "> 校准阶段的分数不并入正式分析；它只用于评分细则和信度检查。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    perfect = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3],
                        [4, 4, 4], [5, 5, 5]], dtype=float)
    assert np.isclose(icc_2k(perfect), 1.0), icc_2k(perfect)
    assert np.isclose(krippendorff_alpha(perfect, level="ordinal"), 1.0)
    assert np.isclose(krippendorff_alpha(perfect, level="interval"), 1.0)
    imperfect = np.array([[1, 2, 1], [2, 2, 3], [3, 3, 2],
                          [4, 5, 4], [5, 4, 5]], dtype=float)
    print("self-test: perfect ICC =", icc_2k(perfect),
          "| perfect alpha =", krippendorff_alpha(perfect, level="ordinal"))
    print("self-test: imperfect ICC =", icc_2k(imperfect),
          "| imperfect alpha =", krippendorff_alpha(imperfect, level="ordinal"))
    print("self-test passed")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E1 calibration ratings.")
    parser.add_argument("--ratings", nargs="+", default=None)
    parser.add_argument("--map", default=None,
                        help="QC map with sheet_id/is_anchor; anchor rows are dropped")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--template", default="annotation/e1_calibration_50_ratings_template.csv")
    parser.add_argument("--meta", default="samples/e1_calibration_50_meta.csv")
    parser.add_argument("--out", default="annotation/calibration_results")
    parser.add_argument("--label", default="E1 50 条校准", help="report title prefix")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-icc", type=float, default=0.70)
    parser.add_argument("--target-icc", type=float, default=0.75)
    parser.add_argument("--min-alpha", type=float, default=0.67)
    parser.add_argument("--min-item-icc", type=float, default=0.50,
                        help="scored-item ICC floor for blocking flags")
    parser.add_argument("--min-item-total", type=float, default=0.30,
                        help="scored-item corrected item-total floor")
    parser.add_argument("--min-distinct", type=int, default=0,
                        help="min distinct levels per rater x scored item; 0 disables")
    parser.add_argument("--raters", type=int, default=3, help="simulated rater count")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.simulate and args.ratings:
        raise SystemExit("use either --simulate or --ratings, not both")
    if not args.simulate and not args.ratings:
        raise SystemExit("pass --ratings ... or --simulate")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.simulate:
        simulated = simulate_ratings(Path(args.template), Path(args.meta),
                                     args.raters, args.seed)
        simulated_path = out_dir / "simulated_ratings.csv"
        simulated.to_csv(simulated_path, index=False, encoding="utf-8-sig")
        rating_paths = [simulated_path]
    else:
        rating_paths = [Path(p) for p in args.ratings]

    df = load_ratings(rating_paths, Path(args.map) if args.map else None)
    raters = sorted(df["rater_id"].unique())
    counts = df.groupby("calib_id")["rater_id"].nunique()
    incomplete = counts[counts < len(raters)]
    complete_ids = counts[counts == len(raters)].index
    df_complete = df[df["calib_id"].isin(complete_ids)].copy()
    if df_complete.empty:
        raise SystemExit("no fully rated calibration items")

    item_df = item_reliability(df_complete, args.bootstrap, args.seed,
                               args.min_item_icc, args.min_item_total,
                               args.min_distinct)
    dim_df = dimension_reliability(df_complete, args.bootstrap, args.seed,
                                   args.target_icc, args.min_icc, args.min_alpha)
    rater_df = rater_summary(df_complete)
    dis_df = disagreement_table(df_complete)

    scored_items = item_df[item_df["role"] == "scored"]
    item_alpha_median = float(scored_items["alpha"].median())
    dimension_alpha_min = float(dim_df["alpha"].min())
    alpha_gate = (np.isfinite(dim_df["alpha"]).all()
                  and (dim_df["alpha"] >= args.min_alpha).all())
    has_flagged_items = bool((scored_items["flag"] != "").any())
    if (dim_df["verdict"] == "FAIL").any():
        overall = "FAIL"
    elif (dim_df["verdict"] == "PASS").all() and alpha_gate and not has_flagged_items:
        overall = "PASS"
    else:
        overall = "REVISE"

    item_df.to_csv(out_dir / "item_reliability.csv", index=False, encoding="utf-8-sig")
    dim_df.to_csv(out_dir / "dimension_reliability.csv", index=False, encoding="utf-8-sig")
    rater_df.to_csv(out_dir / "rater_summary.csv", index=False, encoding="utf-8-sig")
    dis_df.to_csv(out_dir / "disagreements.csv", index=False, encoding="utf-8-sig")
    df_complete.to_csv(out_dir / "ratings_scored.csv", index=False, encoding="utf-8-sig")

    report_path = out_dir / "calibration_report.md"
    write_report(report_path, item_df, dim_df, rater_df, dis_df, args,
                 len(df), len(raters), len(counts), incomplete, overall,
                 item_alpha_median, args.simulate)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulated": bool(args.simulate),
        "ratings_files": [str(p) for p in rating_paths],
        "input_sha256": {str(p): sha256_file(p) for p in rating_paths},
        "qc_map": str(args.map) if args.map else None,
        "qc_map_sha256": sha256_file(Path(args.map)) if args.map else None,
        "n_rating_rows": int(len(df)),
        "n_raters": int(len(raters)),
        "n_calibration_items": int(len(counts)),
        "n_complete_items": int(len(complete_ids)),
        "n_incomplete_items": int(len(incomplete)),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "gates": {
            "target_icc": args.target_icc,
            "min_icc_ci_low": args.min_icc,
            "min_dimension_alpha": args.min_alpha,
            "min_item_icc": args.min_item_icc,
            "min_item_total": args.min_item_total,
            "min_distinct": args.min_distinct,
        },
        "item_alpha_median": item_alpha_median,
        "dimension_alpha_min": dimension_alpha_min,
        "dimension_alpha_gate": bool(alpha_gate),
        "overall": overall,
        "dimension_verdicts": dict(zip(dim_df["dimension"], dim_df["verdict"])),
        "outputs": sorted(p.name for p in out_dir.iterdir()
                          if p.is_file() and p.name != "calibration_manifest.json"),
    }
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"wrote {report_path}")
    print(f"overall={overall} item_alpha_median={item_alpha_median:.3f} "
          f"dimension_alpha_min={dimension_alpha_min:.3f}")
    print(dim_df[["dimension", "icc", "ci_low", "ci_high", "alpha", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
