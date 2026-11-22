#!/usr/bin/env python
"""Batch QC for the v1.3 400-item annotation.

Three checks:

1. Scale use: per rater x scored item (V3 diagnostic excluded), flag mode
   share > .80 or SD < .50.
2. Block drift: per rater x dimension, center each item on the across-rater
   item mean and compare the four block means; if max-min > .50, flag the
   block farthest from the rater's adjusted mean. Raw block means and ranges
   are reported alongside.
3. Anchor consistency: 40 repeated items (10 per block) are rated twice; flag
   mean absolute difference > 1.0 or any single difference > 2.

Usage:
    python annotation/batch_qc.py \
      --ratings annotation/e1_400_v1.3_R1.csv \
                annotation/e1_400_v1.3_R2.csv \
                annotation/e1_400_v1.3_R3.csv \
      --map annotation/e1_annotation_400_v1.3_qc_map.csv \
      --out annotation/e1_400_v1.3_qc_results
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_calibration as ac
from rating_schema import (DIMENSION_ITEMS, ITEMS, REVERSE_ITEMS, SCALE_MAX,
                           SCALE_MIN, SCORED_ITEMS)

MAP_COLUMNS = ["sheet_id", "is_anchor", "anchor_of", "source_block", "cell"]


def load_qc_ratings(paths: list[Path], map_path: Path,
                    rater_ids: list[str]) -> pd.DataFrame:
    mapping = pd.read_csv(map_path, encoding="utf-8-sig")
    missing = [column for column in MAP_COLUMNS if column not in mapping.columns]
    if missing:
        raise SystemExit(f"map is missing columns: {missing}")
    if not mapping["sheet_id"].is_unique:
        raise SystemExit("map sheet_id is not unique")
    expected_ids = set(mapping["sheet_id"])

    frames = []
    for path, rater in zip(paths, rater_ids):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.columns = [str(c).strip() for c in frame.columns]
        required = ["sheet_id", "block", "calib_id"] + ITEMS + \
                   ["threat_binary", "threat_severity", "confidence"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        if len(frame) != len(mapping) or set(frame["sheet_id"]) != expected_ids:
            raise SystemExit(f"{path}: expected {len(mapping)} rows with map sheet_id")
        if frame["sheet_id"].duplicated().any():
            raise SystemExit(f"{path}: duplicate sheet_id rows")
        frame["rater_id"] = rater
        frame = frame.merge(mapping[MAP_COLUMNS], on="sheet_id", how="left")
        if frame["is_anchor"].isna().any():
            raise SystemExit(f"{path}: sheet_id missing from map")

        for item in ITEMS:
            values = pd.to_numeric(frame[item], errors="coerce")
            bad = values.isna() | (values < SCALE_MIN) | (values > SCALE_MAX) | (values % 1 != 0)
            if bad.any():
                examples = frame.loc[bad, ["sheet_id", item]].head(5).to_dict("records")
                raise SystemExit(f"{path}: invalid {item}: {examples}")
            frame[item] = values.astype(int)

        binary = pd.to_numeric(frame["threat_binary"], errors="coerce")
        if binary.isna().any() or not binary.isin([0, 1]).all():
            raise SystemExit(f"{path}: threat_binary must be 0/1")
        frame["threat_binary"] = binary.astype(int)
        severity = pd.to_numeric(frame["threat_severity"], errors="coerce")
        if (severity.isna() & (frame["threat_binary"] == 1)).any():
            raise SystemExit(f"{path}: threat_severity missing while threat_binary=1")
        severity = severity.fillna(0)
        if not severity.isin([0, 1, 2, 3]).all():
            raise SystemExit(f"{path}: threat_severity must be 0-3")
        frame["threat_severity"] = severity.astype(int)
        frame.loc[(frame["threat_binary"] == 0) & (frame["threat_severity"] > 0),
                  "threat_severity"] = 0
        confidence = pd.to_numeric(frame["confidence"], errors="coerce")
        if confidence.isna().any() or not confidence.isin([1, 2, 3]).all():
            raise SystemExit(f"{path}: confidence must be 1/2/3")
        frame["confidence"] = confidence.astype(int)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    for item in ITEMS:
        df[f"{item}__scored"] = np.where(
            item in REVERSE_ITEMS, (SCALE_MIN + SCALE_MAX) - df[item], df[item]
        )
    for dimension, items in DIMENSION_ITEMS.items():
        scored = [f"{item}__scored" for item in items]
        df[f"{dimension}_score"] = df[scored].mean(axis=1)
    return df


def scale_use(df: pd.DataFrame, max_mode_share: float, min_sd: float) -> pd.DataFrame:
    # Diagnostic items such as V3_hostility are near-constant by design and
    # must not trigger the scale-use gate.
    base = df[df["is_anchor"] == 0]
    rows = []
    for rater, sub in base.groupby("rater_id"):
        for item in SCORED_ITEMS:
            values = sub[item]
            counts = values.value_counts()
            mode = int(counts.index[0])
            mode_share = float(counts.iloc[0] / len(values))
            sd = float(values.std(ddof=1))
            flags = []
            if mode_share > max_mode_share:
                flags.append("mode_share>.80")
            if sd < min_sd:
                flags.append("sd<.50")
            rows.append({
                "rater_id": rater,
                "item": item,
                "mean": float(values.mean()),
                "sd": sd,
                "mode": mode,
                "mode_share": mode_share,
                "flag": ";".join(flags),
            })
    return pd.DataFrame(rows)


def quantization(df: pd.DataFrame, min_distinct_warn: int,
                 min_distinct_block: int, max_gap_warn: int) -> pd.DataFrame:
    """Detect coarse scale use that mode share and SD can miss.

    An item scored with only a few distinct levels (or with large gaps between
    used levels) loses rank information even when its SD is large. The
    ``blocking`` column is only enforced when --quantization-gate is set, so
    v1.3 results remain reproducible.
    """
    base = df[df["is_anchor"] == 0]
    rows = []
    for rater, sub in base.groupby("rater_id"):
        for item in SCORED_ITEMS:
            values = np.sort(sub[item].unique())
            n_distinct = int(len(values))
            max_gap = int(np.max(np.diff(values))) if n_distinct > 1 else 0
            blocking = n_distinct < min_distinct_block
            flags = []
            if blocking:
                flags.append(f"distinct<{min_distinct_block}")
            elif n_distinct < min_distinct_warn:
                flags.append(f"distinct<{min_distinct_warn}")
            if max_gap >= max_gap_warn:
                flags.append(f"max_gap>={max_gap_warn}")
            rows.append({
                "rater_id": rater,
                "item": item,
                "n": int(len(sub)),
                "n_distinct": n_distinct,
                "values": " ".join(str(int(v)) for v in values),
                "max_gap": max_gap,
                "blocking": int(blocking),
                "flag": ";".join(flags),
            })
    return pd.DataFrame(rows)


def block_drift(df: pd.DataFrame, max_drift: float) -> pd.DataFrame:
    # Raw block means confound rater drift with item difficulty: a block can
    # look shifted simply because its emails are harder to rate. Center each
    # item's score on the across-rater item mean first, then compare blocks.
    base = df[df["is_anchor"] == 0].copy()
    for dimension in DIMENSION_ITEMS:
        column = f"{dimension}_score"
        base[f"{dimension}__centered"] = \
            base[column] - base.groupby("sheet_id")[column].transform("mean")

    rows = []
    for rater, rater_df in base.groupby("rater_id"):
        for dimension in DIMENSION_ITEMS:
            column = f"{dimension}_score"
            centered = f"{dimension}__centered"
            overall = float(rater_df[column].mean())
            block_means = rater_df.groupby("block")[column].mean()
            raw_range = float(block_means.max() - block_means.min())
            adjusted_means = rater_df.groupby("block")[centered].mean()
            adjusted_range = float(adjusted_means.max() - adjusted_means.min())
            # Reject only the block farthest from the rater's adjusted mean.
            # Flagging every block beyond the threshold would cascade: one
            # shifted block moves the overall mean and drags the others along.
            worst_block = (adjusted_means - adjusted_means.mean()).abs().idxmax() \
                if adjusted_range > max_drift else None
            for block, mean in block_means.items():
                deviation = float(mean - overall)
                adjusted_deviation = float(adjusted_means.loc[block])
                flags = []
                if block == worst_block:
                    flags.append("range_extreme")
                rows.append({
                    "rater_id": rater,
                    "dimension": dimension,
                    "block": int(block),
                    "n": int((rater_df["block"] == block).sum()),
                    "mean": float(mean),
                    "overall_mean": overall,
                    "deviation": deviation,
                    "drift_range": raw_range,
                    "adjusted_mean": adjusted_means.loc[block],
                    "adjusted_deviation": adjusted_deviation,
                    "adjusted_drift_range": adjusted_range,
                    "flag": ";".join(flags),
                })
    return pd.DataFrame(rows)


def anchor_consistency(df: pd.DataFrame, max_mean_abs: float,
                       max_abs: float) -> pd.DataFrame:
    originals = df[df["is_anchor"] == 0][["rater_id", "sheet_id"] + ITEMS].copy()
    originals = originals.rename(columns={"sheet_id": "anchor_of"})
    originals = originals.rename(columns={item: f"{item}__orig" for item in ITEMS})
    repeats = df[df["is_anchor"] == 1][
        ["rater_id", "sheet_id", "anchor_of", "block", "source_block", "cell"] + ITEMS
    ].copy()
    repeats = repeats.rename(columns={item: f"{item}__repeat" for item in ITEMS})
    pairs = repeats.merge(originals, on=["rater_id", "anchor_of"], how="left")
    missing = [f"{item}__orig" for item in ITEMS
               if pairs[f"{item}__orig"].isna().any()]
    if missing:
        raise SystemExit(f"anchor original ratings missing: {missing}")

    rows = []
    for rater, sub in pairs.groupby("rater_id"):
        all_signed = []
        all_abs = []
        for item in ITEMS:
            diff = sub[f"{item}__repeat"] - sub[f"{item}__orig"]
            abs_diff = diff.abs()
            all_signed.extend(diff.tolist())
            all_abs.extend(abs_diff.tolist())
            flags = []
            if abs_diff.mean() > max_mean_abs:
                flags.append("mean_abs>1.0")
            if abs_diff.max() > max_abs:
                flags.append("max>2")
            rows.append({
                "rater_id": rater,
                "item": item,
                "n_pairs": int(len(diff)),
                "mean_diff": float(diff.mean()),
                "mean_abs_diff": float(abs_diff.mean()),
                "max_abs_diff": float(abs_diff.max()),
                "flag": ";".join(flags),
            })
        signed = np.asarray(all_signed, dtype=float)
        absolute = np.asarray(all_abs, dtype=float)
        only_errors = np.abs(signed) > 0
        flags = []
        if absolute.mean() > max_mean_abs:
            flags.append("mean_abs>1.0")
        if absolute.max() > max_abs:
            flags.append("max>2")
        rows.append({
            "rater_id": rater,
            "item": "__overall__",
            "n_pairs": int(len(signed)),
            "mean_diff": float(signed.mean()),
            "mean_abs_diff": float(absolute.mean()),
            "max_abs_diff": float(absolute.max()),
            "n_nonzero": int(only_errors.sum()),
            "flag": ";".join(flags),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", nargs=3, type=Path, required=True)
    parser.add_argument("--map", type=Path,
                        default=Path("annotation/e1_annotation_400_v1.3_qc_map.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("annotation/e1_400_v1.3_qc_results"))
    parser.add_argument("--rater-ids", nargs=3, default=["R1", "R2", "R3"])
    parser.add_argument("--max-mode-share", type=float, default=0.80)
    parser.add_argument("--min-sd", type=float, default=0.50)
    parser.add_argument("--max-block-drift", type=float, default=0.50)
    parser.add_argument("--max-anchor-mean-abs", type=float, default=1.0)
    parser.add_argument("--max-anchor-max", type=float, default=2.0)
    parser.add_argument("--quantization-gate", action="store_true",
                        help="treat distinct-level < --min-distinct-block as a "
                             "blocking QC flag (v2.0 workflow)")
    parser.add_argument("--min-distinct-warn", type=int, default=5)
    parser.add_argument("--min-distinct-block", type=int, default=4)
    parser.add_argument("--max-gap-warn", type=int, default=2)
    parser.add_argument("--label", default="E1 400 条批次 QC")
    args = parser.parse_args()

    df = load_qc_ratings(args.ratings, args.map, args.rater_ids)
    scale_df = scale_use(df, args.max_mode_share, args.min_sd)
    quant_df = quantization(df, args.min_distinct_warn,
                            args.min_distinct_block, args.max_gap_warn)
    drift_df = block_drift(df, args.max_block_drift)
    anchor_df = anchor_consistency(df, args.max_anchor_mean_abs, args.max_anchor_max)

    scale_flags = scale_df[scale_df["flag"] != ""]
    quant_flags = quant_df[quant_df["flag"] != ""]
    quant_blocks = quant_df[quant_df["blocking"] == 1]
    drift_flags = drift_df[drift_df["flag"] != ""]
    anchor_overall = anchor_df[anchor_df["item"] == "__overall__"]
    anchor_flags = anchor_df[anchor_df["flag"] != ""]
    anchor_item_flags = anchor_df[(anchor_df["item"] != "__overall__")
                                  & (anchor_df["flag"] != "")]
    rejected_blocks = sorted(
        {(row.rater_id, row.block) for row in drift_flags.itertuples()}
    )
    quantization_failed = args.quantization_gate and not quant_blocks.empty
    verdict = "PASS" if (scale_flags.empty and drift_flags.empty
                         and anchor_flags.empty
                         and not quantization_failed) else "REVISE"

    args.out.mkdir(parents=True, exist_ok=True)
    scale_df.to_csv(args.out / "qc_scale_use.csv", index=False, encoding="utf-8-sig")
    quant_df.to_csv(args.out / "qc_quantization.csv", index=False, encoding="utf-8-sig")
    drift_df.to_csv(args.out / "qc_block_drift.csv", index=False, encoding="utf-8-sig")
    anchor_df.to_csv(args.out / "qc_anchor_consistency.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "ratings_files": [str(path) for path in args.ratings],
        "input_sha256": {str(path): ac.sha256_file(path) for path in args.ratings},
        "map_file": str(args.map),
        "map_sha256": ac.sha256_file(args.map),
        "raters": list(args.rater_ids),
        "n_rows": int(len(df)),
        "n_unique_items": int((df["is_anchor"] == 0).sum() // len(args.rater_ids)),
        "n_anchor_pairs_per_rater": int((df["is_anchor"] == 1).sum() // len(args.rater_ids)),
        "thresholds": {
            "max_mode_share": args.max_mode_share,
            "min_sd": args.min_sd,
            "max_block_drift": args.max_block_drift,
            "max_anchor_mean_abs": args.max_anchor_mean_abs,
            "max_anchor_max": args.max_anchor_max,
            "quantization_gate": bool(args.quantization_gate),
            "min_distinct_warn": args.min_distinct_warn,
            "min_distinct_block": args.min_distinct_block,
            "max_gap_warn": args.max_gap_warn,
        },
        "counts": {
            "scale_flags": int(len(scale_flags)),
            "quantization_flags": int(len(quant_flags)),
            "quantization_blocks": int(len(quant_blocks)),
            "block_flags": int(len(drift_flags)),
            "anchor_flags": int(len(anchor_flags)),
        },
        "rejected_blocks": [
            {"rater_id": rater, "block": int(block)}
            for rater, block in rejected_blocks
        ],
    }
    (args.out / "qc_verdict.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        f"# {args.label}报告",
        "",
        f"- 判定：**{verdict}**",
        f"- 评分文件：{', '.join(path.name for path in args.ratings)}",
        f"- 映射：`{args.map.as_posix()}`",
        f"- 行数：{len(df)}（每人 {len(df) // len(args.rater_ids)} 行）",
        "",
        "## 1. 量表使用",
        "",
        "规则：评分员 × 计分条目众数占比 > .80 或 SD < .50 → 标记（`V3_hostility` 为诊断项，不参与）。",
        "",
        ac.md_table(scale_flags if not scale_flags.empty else scale_df,
                    max_rows=None if not scale_flags.empty else 12),
        "",
        "## 2. 区块均值漂移",
        "",
        "规则：同一评分员 × 维度，各块相对于跨评分员条目均值的偏差 max−min > .50 → 退回极端区块。原始块均值与调整后范围同时列出。",
        "",
        ac.md_table(drift_df[drift_df["flag"] != ""] if not drift_flags.empty else drift_df,
                    max_rows=None if not drift_flags.empty else 20),
        "",
        "## 3. 重复锚题一致性",
        "",
        "规则：重复 − 原题的平均绝对差 > 1.0，或单条最大绝对差 > 2 → 标记。",
        "",
        ac.md_table(anchor_overall, max_rows=None),
        "",
    ]
    if not anchor_item_flags.empty:
        lines.extend([
            "单条差异 > 2 的锚题：",
            "",
            ac.md_table(anchor_item_flags, max_rows=None),
            "",
        ])
    quant_note = ("已启用阻断：distinct levels < "
                  f"{args.min_distinct_block} 会使判定为 REVISE。"
                  if args.quantization_gate else
                  "未启用阻断；仅作为诊断警告。")
    lines.extend([
        "## 4. 粗量化诊断",
        "",
        f"规则：评分员 × 计分条目，distinct levels < {args.min_distinct_warn} "
        f"警告、< {args.min_distinct_block} 阻断；相邻已用档位最大间隔 ≥ "
        f"{args.max_gap_warn} 警告。{quant_note}",
        "",
        ac.md_table(quant_flags if not quant_flags.empty else quant_df,
                    max_rows=None if not quant_flags.empty else 15),
        "",
        "## 5. 处理建议",
        "",
        "- 量表使用被标记：核对对应条目的锚点；必要时重评该评分员的整块。",
        "- 粗量化被标记：该评分员只用了少数档位，排序信息已丢失；重评该条目并强调中间档（3/5）可用。",
        "- 区块漂移被标记：退回 `rejected_blocks` 中的区块重新评分。",
        "- 锚题不一致：说明同一评分员在批次内改变了判断标准，优先重评相关区块。",
        "- QC 未 PASS 时不要运行主分析。",
    ])
    (args.out / "qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out / 'qc_report.md'}")
    print(f"overall={verdict} scale_flags={len(scale_flags)} "
          f"quant_flags={len(quant_flags)} quant_blocks={len(quant_blocks)} "
          f"block_flags={len(drift_flags)} anchor_flags={len(anchor_flags)}")
    if rejected_blocks:
        print(f"rejected_blocks={rejected_blocks}")


if __name__ == "__main__":
    main()
