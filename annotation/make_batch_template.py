#!/usr/bin/env python
"""Build the v1.3 400-item QC rating sheet with repeated anchor items.

The sheet contains the 400 unique items plus `anchors-per-block` repeats in
each of the four 100-item blocks. Repeats are drawn from the previous block
(1 -> 2, 2 -> 3, 3 -> 4, 4 -> 1), stratified by cell, so every block gets a
balanced set of anchors from a different part of the batch.

Outputs:
    e1_annotation_400_v1.3_qc_template.csv   rating sheet (23 columns)
    e1_annotation_400_v1.3_qc_map.csv        sheet_id -> is_anchor/anchor_of
    e1_annotation_400_v1.3_qc_design.md      design and QC rules

Usage:
    python annotation/make_batch_template.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rating_schema import ITEMS, THREAT_COLUMNS

RATING_COLUMNS = (["rater_id", "sheet_id", "block", "order", "calib_id",
                   "subject", "body"] + ITEMS + THREAT_COLUMNS
                  + ["confidence", "notes"])


def md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table, avoids a tabulate dependency."""
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def pick_anchor_ids(block_df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """Pick n item ids from one block, stratified by cell."""
    cells = sorted(block_df["cell"].dropna().unique())
    if not cells:
        raise SystemExit("no cell labels in items metadata")
    base, extra = divmod(n, len(cells))
    picked: list[str] = []
    for index, cell in enumerate(cells):
        count = base + (1 if index < extra else 0)
        ids = block_df.loc[block_df["cell"] == cell, "internal_id"].to_numpy(copy=True)
        if len(ids) < count:
            raise SystemExit(f"cell {cell} has only {len(ids)} items, need {count}")
        rng.shuffle(ids)
        picked.extend(ids[:count].tolist())
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path,
                        default=Path("annotation/e1_annotation_400_ratings_template.csv"))
    parser.add_argument("--meta", type=Path, default=Path("samples/e1_items.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("annotation/e1_annotation_400_v1.3_qc_template.csv"))
    parser.add_argument("--map-out", type=Path,
                        default=Path("annotation/e1_annotation_400_v1.3_qc_map.csv"))
    parser.add_argument("--design-out", type=Path,
                        default=Path("annotation/e1_annotation_400_v1.3_qc_design.md"))
    parser.add_argument("--anchors-per-block", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = pd.read_csv(args.items, encoding="utf-8-sig")
    meta = pd.read_csv(args.meta, encoding="utf-8-sig")[["sample_id", "cell"]]
    items = items.merge(meta, left_on="calib_id", right_on="sample_id", how="left")
    if items["cell"].isna().any():
        missing = items.loc[items["cell"].isna(), "calib_id"].tolist()[:5]
        raise SystemExit(f"missing cell labels for {missing}")

    # Allow a minimal items sheet (order/calib_id/subject/body) by filling the
    # remaining rating columns with blanks; they are blanked again below.
    for column in RATING_COLUMNS:
        if column not in items.columns:
            items[column] = ""

    if len(items) != 400:
        raise SystemExit(f"expected 400 items, found {len(items)}")
    items = items.sort_values("order").reset_index(drop=True)
    items["block"] = ((items["order"] - 1) // 100 + 1).astype(int)
    items["internal_id"] = [f"S{i:04d}" for i in range(1, len(items) + 1)]
    items["is_anchor"] = 0
    items["anchor_of"] = ""
    items["source_block"] = items["block"]

    rng = np.random.default_rng(args.seed)
    anchor_rows = []
    for source_block in sorted(items["block"].unique()):
        target_block = source_block % 4 + 1
        pool = items[items["block"] == source_block]
        picked = pick_anchor_ids(pool, args.anchors_per_block, rng)
        for index, internal_id in enumerate(sorted(picked), start=1):
            row = items.loc[items["internal_id"] == internal_id].iloc[0]
            anchor_rows.append({
                "rater_id": "",
                "internal_id": f"A{source_block}{index:02d}",
                "block": target_block,
                "calib_id": row["calib_id"],
                "subject": row["subject"],
                "body": row["body"],
                "is_anchor": 1,
                "anchor_of": internal_id,
                "source_block": source_block,
                "cell": row["cell"],
            })
    anchors = pd.DataFrame(anchor_rows)

    # Shuffle each block independently, then assign one global presentation order.
    blocks = []
    for block in sorted(items["block"].unique()):
        block_rows = pd.concat([items[items["block"] == block],
                                anchors[anchors["block"] == block]],
                               ignore_index=True)
        order = rng.permutation(len(block_rows))
        blocks.append(block_rows.iloc[order].reset_index(drop=True))
    sheet = pd.concat(blocks, ignore_index=True)
    sheet["order"] = np.arange(1, len(sheet) + 1)

    # Opaque row ids: originals and anchors use the same R#### format, so raters
    # cannot tell which rows are repeats. anchor_of is remapped to final ids.
    sheet["sheet_id"] = [f"R{i:04d}" for i in range(1, len(sheet) + 1)]
    final_by_internal = sheet.set_index("internal_id")["sheet_id"]
    sheet["anchor_of"] = sheet["anchor_of"].map(final_by_internal).fillna("")

    rating_sheet = sheet[RATING_COLUMNS].copy()
    for column in ITEMS + THREAT_COLUMNS + ["confidence", "notes"]:
        rating_sheet[column] = ""
    mapping = sheet[["sheet_id", "calib_id", "block", "is_anchor",
                     "anchor_of", "source_block", "cell"]].copy()

    # Validation before writing.
    if not rating_sheet["sheet_id"].is_unique:
        raise SystemExit("sheet_id is not unique")
    expected_rows = len(items) + 4 * args.anchors_per_block
    if len(rating_sheet) != expected_rows:
        raise SystemExit(f"expected {expected_rows} rows, found {len(rating_sheet)}")
    if not mapping[mapping["is_anchor"] == 0]["calib_id"].is_unique:
        raise SystemExit("original calib_id is not unique")
    if not mapping.loc[mapping["is_anchor"] == 1, "anchor_of"].is_unique:
        raise SystemExit("an item is used as an anchor more than once")
    if not set(mapping.loc[mapping["is_anchor"] == 1, "anchor_of"]) <= set(
            mapping.loc[mapping["is_anchor"] == 0, "sheet_id"]):
        raise SystemExit("anchor_of points to a missing original")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rating_sheet.to_csv(args.out, index=False, encoding="utf-8-sig")
    mapping.to_csv(args.map_out, index=False, encoding="utf-8-sig")

    block_counts = (mapping.groupby(["block", "is_anchor"]).size()
                    .unstack(fill_value=0).rename(columns={0: "unique", 1: "anchors"}))
    cell_counts = (mapping[mapping["is_anchor"] == 1]
                   .groupby(["source_block", "cell"]).size().unstack(fill_value=0))
    anchor_table = mapping[mapping["is_anchor"] == 1][
        ["sheet_id", "calib_id", "source_block", "block", "cell"]]
    anchor_table = anchor_table.rename(columns={"block": "target_block"})

    lines = [
        "# E1 400 条 v1.3 QC 评分表设计",
        "",
        f"- 生成脚本：`annotation/make_batch_template.py`（seed={args.seed}）",
        f"- 评分表：`{args.out.as_posix()}`（{len(rating_sheet)} 行 × {len(RATING_COLUMNS)} 列）",
        f"- 映射表：`{args.map_out.as_posix()}`（只给分析脚本，不给评分员）",
        f"- 400 条唯一邮件 + 4 × {args.anchors_per_block} 条重复锚题 = {expected_rows} 行",
        "- 区块来源→目标：1→2、2→3、3→4、4→1；每块按 cell 分层抽锚题",
        "",
        "## 区块结构",
        "",
        md_table(block_counts.reset_index()),
        "",
        "## 锚题来源（按 cell）",
        "",
        md_table(cell_counts.reset_index()),
        "",
        "## 锚题清单",
        "",
        md_table(anchor_table),
        "",
        "## QC 规则",
        "",
        "| 检查 | 阈值 | 处理 |",
        "|---|---|---|",
        "| 评分员 × 计分条目众数占比 | > .80 | 量表使用异常；复核/重评该条目的区块 |",
        "| 评分员 × 计分条目 SD | < .50 | 同上 |",
        "| 区块均值漂移（同一评分员 × 维度，4 块均值 max−min） | > .50 | 退回漂移最大的区块 |",
        "| 重复锚题平均绝对差（重复 − 原题） | > 1.0 | 评分员批次内不一致；重评相关区块 |",
        "| 重复锚题单条最大绝对差 | > 2 | 同上 |",
        "",
        "`V3_hostility` 是稀有事件诊断项，常见众数占比 > .80、SD < .50，这是结构性特征，不参与量表使用检查。",
        "",
        "主分析使用 `analyze_calibration.py --map` 自动剔除 `is_anchor=1` 的行。",
    ]
    args.design_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out} ({len(rating_sheet)} rows)")
    print(f"wrote {args.map_out} ({len(mapping)} rows, {int(mapping['is_anchor'].sum())} anchors)")
    print(f"wrote {args.design_out}")
    print(block_counts.to_string())


if __name__ == "__main__":
    main()
