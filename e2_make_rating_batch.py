#!/usr/bin/env python3
"""Build the E2 v1.3 rating batch: 100 seeds x 6 versions + 40 hidden anchors.

Sampling (prereg section 5): from the complete-pair seeds (all five conditions
included), take the first 100 by SHA256(seed_id). Each seed contributes the
original plus its five included rewrites = 600 unique emails. Four blocks of
150 unique rows each receive 10 repeated anchors from the previous block,
balanced two per condition, giving 640 rows.

Anchors share their source item's calib_id and are identified only in the map
file, so raters cannot tell which rows are repeats.

Outputs
  e2_ratings/e2_annotation_600_template.csv   640-row rating sheet
  e2_ratings/e2_annotation_600_map.csv        sheet_id -> item/condition/anchor
  e2_ratings/e2_annotation_600_design.md      design + QC rules
  e2_ratings/README_E2_RATING.md              instructions for raters
  e2_ratings/e2_seed_sample.csv               100 selected seeds
"""
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "annotation"))
from rating_schema import ITEMS, THREAT_COLUMNS  # noqa: E402

OUT = ROOT / "e2_ratings"
PAIRS = ROOT / "e2_analysis/e2_pairs_G1.csv"
COMPLETE = ROOT / "e2_validation/G1_inclusion_round3_recode_final_audit_adjusted_screen_adjusted_complete_pairs.csv"
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    pairs = pd.read_csv(PAIRS, encoding="utf-8-sig")
    complete = pd.read_csv(COMPLETE, encoding="utf-8-sig")["seed_id"].astype(str).tolist()
    seeds = sorted(complete, key=h)[:100]
    assert len(seeds) == 100

    rng = np.random.default_rng(42)
    block_counts = [0, 0, 0, 0]
    items = []
    for seed in seeds:
        sub = pairs[pairs["seed_id"].astype(str) == seed]
        assert set(sub["condition"]) == set(CONDS), seed
        first = sub.iloc[0]
        versions = [{
            "seed_id": seed,
            "condition": "original",
            "subject": first["subject_original"],
            "body": first["body_original"],
        }]
        for cond in CONDS:
            row = sub[sub["condition"] == cond].iloc[0]
            versions.append({
                "seed_id": seed,
                "condition": cond,
                "subject": row["subject_rewrite"],
                "body": row["body_rewrite"],
            })
        rng.shuffle(versions)
        for version in versions:
            block = min(range(4), key=lambda i: (block_counts[i], rng.random()))
            block_counts[block] += 1
            version["block"] = block + 1
            items.append(version)

    assert block_counts == [150] * 4, block_counts
    for index, item in enumerate(items, start=1):
        item["internal_id"] = f"U{index:04d}"
        item["calib_id"] = f"E2V-{index:04d}"

    # Anchors: 10 per block from the previous block, two per condition.
    anchors, used = [], set()
    for block in range(1, 5):
        source = block - 1 if block > 1 else 4
        position = 0
        for cond in CONDS:
            pool = [it for it in items if it["block"] == source
                    and it["condition"] == cond and it["internal_id"] not in used]
            picked = rng.choice(len(pool), size=2, replace=False)
            for idx in picked:
                src = pool[int(idx)]
                used.add(src["internal_id"])
                position += 1
                anchors.append({**src, "block": block, "anchor_of": src["internal_id"],
                                "source_block": source,
                                "internal_id": f"A{block}{position:02d}"})

    rows = []
    for item in items:
        rows.append({**item, "is_anchor": 0, "anchor_of": "",
                     "source_block": item["block"]})
    for item in anchors:
        rows.append({**item, "is_anchor": 1})

    blocks = []
    for block in range(1, 5):
        sub = [r for r in rows if r["block"] == block]
        order = rng.permutation(len(sub))
        blocks.extend([sub[i] for i in order])
    sheet = pd.DataFrame(blocks)
    sheet["order"] = np.arange(1, len(sheet) + 1)
    sheet["sheet_id"] = [f"R{i:04d}" for i in range(1, len(sheet) + 1)]
    final_id = sheet.set_index("internal_id")["sheet_id"]
    sheet["anchor_of"] = sheet["anchor_of"].map(final_id).fillna("")

    # Validation
    assert len(sheet) == 640
    assert sheet["sheet_id"].is_unique
    unique = sheet[sheet["is_anchor"] == 0]
    assert len(unique) == 600
    assert unique["calib_id"].is_unique
    assert sheet.groupby(["block", "is_anchor"]).size().to_dict() == {
        (b, 0): 150 for b in range(1, 5)} | {(b, 1): 10 for b in range(1, 5)}
    assert not sheet.loc[sheet["is_anchor"] == 1, "anchor_of"].duplicated().any()
    assert set(sheet.loc[sheet["is_anchor"] == 1, "anchor_of"]) <= set(unique["sheet_id"])
    for anchor in sheet[sheet["is_anchor"] == 1].itertuples():
        source_calib = sheet.loc[sheet["sheet_id"] == anchor.anchor_of, "calib_id"].iloc[0]
        assert anchor.calib_id == source_calib

    OUT.mkdir(exist_ok=True)
    columns = (["rater_id", "sheet_id", "block", "order", "calib_id", "subject", "body"]
               + ITEMS + THREAT_COLUMNS + ["confidence", "notes"])
    rating = sheet.copy()
    for column in columns:
        if column not in rating.columns:
            rating[column] = ""
    rating = rating[columns]
    rating.to_csv(OUT / "e2_annotation_600_template.csv", index=False, encoding="utf-8-sig")

    mapping = sheet[["sheet_id", "calib_id", "block", "source_block", "order",
                     "seed_id", "condition", "is_anchor", "anchor_of", "internal_id"]].copy()
    mapping["cell"] = mapping["condition"]
    mapping = mapping.rename(columns={"block": "target_block"})
    mapping.to_csv(OUT / "e2_annotation_600_map.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"seed_id": seeds}).to_csv(OUT / "e2_seed_sample.csv", index=False,
                                            encoding="utf-8-sig")

    design = f"""# E2 温度子样本 600 + 40 锚题：设计

- 生成脚本：`e2_make_rating_batch.py`（seed=42）
- 评分表：`e2_ratings/e2_annotation_600_template.csv`（640 行 × {len(columns)} 列）
- 映射表：`e2_ratings/e2_annotation_600_map.csv`（只给分析脚本，不给评分员）
- 抽样：115 个完整配对 seed 中按 SHA256(seed_id) 取前 100；每 seed 原文 + 5 个条件各 1 条纳入改写 = 600 条唯一邮件
- 锚题：4 块 × 10 条重复（来源 1→2、2→3、3→4、4→1），每块每条件 2 条；锚题沿用原题 `calib_id`，只在 map 中标记
- `calib_id` 为不透明版本号（E2V-####），评分表中没有 seed、条件或版本标签

## 区块结构

| block | 唯一 | 锚题 | 合计 |
|---|---|---|---|
| 1 | 150 | 10 | 160 |
| 2 | 150 | 10 | 160 |
| 3 | 150 | 10 | 160 |
| 4 | 150 | 10 | 160 |

## QC 规则（沿用 v1.3）

| 检查 | 阈值 | 处理 |
|---|---|---|
| 评分员 × 计分条目众数占比 | > .80 | 量表使用异常；复核该条目 |
| 评分员 × 计分条目 SD | < .50 | 同上 |
| 区块均值漂移（评分员 × 维度，4 块均值 max−min） | > .50 | 退回漂移最大的区块 |
| 重复锚题平均绝对差（重复 − 原题） | > 1.0 | 批次内不一致；重评相关区块 |
| 重复锚题单条最大绝对差 | > 2 | 同上 |

`V3_hostility` 为稀有事件诊断项，不参与量表使用检查。

## 分析口径

- 用 `annotation/analyze_calibration.load_ratings` + map 自动剔除锚题；
- 每评分员 × 维度在 600 条唯一邮件内做 z 分；
- 每条件 vs 原文的 seed 内配对差，报告效应量与 bootstrap 95% CI；
- distance 标为探索性；不做跨评分员绝对比较、不做 EFA/CFA。
"""
    (OUT / "e2_annotation_600_design.md").write_text(design, encoding="utf-8")

    readme = f"""# E2 邮件情感温度评分任务（640 行）

请使用 v1.3 评分手册（`annotation/RATING_CODEBOOK_v1.3.md`）对 `e2_annotation_600_template.csv` 的 **每一行** 按 1–7 同意度量表评分。

## 文件

- 评分表：`e2_annotation_600_template.csv`，共 640 行，列结构与 E1 400 条批次相同。
- 手册：`RATING_CODEBOOK_v1.3.md`（条目定义、锚点、threat 规则）。
- 不要修改 `sheet_id`、`block`、`order`、`calib_id`、`subject`、`body` 六列，也不要增删行。

## 填法

1. `rater_id` 填你的编号（R1 / R2 / R3）。
2. 12 个条目按 1–7 整数评分：V1、V2、V3、A1、A2、A3、W1、W2、W3、D1、D2、D3。
3. `threat_binary` 填 0/1；`threat_severity` 仅当 binary=1 时填 1–3，否则填 0。
4. `confidence` 填 1–3（1=低，3=高）；`notes` 可留空。
5. 全部 640 行都要填满；不要留空、不要用 4 分代替判断。
6. 逐行独立评分，不要回头对照其他行，也不要猜测哪些行是重复题。
7. 建议每 80–100 行休息一次，分 6–8 次完成；不要参考其他评分员的结果。

## 交回

- R1：`e2_600_R1_filled.csv`
- R2：`e2_600_R2_filled.csv`
- R3：`e2_600_R3_filled.csv`

交回前检查：640 行、`rater_id` 已填、12 个条目与两个 threat 列无空值。
"""
    (OUT / "README_E2_RATING.md").write_text(readme, encoding="utf-8")

    print(f"items={len(items)} anchors={len(anchors)} rows={len(sheet)}")
    print(f"blocks={sheet.groupby('block').size().to_dict()}")
    print(f"conditions={sheet[sheet['is_anchor']==0].groupby('condition').size().to_dict()}")
    print(f"wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
