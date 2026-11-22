#!/usr/bin/env python3
"""Build one self-contained rating folder per rater (R1/R2/R3).

Each folder gets
  README_FIRST.md            E2-specific instructions for that rater
  RATING_CODEBOOK_v1.3.md    frozen codebook (verbatim copy)
  COLUMN_GUIDE_v1.3.md       frozen column guide (verbatim copy)
  e2_600_<rater>_filled.csv  640-row sheet with rater_id pre-filled

The map file, anchor labels, seed ids and conditions are never copied.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "e2_ratings/e2_annotation_600_template.csv"
CODEBOOK = ROOT / "annotation/RATING_CODEBOOK_v1.3.md"
COLUMNS = ROOT / "annotation/COLUMN_GUIDE_v1.3.md"
OUT = ROOT / "e2_ratings/packages"
RATERS = ["R1", "R2", "R3"]


def readme(rater):
    return f"""# E2 邮件情感温度评分任务（640 行）— {rater}

本文件夹是自包含任务包。请只评 `e2_600_{rater}_filled.csv` 中的邮件，并只使用本文件夹内的两份参考文件。

## 文件

- `e2_600_{rater}_filled.csv`：**要填的评分表**，640 行 × 23 列。`rater_id` 已预填为 `{rater}`，不要修改。
- `RATING_CODEBOOK_v1.3.md`：评分手册（维度定义、12 个条目、1–7 量表、威胁编码）。
- `COLUMN_GUIDE_v1.3.md`：逐列字段说明。第 13 节给出 23 列评分表的列顺序；其余章节中的列号按 21 列校准表编号，本批次请以第 13 节为准。

两份参考文件原本为 E1 批次编写，条目定义、量表、威胁规则和 23 列结构与本批次完全相同。文中出现的 E1 文件名、440 行、每块 110 行等数字属于上一批次；**本批次是 640 行，4 个区块，每块 160 行**。不要去找文中的 E1 文件。

## 任务

对每一行的 `subject` 和 `body` 独立评分。只依据文本表达出来的情感姿态，不判断邮件是否为垃圾/钓鱼，不查发件人，不打开链接或附件。

1. 12 个条目全部按 **1–7 整数**评分：V1、V2、V3、A1、A2、A3、W1、W2、W3、D1、D2、D3。
2. 使用完整量表，不要习惯性给 4 分；只有文本确实中性、无法判断时才填 4。
3. `threat_binary` 填 0 或 1；`threat_severity` 仅在 binary=1 时填 1–3，否则填 0。
4. `confidence` 填 1–3（1=低，3=高）。
5. `notes` 可留空；如需解释，只写一句评分理由，不写链接、姓名或其他个人信息。
6. **640 行全部填满**，不要留空、不要增删行、不要修改前 7 列（`rater_id` 到 `body`）。
7. 逐行独立评分；不要回头对照其他行，也不要猜测或标记哪些行是重复题。表中出现少量重复邮件是正常的批次质检安排，两行都要独立评分。
8. 不要参考其他评分员的结果。建议每 80–100 行休息一次，分 6–8 次完成。

## 交回

直接交回填好的 `e2_600_{rater}_filled.csv`，文件名和格式不变（UTF-8 CSV）。

交回前自检：

- 640 条数据记录、23 列（正文单元格可含换行，不要按物理行数判断记录数）；
- `rater_id` 全部为 `{rater}`；
- 12 个条目、`threat_binary`、`threat_severity`、`confidence` 均无空值；
- 所有条目为 1–7 整数，`threat_binary` 为 0/1，`threat_severity` 为 0–3，`confidence` 为 1–3。

如有疑问，先停下来询问任务发布者，不要自行改动评分规则。
"""


def main():
    template = pd.read_csv(SRC, encoding="utf-8-sig")
    assert template.shape == (640, 23), template.shape
    assert template["sheet_id"].is_unique
    assert template.iloc[:, 7:].isna().all().all(), "rating columns must start empty"
    assert template["subject"].notna().all() and template["body"].notna().all()
    assert template.groupby("block").size().to_dict() == {1: 160, 2: 160, 3: 160, 4: 160}

    codebook = CODEBOOK.read_text(encoding="utf-8")
    column_guide = COLUMNS.read_text(encoding="utf-8")

    for rater in RATERS:
        folder = OUT / f"rater_{rater}"
        folder.mkdir(parents=True, exist_ok=True)
        sheet = template.copy()
        sheet["rater_id"] = rater
        sheet.to_csv(folder / f"e2_600_{rater}_filled.csv", index=False,
                     encoding="utf-8-sig")
        (folder / "RATING_CODEBOOK_v1.3.md").write_text(codebook, encoding="utf-8")
        (folder / "COLUMN_GUIDE_v1.3.md").write_text(column_guide, encoding="utf-8")
        (folder / "README_FIRST.md").write_text(readme(rater), encoding="utf-8")

        check = pd.read_csv(folder / f"e2_600_{rater}_filled.csv", encoding="utf-8-sig")
        assert check.shape == (640, 23)
        assert (check["rater_id"] == rater).all()
        assert check.iloc[:, 7:].isna().all().all()
        assert not any(p.name in {"e2_annotation_600_map.csv"} for p in folder.iterdir())
        print(f"{folder.relative_to(ROOT)}: {len(list(folder.iterdir()))} files, 640 rows")


if __name__ == "__main__":
    main()
