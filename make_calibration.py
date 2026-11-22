#!/usr/bin/env python3
"""Build a 50-item calibration subset from the fixed 400-item E1 pool.

The subset is stratified by:
  - cell (human/synthetic x spam/ham), with quotas 13/12/12/13;
  - within-cell rank bins of urgency_proxy (3 bins) and warmth_proxy (2 bins).

The proxies only spread the sample across their empirical ranges. They are not
temperature labels and must not be analysed as dependent variables.

Outputs (under --out, names controlled by --basename):
  <basename>.csv               blinded sheet: order, calib_id, subject, body
  <basename>_meta.csv          mapping to sample_id + cell + proxy bins
  <basename>_README.md         design note and rating instructions
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

CELL_QUOTAS = {
    "human_spam": 13,
    "human_ham": 12,
    "synthetic_spam": 12,
    "synthetic_ham": 13,
}

URGENCY_BINS = 3
WARMTH_BINS = 2


def rank_bins(values: list[float], n_bins: int) -> list[int]:
    """Equal-size, tie-tolerant rank bins (0 = lowest)."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (values[i], i))
    bins = [0] * n
    for rank, i in enumerate(order):
        bins[i] = min(n_bins - 1, rank * n_bins // n)
    return bins


def allocate(total: int, caps: dict[tuple[int, int], int], rng: random.Random) -> dict[tuple[int, int], int]:
    """Round-robin allocation across strata, bounded by stratum capacity."""
    keys = list(caps)
    rng.shuffle(keys)
    alloc = {k: 0 for k in keys}
    while sum(alloc.values()) < total:
        progressed = False
        for k in keys:
            if sum(alloc.values()) >= total:
                break
            if alloc[k] < caps[k]:
                alloc[k] += 1
                progressed = True
        if not progressed:
            raise RuntimeError(f"cannot allocate {total} items within stratum capacities {caps}")
    return alloc


def take_with_template_cap(pool: pd.DataFrame, n: int, cap: int | None,
                           rng: random.Random) -> list[int]:
    indices = list(pool.index)
    rng.shuffle(indices)
    counts: dict[str, int] = {}
    picked: list[int] = []
    for i in indices:
        key = pool.at[i, "template_key"]
        key = key if isinstance(key, str) and key else f"row-{i}"
        if cap is None or counts.get(key, 0) < cap:
            picked.append(i)
            counts[key] = counts.get(key, 0) + 1
            if len(picked) == n:
                break
    return picked


def sample_cell(group: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    g = group.copy()
    g["urgency_bin"] = rank_bins(g["urgency_proxy"].tolist(), URGENCY_BINS)
    g["warmth_bin"] = rank_bins(g["warmth_proxy"].tolist(), WARMTH_BINS)

    strata = {(int(ub), int(wb)): sub for (ub, wb), sub in g.groupby(["urgency_bin", "warmth_bin"])}
    alloc = allocate(n, {k: len(v) for k, v in strata.items()}, rng)

    picked: list[int] = []
    for key in sorted(strata):
        need = alloc[key]
        if need == 0:
            continue
        for cap in (1, 2, None):
            ids = take_with_template_cap(strata[key], need, cap, rng)
            if len(ids) == need:
                picked.extend(ids)
                break
        else:
            raise RuntimeError(f"stratum {key} has fewer than {need} usable items")
    return g.loc[picked].copy()


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def build_README(sheet: pd.DataFrame, meta: pd.DataFrame, args: argparse.Namespace) -> str:
    cell_counts = meta["cell"].value_counts().to_dict()
    bin_table = pd.crosstab(
        [meta["cell"], meta["urgency_bin"]], meta["warmth_bin"]
    ).reset_index()
    bin_table = bin_table.rename(columns={0: "warmth_0", 1: "warmth_1"})
    exclude_note = ""
    if args.exclude_meta:
        listed = "、".join(f"`{p.as_posix()}`" for p in args.exclude_meta)
        exclude_note = (
            f"- 已排除 {listed} 中列出的 "
            f"{args.excluded_count} 条旧校准条目，避免新旧校准样本重叠。\n"
        )
    lines = [
        f"# E1 {len(sheet)} 条校准表（{args.basename}）",
        "",
        f"从 `e1_items.csv` 的 400 条固定池中分层抽取 {len(sheet)} 条，用于标注员校准与评分细则修订。",
        f"抽取脚本：`make_calibration.py`（seed={args.seed}）。",
        exclude_note.rstrip(),
        "",
        "## 1. 设计",
        "",
        "- 4 个 cell 全覆盖：human/synthetic × spam/ham。",
        "- 每个 cell 内按 `urgency_proxy` 做 3 等分秩次、按 `warmth_proxy` 做 2 等分秩次，",
        "  共 6 个层；每个 cell 的 12–13 条先每层取 2 条，余数分给不同的层。",
        "- 代理分箱是 **cell 内相对秩次**，不是绝对温度；不同语料的代理分数不可直接比较。",
        f"- `{args.basename}.csv` 已去掉 label / provenance / corpus 并打乱顺序；",
        f"  `{args.basename}_meta.csv` 保留映射，供分析用，**不要发给标注员**。",
        "",
        "## 2. 配额",
        "",
        "| cell | n |",
        "|---|---|",
    ]
    for cell in CELL_QUOTAS:
        lines.append(f"| {cell} | {cell_counts.get(cell, 0)} |")
    lines += [
        "",
        f"合计 {len(sheet)} 条；human {int(meta['provenance'].eq('human').sum())} / "
        f"synthetic {int(meta['provenance'].eq('synthetic').sum())}；"
        f"spam {int(meta['label'].eq(1).sum())} / ham {int(meta['label'].eq(0).sum())}。",
        "",
        "## 3. 分层覆盖（cell × urgency_bin × warmth_bin）",
        "",
        md_table(bin_table),
        "",
        "## 4. 用法",
        "",
        f"评分手册与条目锚点见 `{args.codebook}`；"
        f"评分模板见 `{args.template}`。",
        "",
        f"1. 三名标注员独立给 {len(sheet)} 条打分；维度和量表与正式 400 条一致：",
        "   valence、arousal/urgency、interpersonal warmth、distance/institutionalization，",
        "   并单独标记 threat/coercion 是否存在。",
        "2. 先各自完成，再逐条比对分歧；只有预注册允许的修订才能改细则。",
        "3. 计算 ICC(2,k) 或 Krippendorff α；达到预注册阈值后再进入 400 条正式标注。",
        "4. 校准阶段的分数不并入正式分析；只用于细则与一致性检查。",
        "",
        f"校准表 `{sheet['calib_id'].iloc[0]}`–`{sheet['calib_id'].iloc[-1]}`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 50-item E1 calibration subset.")
    parser.add_argument("--items", default="samples/e1_items.csv")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--basename", default="e1_calibration_50",
                        help="output filename stem")
    parser.add_argument("--id-prefix", default="CAL",
                        help="prefix for calib_id, e.g. CAL or CAL2")
    parser.add_argument("--exclude-meta", type=Path, action="append", default=None,
                        help="meta CSV whose sample_id rows must be excluded; repeatable")
    parser.add_argument("--codebook", default="annotation/RATING_CODEBOOK.md")
    parser.add_argument("--template",
                        default="annotation/e1_calibration_50_ratings_template.csv")
    args = parser.parse_args()

    items_path = Path(args.items)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(items_path, encoding="utf-8-sig")
    missing = [c for c in ["sample_id", "subject", "body", "cell", "corpus", "provenance",
                           "label", "label_raw", "word_count", "urgency_proxy",
                           "warmth_proxy", "template_key"]
               if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in {items_path}: {missing}")

    excluded: set[str] = set()
    args.excluded_count = 0
    for meta_path in args.exclude_meta or []:
        exclude_df = pd.read_csv(meta_path, encoding="utf-8-sig")
        if "sample_id" not in exclude_df.columns:
            raise SystemExit(f"{meta_path}: missing sample_id column")
        excluded |= set(exclude_df["sample_id"].astype(str))
    if excluded:
        before = len(df)
        df = df[~df["sample_id"].astype(str).isin(excluded)].reset_index(drop=True)
        args.excluded_count = before - len(df)
        print(f"excluded {args.excluded_count} items listed in "
              f"{', '.join(str(p) for p in args.exclude_meta)}")

    unknown = set(df["cell"]) - set(CELL_QUOTAS)
    if unknown:
        raise SystemExit(f"unexpected cells: {sorted(unknown)}")
    if len(df) < sum(CELL_QUOTAS.values()):
        raise SystemExit("not enough items")

    rng = random.Random(args.seed)
    parts: list[pd.DataFrame] = []
    for cell, quota in CELL_QUOTAS.items():
        group = df[df["cell"] == cell]
        if len(group) < quota:
            raise SystemExit(f"cell {cell} has {len(group)} items, needs {quota}")
        parts.append(sample_cell(group, quota, rng))

    selected = pd.concat(parts, ignore_index=True)
    if len(selected) != sum(CELL_QUOTAS.values()):
        raise RuntimeError(f"expected 50 rows, got {len(selected)}")
    if selected["sample_id"].duplicated().any():
        raise RuntimeError("duplicate sample_id in calibration subset")
    if excluded and not set(selected["sample_id"].astype(str)).isdisjoint(excluded):
        raise RuntimeError("calibration selection overlaps the excluded sample")

    selected = selected.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    selected["calib_id"] = [f"{args.id_prefix}-{i:02d}"
                            for i in range(1, len(selected) + 1)]

    sheet = selected[["calib_id", "subject", "body"]].copy()
    sheet.insert(0, "order", range(1, len(sheet) + 1))

    meta_cols = ["calib_id", "sample_id", "cell", "corpus", "provenance", "label",
                 "label_raw", "word_count", "urgency_proxy", "warmth_proxy",
                 "urgency_bin", "warmth_bin", "template_key"]
    meta = selected[meta_cols].copy()

    sheet_path = out_dir / f"{args.basename}.csv"
    meta_path = out_dir / f"{args.basename}_meta.csv"
    readme_path = out_dir / f"{args.basename}_README.md"
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    readme_path.write_text(build_README(sheet, meta, args), encoding="utf-8")

    print(f"wrote {sheet_path} ({len(sheet)} rows)")
    print(f"wrote {meta_path}")
    print(f"wrote {readme_path}")
    print("cell counts:", meta["cell"].value_counts().to_dict())
    print("urgency bins:", meta["urgency_bin"].value_counts().sort_index().to_dict())
    print("warmth bins:", meta["warmth_bin"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
