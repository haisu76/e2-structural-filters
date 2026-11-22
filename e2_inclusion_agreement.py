#!/usr/bin/env python3
"""Merge the two G1 human inclusion codings, compute agreement, and build the inclusion set.

Inputs
  annotation/FILL_ME_G1_inclusion_sample_300_filled.csv
  annotation/FILL_ME_G1_inclusion_sample_300_coder2_filled.csv
  e2_validation/PI_only/G1_inclusion_key.csv
  e2_generation/wave1_deepseek.jsonl   (for candidate hashes / validator flags)

Outputs (e2_validation/)
  G1_inclusion_agreement.md
  G1_inclusion_agreement_2x2.csv
  G1_inclusion_disagreements.csv
  G1_inclusion_final.csv
  G1_inclusion_complete_pairs.csv
"""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from e2_prepare_inclusion import load_latest, representative_pairs

ROOT = Path(__file__).resolve().parent
A_PATH = ROOT / "annotation/FILL_ME_G1_inclusion_sample_300_filled.csv"
B_PATH = ROOT / "annotation/FILL_ME_G1_inclusion_sample_300_coder2_filled.csv"
KEY_PATH = ROOT / "e2_validation/PI_only/G1_inclusion_key.csv"
LOG_PATH = ROOT / "e2_generation/wave1_deepseek.jsonl"
OUT = ROOT / "e2_validation"


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_valid(value):
    v = (value or "").strip()
    if v not in {"0", "1"}:
        raise ValueError(f"unexpected valid value: {value!r}")
    return int(v)


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    pabak = 2 * po - 1
    return po, pe, kappa, pabak


def sha12(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coder-a", default=str(A_PATH))
    parser.add_argument("--coder-b", default=str(B_PATH))
    parser.add_argument("--key", default=str(KEY_PATH))
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    args = parser.parse_args()
    tag = args.tag

    a_rows = load_csv(Path(args.coder_a))
    b_rows = load_csv(Path(args.coder_b))
    key_rows = load_csv(Path(args.key))

    # ---- structural checks -------------------------------------------------
    ids_a = [r["blind_id"].strip() for r in a_rows]
    ids_b = [r["blind_id"].strip() for r in b_rows]
    assert len(a_rows) == len(b_rows) == 300, "expected 300 rows per coder"
    assert len(set(ids_a)) == 300 and len(set(ids_b)) == 300, "duplicate blind_id"
    assert ids_a == ids_b, "coder files are not row-aligned"

    key = {r["blind_id"]: r for r in key_rows}
    sample_ids = [r["blind_id"] for r in key_rows if r["sheet"] == "sample"]
    assert set(sample_ids) == set(ids_a), "sample IDs do not match the key"
    assert len(sample_ids) == 300

    va = [as_valid(r["valid"]) for r in a_rows]
    vb = [as_valid(r["valid"]) for r in b_rows]

    # ---- agreement ---------------------------------------------------------
    po, pe, kappa, pabak = cohen_kappa(va, vb)
    table = Counter(zip(va, vb))
    n11, n10, n01, n00 = table[(1, 1)], table[(1, 0)], table[(0, 1)], table[(0, 0)]
    both_valid = n11
    any_valid = n11 + n10 + n01

    # per-condition descriptive breakdown
    per_cond = defaultdict(lambda: {"n": 0, "both1": 0, "disagree": 0, "a1": 0, "b1": 0})
    for i, bid in enumerate(ids_a):
        cond = key[bid]["condition"]
        c = per_cond[cond]
        c["n"] += 1
        c["a1"] += va[i]
        c["b1"] += vb[i]
        c["both1"] += int(va[i] == 1 and vb[i] == 1)
        c["disagree"] += int(va[i] != vb[i])

    # ---- disagreements -----------------------------------------------------
    dis_path = OUT / f"G1_inclusion{tag}_disagreements.csv"
    with open(dis_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "seed_id", "condition", "candidate",
                    "coder_A_valid", "coder_A_notes", "coder_B_valid", "coder_B_notes",
                    "subject_original", "subject_rewrite"])
        for i, bid in enumerate(ids_a):
            if va[i] == vb[i]:
                continue
            k = key[bid]
            w.writerow([bid, k["seed_id"], k["condition"], k["candidate"],
                        va[i], a_rows[i]["notes"].strip(), vb[i], b_rows[i]["notes"].strip(),
                        a_rows[i]["subject_original"], a_rows[i]["subject_rewrite"]])

    # ---- provisional inclusion set ----------------------------------------
    # Representative pair per (seed, condition) = first validator-pass candidate by
    # SHA256(seed|candidate), identical to e2_prepare_inclusion.py.
    reps = representative_pairs(load_latest(LOG_PATH))

    coder_by_bid = {bid: (va[i], vb[i], a_rows[i]["notes"].strip(), b_rows[i]["notes"].strip())
                    for i, bid in enumerate(ids_a)}
    sample_bid = {(key[b]["seed_id"], key[b]["condition"]): b for b in sample_ids}

    final_rows = []
    for (seed, cond), rec in sorted(reps.items()):
        bid = sample_bid.get((seed, cond), "")
        if bid:
            ca, cb, na, nb = coder_by_bid[bid]
            included = int(ca == 1 and cb == 1)
            source = "human_sample"
        else:
            ca = cb = ""
            na = nb = ""
            included = 1
            source = "auto_hard_validation"
        final_rows.append({
            "seed_id": seed, "condition": cond, "candidate": rec["candidate"],
            "blind_id": bid, "source": source,
            "coder_A_valid": ca, "coder_B_valid": cb,
            "coder_A_notes": na, "coder_B_notes": nb,
            "included": included,
        })

    # The preregistration says a failed agreement gate voids the round, so the
    # inclusion set is only written as final when the gate passes.
    gate_passed = kappa >= 0.60
    final_path = OUT / (f"G1_inclusion{tag}_final.csv" if gate_passed
                        else f"PROVISIONAL_G1_inclusion{tag}_pairs.csv")
    with open(final_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        w.writeheader()
        w.writerows(final_rows)

    # ---- complete pairs ----------------------------------------------------
    conds = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
    seed_conds = defaultdict(set)
    for r in final_rows:
        if r["included"]:
            seed_conds[r["seed_id"]].add(r["condition"])
    complete = sorted(s for s, cs in seed_conds.items() if all(c in cs for c in conds))
    per_cond_included = Counter(r["condition"] for r in final_rows if r["included"])

    comp_path = OUT / (f"G1_inclusion{tag}_complete_pairs.csv" if gate_passed
                       else f"PROVISIONAL_G1{tag}_complete_pairs.csv")
    with open(comp_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_id"])
        for s in complete:
            w.writerow([s])

    # ---- 2x2 csv -----------------------------------------------------------
    with open(OUT / f"G1_inclusion{tag}_agreement_2x2.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "coder_B_valid=1", "coder_B_valid=0"])
        w.writerow(["coder_A_valid=1", n11, n10])
        w.writerow(["coder_A_valid=0", n01, n00])

    # ---- markdown report ---------------------------------------------------
    rate_a = sum(va) / len(va)
    rate_b = sum(vb) / len(vb)
    gate = "PASS" if kappa >= 0.60 else "FAIL"
    lines = [
        "# G1 纳入编码：一致性合并结果",
        "",
        "- 样本：300 对（每条件 60），两名编码者独立盲判",
        f"- 合并键：`blind_id`（与 `{Path(args.key).name}` 完全匹配）",
        "",
        "## 结构校验",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 编码者 A 行数 / 唯一 ID | 300 / 300 |",
        f"| 编码者 B 行数 / 唯一 ID | 300 / 300 |",
        f"| 两份文件行序一致 | 是 |",
        f"| 与 key 的 sample 集合一致 | 是 |",
        f"| valid 取值仅含 0/1 | 是 |",
        "",
        "## 2×2 列联表",
        "",
        "| | B = 1 | B = 0 | 合计 |",
        "|---|---|---|---|",
        f"| **A = 1** | {n11} | {n10} | {n11 + n10} |",
        f"| **A = 0** | {n01} | {n00} | {n01 + n00} |",
        f"| 合计 | {n11 + n01} | {n10 + n00} | 300 |",
        "",
        f"- 观察一致率 Po = **{po:.3f}**",
        f"- 期望一致率 Pe = **{pe:.3f}**",
        f"- Cohen's κ = **{kappa:.3f}**",
        f"- PABAK = **{pabak:.3f}**",
        f"- 预注册门槛 κ ≥ .60：**{gate}**",
        "",
        "## 判定分布",
        "",
        f"- 编码者 A：valid=1 {sum(va)} / 300（{rate_a:.1%}）",
        f"- 编码者 B：valid=1 {sum(vb)} / 300（{rate_b:.1%}）",
        f"- 两人均判 1：{both_valid}",
        f"- 至少一人判 1：{any_valid}",
        f"- 分歧：{n10 + n01}（A=1/B=0 {n10}；A=0/B=1 {n01}）",
        "",
        "## 分条件描述（本表仅 PI 可见，编码者未接触条件标签）",
        "",
        "| 条件 | n | A valid=1 | B valid=1 | 均判 1 | 分歧 |",
        "|---|---|---|---|---|---|",
    ]
    for c in conds:
        d = per_cond[c]
        lines.append(f"| {c} | {d['n']} | {d['a1']} | {d['b1']} | {d['both1']} | {d['disagree']} |")
    lines += [
        "",
        "## 预注册门槛判定",
        "",
    ]
    if gate_passed:
        lines += [
            f"κ = {kappa:.3f} ≥ .60，**通过**。进入 PI 裁决与正式纳入。",
            "",
            "## 纳入集与完整配对",
            "",
            f"- 代表性配对总数（去重后每 seed×condition 一条）：{len(final_rows)}",
            f"- 人工样本对：300；其中两人均判 1 纳入 {both_valid}，排除 {300 - both_valid}",
            f"- 其余自动纳入（通过硬校验）：{len(final_rows) - 300}",
            f"- 当前纳入总数：{sum(per_cond_included.values())}",
            "",
            "| 条件 | 纳入条数 |",
            "|---|---|",
        ]
        for c in conds:
            lines.append(f"| {c} | {per_cond_included[c]} |")
        lines += [
            "",
            f"- **完整配对（5 条件均有纳入）：{len(complete)}**",
            "",
            "## 下一步",
            "",
            "1. PI 逐条裁决分歧（`G1_inclusion_disagreements.csv`），记录理由。",
            "2. 发放 `G1_inclusion_audit_10pct.csv`（65 对）做 10% 复核；",
            "   复核发现语义变化的配对从 `G1_inclusion_final.csv` 中排除并报告比例。",
            "3. 复核完成后，`G1_inclusion_final.csv` 与完整配对表即为 E2 主分析输入。",
            "",
        ]
    else:
        lines += [
            f"κ = {kappa:.3f} < .60，**未通过**。按预注册第 4.2 节：",
            "",
            "> 停止，修订编码规则后重新抽样编码（本轮结果作废）。",
            "",
            "本轮 300 对**不得**用于正式纳入。以下仅为诊断性重算，",
            f"写入 `PROVISIONAL_G1_inclusion{tag}_pairs.csv` 与 `PROVISIONAL_G1{tag}_complete_pairs.csv`，",
            "不得作为分析输入。",
            "",
            f"- 代表性配对总数：{len(final_rows)}",
            f"- 两人均判 1：{both_valid}；至少一人判 1：{any_valid}",
            f"- 若忽略门槛，纳入总数：{sum(per_cond_included.values())}，完整配对：{len(complete)}",
            "",
            "| 条件 | 诊断性纳入条数 |",
            "|---|---|",
        ]
        for c in conds:
            lines.append(f"| {c} | {per_cond_included[c]} |")
        top = sorted(((c, d["disagree"]) for c, d in per_cond.items()),
                     key=lambda x: -x[1])
        top_txt = "、".join(f"{c} {n}" for c, n in top[:2] if n)
        po_needed = 0.60 * (1 - pe) + pe
        max_disagree = int((1 - po_needed) * len(va))
        union_invalid = n10 + n01 + n00
        lines += [
            "",
            "## 失败诊断",
            "",
            f"- Po = {po:.3f}，Pe = {pe:.3f}，κ = {kappa:.3f}，PABAK = {pabak:.3f}。",
            f"- 阳性率：A {rate_a:.1%}、B {rate_b:.1%}；分歧 {n10 + n01} 条，方向 A=1/B=0 {n10}、A=0/B=1 {n01}。",
            f"- 分歧最多的条件：{top_txt or '无'}。",
            f"- 在至少一人判 0 的 {union_invalid} 对中，两人都判 0 的只有 {n00} 对，"
            f"其余 {n10 + n01} 对为分歧。",
            "  这是按结果选样的事后描述，不能当作独立可靠性估计，但说明边界案例确实难判。",
        ]
        if pabak >= 0.60:
            lines += [
                "",
                f"**κ 低主要受基础率压缩，但不全是伪影。** PABAK = {pabak:.3f} ≥ .60，",
                "说明整体一致性很高；两人几乎都判 1，把 κ 压低。",
                f"在当前边际分布下，κ 要达到 .60 需要 Po ≥ {po_needed:.3f}，",
                f"即最多 {max_disagree} 条分歧（目前 {n10 + n01} 条）。",
                "但上一条也显示：争议集中在少数边界案例，这些案例本身确实难判。",
                "",
                "可选的 PI 决策（需记录，因为预注册没有预设此分支）：",
                "",
                "1. **记录偏离并继续**：以 Po/PABAK 作为可靠性证据，PI 逐条裁决分歧，",
                "   论文中同时报告 κ 与基础率解释；不再重做。",
                "2. **补充平衡可靠性样本**：保留本轮纳入判断，另抽一组包含全部无效对的",
                "   平衡样本重判，用其估计不受基础率压缩的 κ；作为补充证据。",
                "3. **按字面重做第三轮**：修订剩余边界后重新抽样 300 对；",
                "   但在 ~96% 阳性率下仍可能因同一原因达不到 κ ≥ .60。",
            ]
            next_steps = [
                "1. PI 在以上三个选项中决定，并写入偏差记录。",
                f"2. 若选择继续：PI 逐条裁决 `G1_inclusion{tag}_disagreements.csv`，",
                "   记录理由；按预注册“两人都判 1 才纳入”生成最终纳入集。",
                "3. 主编码处置完成后，安排 audit。",
            ]
        else:
            lines += [
                "",
                "**PABAK 也低于 .60，属于实质性标准分歧，不只是基础率问题。**",
                "应先修订编码说明并做校准，再重新抽样编码。",
            ]
            next_steps = [
                "1. 用分歧行做校准，逐条对齐判定边界。",
                "2. 修订编码说明并给每个原因代码补锚点。",
                "3. 重新抽样 300 对，两人重判；本轮结果作废。",
                "4. 新轮 κ ≥ .60 后再进入 PI 裁决与正式纳入。",
            ]
        lines += ["", "## 下一步", ""] + next_steps + [""]
    (OUT / f"G1_inclusion{tag}_agreement.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Po={po:.4f} Pe={pe:.4f} kappa={kappa:.4f} PABAK={pabak:.4f} gate={gate}")
    print(f"2x2: n11={n11} n10={n10} n01={n01} n00={n00}")
    print(f"disagreements={n10 + n01} -> {dis_path}")
    print(f"final pairs={len(final_rows)} included={sum(r['included'] for r in final_rows)}")
    print("per condition included:", dict(per_cond_included))
    print(f"complete pairs={len(complete)} -> {comp_path}")


if __name__ == "__main__":
    main()
