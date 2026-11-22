#!/usr/bin/env python3
"""Build coder C's second G2 batch: reviews for frozen-rule screen hits.

Amendment 2, G2-F4 step 3: the six frozen rules (source hash prefix
cc2a1ed44de07ec7) run over every representative pair; hits go to coder C in a
separate batch, after the random audit. Rules only select review objects.

The rules are imported unchanged from e2_make_screen_review.py, so their
behaviour is identical to G1. Pairs already in the 65-pair audit are not
re-sent; audited-invalid pairs are already excluded from the pool.

Outputs
  e2_validation/audit_G2_screen/README_FIRST.md
  e2_validation/audit_G2_screen/HOW_TO_CODE_G2.md
  e2_validation/audit_G2_screen/G2_screen_pairs.csv
  e2_validation/PI_only/G2_screen_key.csv
"""
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

from e2_make_screen_review import screen
from e2_prepare_inclusion import load_latest, representative_pairs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
PKG = OUT / "audit_G2_screen"
SALT = "screen_g2"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    rows = [r for r in load(OUT / "G2_inclusion_audit_adjusted.csv") if r["included"] == "1"]
    reps = representative_pairs(load_latest(ROOT / "e2_generation/wave1_qwen.jsonl"))
    key = load(OUT / "PI_only/G2_audit_65_key.csv")
    audited = {(r["seed_id"], r["condition"]) for r in key}
    audit_ids = {r["blind_id"] for r in key}

    picked = []
    for r in rows:
        rep = reps[(r["seed_id"], r["condition"])]
        orig = rep.get("subject_original", "") + "\n" + rep.get("original", "")
        rew = rep.get("subject_rewrite", "") + "\n" + rep.get("paraphrase", "")
        flags = screen(orig, rew)
        if flags and (r["seed_id"], r["condition"]) not in audited:
            picked.append((r, rep, flags))

    picked.sort(key=lambda x: h(f"{SALT}|{x[0]['seed_id']}|{x[0]['condition']}"))
    sheet, key_rows = [], []
    for r, rep, flags in picked:
        bid = h(f"{SALT}|{r['seed_id']}|{r['condition']}")[:12]
        if bid in audit_ids or any(k["blind_id"] == bid for k in key_rows):
            bid = h(f"{SALT}|{r['seed_id']}|{r['condition']}|x")[:12]
        sheet.append({
            "blind_id": bid,
            "subject_original": rep.get("subject_original", ""),
            "body_original": rep.get("original", ""),
            "subject_rewrite": rep.get("subject_rewrite", ""),
            "body_rewrite": rep.get("paraphrase", ""),
            "valid": "",
            "notes": "",
        })
        key_rows.append({
            "blind_id": bid, "seed_id": r["seed_id"], "condition": r["condition"],
            "candidate": r["candidate"], "screen_flags": ";".join(flags),
        })

    assert len({r["blind_id"] for r in sheet}) == len(sheet)
    fields = ["blind_id", "subject_original", "body_original", "subject_rewrite",
              "body_rewrite", "valid", "notes"]

    def write(path, data, fieldnames):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)

    write(PKG / "G2_screen_pairs.csv", sheet, fields)
    write(OUT / "PI_only/G2_screen_key.csv", key_rows,
          ["blind_id", "seed_id", "condition", "candidate", "screen_flags"])
    shutil.copy(OUT / "audit_G2/HOW_TO_CODE_G2.md", PKG / "HOW_TO_CODE_G2.md")

    readme = f"""# G2 第二批复核任务（{len(sheet)} 对）

你已经完成上一批 65 对随机审计，规则不变，**不需要再做校准**。

- 表格：`G2_screen_pairs.csv`，共 {len(sheet)} 对。
- 规则：`HOW_TO_CODE_G2.md`（与你上次收到的相同）。
- 这批是从已纳入样本里筛出的重点复核项，**不要推测筛选原因，也不要参考上一批的判定**；逐对独立判断。
- 填法不变：`valid` 填 1/0；判 0 时 `notes` 写 `原因代码: "原文片段" → "改写片段"`（同句）。判 1 时 `notes` 留空。
- 不改行数、顺序或前五列。
- 建议每次 20–30 行，分 2–3 次完成。
- 交回文件名：`G2_screen_pairs_filled.csv`（UTF-8）。

判 0 却写不出同句证据时改判 1；拿不准的写 `other: 具体说明`。
"""
    (PKG / "README_FIRST.md").write_text(readme, encoding="utf-8")

    flags = Counter(f for k in key_rows for f in k["screen_flags"].split(";"))
    print(f"review={len(sheet)} flags={dict(flags)}")
    print(f"wrote {PKG.name}/ and PI_only/G2_screen_key.csv")


if __name__ == "__main__":
    main()
