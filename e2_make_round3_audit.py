#!/usr/bin/env python3
"""Draw a clean 10%-per-condition audit sample for the G1 Round-3 inclusion set.

The old per-round audit sheets overlap heavily with later-round samples (80/102),
so they are not independent. This script draws fresh pairs from the auto-included
pairs in the final main set, using a new hash salt.

Inputs
  e2_validation/G1_inclusion_round3_recode_final.csv
  e2_generation/wave1_deepseek.jsonl

Outputs
  e2_validation/PI_only/G1_inclusion_round3_audit_clean.csv
  e2_validation/PI_only/G1_inclusion_round3_audit_clean_key.csv
  e2_validation/audit_round3/G1_inclusion_round3_audit_clean.csv
  e2_validation/audit_round3/HOW_TO_CODE_v3.1.md
  e2_validation/audit_round3/README_FIRST.md
"""
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from e2_prepare_inclusion import load_latest, representative_pairs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
SALT = "audit_g1_round3"
FRAC = 0.10


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prior_tag(seed, cond):
    checks = [
        ("R1", "annotation/FILL_ME_G1_inclusion_sample_300_filled.csv",
         "annotation/FILL_ME_G1_inclusion_sample_300_coder2_filled.csv",
         "e2_validation/PI_only/G1_inclusion_key.csv"),
        ("R2", "annotation/FILL_ME_G1_inclusion_round2_sample_300_filled.csv",
         "annotation/FILL_ME_G1_inclusion_round2_sample_300_coder2_filled.csv",
         "e2_validation/PI_only/G1_inclusion_round2_key.csv"),
    ]
    for tag, a_path, b_path, key_path in checks:
        a = {r["blind_id"]: r["valid"].strip() for r in load_csv(ROOT / a_path)}
        b = {r["blind_id"]: r["valid"].strip() for r in load_csv(ROOT / b_path)}
        for r in load_csv(ROOT / key_path):
            if r.get("sheet", "sample") == "sample" and (r["seed_id"], r["condition"]) == (seed, cond):
                bid = r["blind_id"]
                if bid in a and bid in b:
                    return f"{tag}:{a[bid]}/{b[bid]}"
    return "never_coded"


def main():
    rows = load_csv(OUT / "G1_inclusion_round3_recode_final.csv")
    pool = [r for r in rows if r["inclusion_basis"] == "auto_hard_validation"]
    reps = representative_pairs(load_latest(ROOT / "e2_generation/wave1_deepseek.jsonl"))

    by_cond = defaultdict(list)
    for r in pool:
        by_cond[r["condition"]].append(r)

    picked = []
    for cond in CONDS:
        keys = sorted(by_cond[cond], key=lambda r: h(f"{SALT}|{r['seed_id']}|{r['condition']}"))
        n = max(1, round(len(keys) * FRAC))
        picked.extend(keys[:n])

    picked.sort(key=lambda r: h(f"{SALT}|{r['seed_id']}|{r['condition']}"))
    sheet, key_rows = [], []
    for r in picked:
        key = (r["seed_id"], r["condition"])
        rep = reps[key]
        bid = h(f"{SALT}|{r['seed_id']}|{r['condition']}")[:12]
        sheet.append({
            "blind_id": bid,
            "subject_original": rep.get("subject_original", ""),
            "body_original": rep.get("original", ""),
            "subject_rewrite": rep.get("subject_paraphrase", ""),
            "body_rewrite": rep.get("paraphrase", ""),
            "valid": "",
            "notes": "",
        })
        key_rows.append({
            "blind_id": bid, "seed_id": r["seed_id"], "condition": r["condition"],
            "candidate": r["candidate"], "prior_labels": prior_tag(*key),
        })

    assert len({r["blind_id"] for r in sheet}) == len(sheet)
    fields = ["blind_id", "subject_original", "body_original", "subject_rewrite",
              "body_rewrite", "valid", "notes"]

    def write_rows(path, rows, fieldnames):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_rows(OUT / "PI_only/G1_inclusion_round3_audit_clean.csv", sheet, fields)
    write_rows(OUT / "PI_only/G1_inclusion_round3_audit_clean_key.csv", key_rows,
               ["blind_id", "seed_id", "condition", "candidate", "prior_labels"])
    write_rows(OUT / "audit_round3/G1_inclusion_round3_audit_clean.csv", sheet, fields)
    shutil.copy(ROOT / "e2_validation/coder_A_round3_recode/HOW_TO_CODE_v3.1.md",
                OUT / "audit_round3/HOW_TO_CODE_v3.1.md")

    counts = Counter(r["condition"] for r in picked)
    readme = f"""# G1 纳入复核（audit）任务

- 表格：`G1_inclusion_round3_audit_clean.csv`，共 {len(sheet)} 对（每条件约 10%）。
- 规则：`HOW_TO_CODE_v3.1.md`。判断改写是否保持内容与恶意性不变、是否有新增请求/事实。
- 填法：`valid` 填 1 或 0；判 0 时 `notes` 写 `原因代码: "原文片段" → "改写片段"`，片段必须在同一句内。
- 不改行数、顺序或前五列。
- 不要打开 `PI_only/`、任何 `*_key.csv` 或其他编码者文件。
- 交回文件名：`G1_inclusion_round3_audit_clean_filled.csv`。
"""
    (OUT / "audit_round3/README_FIRST.md").write_text(readme, encoding="utf-8")

    print(f"pool={len(pool)} audit={len(sheet)} per_condition={ {c: counts[c] for c in CONDS} }")
    print("wrote PI_only/G1_inclusion_round3_audit_clean.csv (+key) and audit_round3/")


if __name__ == "__main__":
    main()
