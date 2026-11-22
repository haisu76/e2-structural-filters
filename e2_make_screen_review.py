#!/usr/bin/env python3
"""Freeze the post-audit screen and build coder C's targeted review pack.

Screen rules (frozen 2026-11-17, recall 11/11 on the 65-pair audit):
  help_insert   rewrite adds "help*" while original has none
  attrib        rewrite changes the set of {designed,intended,claimed,
                considered,presented,meant,built} relative to original
  no_longer     rewrite adds "no longer"
  fight_drop    original has "fight", rewrite does not
  self_frame    rewrite adds "see it as / see myself as / describe myself as"
  competition   original "the competition" replaced by "competing/compete/..."

Pool: the audit-adjusted main set (11 audited-invalid pairs already removed).
Pairs already in the 65-pair audit are not re-sent.

Outputs
  e2_validation/audit_round3_coderC_screen/README_FIRST.md
  e2_validation/audit_round3_coderC_screen/HOW_TO_CODE_AUDIT.md
  e2_validation/audit_round3_coderC_screen/G1_audit_screen_pairs.csv
  e2_validation/PI_only/G1_audit_screen_key.csv
"""
import csv
import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

from e2_prepare_inclusion import load_latest, representative_pairs
from e2_apply_prior_exclusions import prior_labels

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
PKG = OUT / "audit_round3_coderC_screen"
SALT = "audit_screen_round1"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def screen(orig, rewrite):
    flags = []
    if re.search(r"\bhelp(s|ed|ing)?\b", rewrite, re.I) and not re.search(r"\bhelp", orig, re.I):
        flags.append("help_insert")
    toks = r"\b(designed|intended|claimed|considered|presented|meant|built)\b"
    if re.findall(toks, rewrite, re.I) and \
       {w.lower() for w in re.findall(toks, rewrite, re.I)} != {w.lower() for w in re.findall(toks, orig, re.I)}:
        flags.append("attrib")
    if re.search(r"no longer", rewrite, re.I) and not re.search(r"no longer", orig, re.I):
        flags.append("no_longer")
    if re.search(r"fight", orig, re.I) and not re.search(r"fight", rewrite, re.I):
        flags.append("fight_drop")
    if re.search(r"(see it as|see myself as|describe myself as)", rewrite, re.I) and \
       not re.search(r"(see it as|see myself as|describe myself as)", orig, re.I):
        flags.append("self_frame")
    if re.search(r"the competition", orig, re.I) and re.search(r"compet(ing|e|ition)", rewrite, re.I) \
       and not re.search(r"the competition", rewrite, re.I):
        flags.append("competition")
    return flags


def main():
    rows = [r for r in load(OUT / "G1_inclusion_round3_recode_final_audit_adjusted.csv")
            if r["included"] == "1"]
    reps = representative_pairs(load_latest(ROOT / "e2_generation/wave1_deepseek.jsonl"))
    prior = prior_labels()
    audited = {(r["seed_id"], r["condition"])
               for r in load(OUT / "PI_only/G1_inclusion_round3_audit_clean_key.csv")}

    picked = []
    for r in rows:
        rep = reps[(r["seed_id"], r["condition"])]
        orig = rep.get("subject_original", "") + "\n" + rep.get("original", "")
        rew = rep.get("subject_paraphrase", "") + "\n" + rep.get("paraphrase", "")
        flags = screen(orig, rew)
        if not flags or (r["seed_id"], r["condition"]) in audited:
            continue
        picked.append((r, rep, flags))

    picked.sort(key=lambda x: h(f"{SALT}|{x[0]['seed_id']}|{x[0]['condition']}"))
    sheet, key_rows = [], []
    for r, rep, flags in picked:
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
            "candidate": r["candidate"],
            "pool": "auto" if r["inclusion_basis"].startswith("auto") else "human",
            "screen_flags": ";".join(flags),
            "prior_labels": "/".join(prior.get((r["seed_id"], r["condition"]), ("", ""))) or "never_coded",
        })

    assert len({r["blind_id"] for r in sheet}) == len(sheet)
    fields = ["blind_id", "subject_original", "body_original", "subject_rewrite",
              "body_rewrite", "valid", "notes"]

    def write(path, rows, fieldnames):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write(PKG / "G1_audit_screen_pairs.csv", sheet, fields)
    write(OUT / "PI_only/G1_audit_screen_key.csv", key_rows,
          ["blind_id", "seed_id", "condition", "candidate", "pool", "screen_flags", "prior_labels"])
    shutil.copy(OUT / "audit_round3_coderC/HOW_TO_CODE_AUDIT.md", PKG / "HOW_TO_CODE_AUDIT.md")

    readme = f"""# 第二批复核任务（211 对）

你已经通过上一批 8 对校准，规则不变，**不需要再做校准**。

- 表格：`G1_audit_screen_pairs.csv`，共 {len(sheet)} 对。
- 规则：`HOW_TO_CODE_AUDIT.md`（与你上次收到的相同）。
- 这批是从已纳入样本里筛出的重点复核项，**不要推测筛选原因，也不要参考上一批的判定**；逐对独立判断。
- 填法不变：`valid` 填 1/0；判 0 时 `notes` 写 `原因代码: "原文片段" → "改写片段"`（同句）。
- 不改行数、顺序或前五列。
- 建议每次 40–50 行，分 4–5 次完成；约 3–4 小时。
- 交回文件名：`G1_audit_screen_pairs_filled.csv`。

判 1 时 `notes` 留空。判 0 却写不出同句证据时改判 1；拿不准的写 `other: 具体说明`。
"""
    (PKG / "README_FIRST.md").write_text(readme, encoding="utf-8")

    pools = Counter(k["pool"] for k in key_rows)
    flags = Counter(f for k in key_rows for f in k["screen_flags"].split(";"))
    print(f"review={len(sheet)} pools={dict(pools)} flags={dict(flags)}")
    print(f"wrote {PKG.name}/ and PI_only/G1_audit_screen_key.csv")


if __name__ == "__main__":
    main()
