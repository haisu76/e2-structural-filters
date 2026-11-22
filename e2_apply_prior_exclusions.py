#!/usr/bin/env python3
"""Apply PI decision A to the adjudicated Round-3 base set.

Rounds 1-2 are void for reliability, but their human labels still identify
semantic changes among the auto-included pairs. PI decision A:

  main set        exclude pairs both coders judged invalid        (both_zero)
  sensitivity set exclude pairs either coder judged invalid      (any_zero)
  archive         keep every hard-valid pair                     (none)

Input
  e2_validation/G1_inclusion_round3_recode_adjudicated_base.csv

Outputs (default names by mode)
  G1_inclusion_round3_recode_final.csv
  G1_inclusion_round3_recode_complete_pairs.csv
  G1_inclusion_round3_recode_final_sensitivity_exclude101.csv
  G1_inclusion_round3_recode_final_prereg_literal.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]

PRIOR = [
    ("annotation/FILL_ME_G1_inclusion_sample_300_filled.csv",
     "annotation/FILL_ME_G1_inclusion_sample_300_coder2_filled.csv",
     "e2_validation/PI_only/G1_inclusion_key.csv"),
    ("annotation/FILL_ME_G1_inclusion_round2_sample_300_filled.csv",
     "annotation/FILL_ME_G1_inclusion_round2_sample_300_coder2_filled.csv",
     "e2_validation/PI_only/G1_inclusion_round2_key.csv"),
]


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prior_labels():
    out = {}
    for a_path, b_path, key_path in PRIOR:
        a = {r["blind_id"]: r["valid"].strip() for r in load_csv(ROOT / a_path)}
        b = {r["blind_id"]: r["valid"].strip() for r in load_csv(ROOT / b_path)}
        for r in load_csv(ROOT / key_path):
            if r.get("sheet", "sample") != "sample":
                continue
            bid = r["blind_id"]
            if bid in a and bid in b:
                out[(r["seed_id"], r["condition"])] = (a[bid], b[bid])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(OUT / "G1_inclusion_round3_recode_adjudicated_base.csv"))
    parser.add_argument("--mode", choices=["both_zero", "any_zero", "none"], default="both_zero")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    rows = load_csv(Path(args.base))
    prior = prior_labels()

    for r in rows:
        key = (r["seed_id"], r["condition"])
        labels = prior.get(key)
        if not labels:
            continue
        if args.mode == "both_zero" and labels == ("0", "0"):
            r["included"] = "0"
            r["inclusion_basis"] = "prior_both_zero_excluded"
        elif args.mode == "any_zero" and "0" in labels:
            r["included"] = "0"
            r["inclusion_basis"] = "prior_any_zero_excluded"

    final_path = OUT / f"G1_inclusion_round3_recode_final{args.suffix}.csv"
    with open(final_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    seed_conds = defaultdict(set)
    for r in rows:
        if r["included"] == "1":
            seed_conds[r["seed_id"]].add(r["condition"])
    complete = sorted(s for s, cs in seed_conds.items() if all(c in cs for c in CONDS))
    comp_path = OUT / f"G1_inclusion_round3_recode_complete_pairs{args.suffix}.csv"
    with open(comp_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_id"])
        for s in complete:
            w.writerow([s])

    inc = Counter(r["condition"] for r in rows if r["included"] == "1")
    print(f"mode={args.mode} included={sum(1 for r in rows if r['included'] == '1')} "
          f"complete={len(complete)} per_condition={ {c: inc[c] for c in CONDS} }")
    print(f"wrote {final_path.name} and {comp_path.name}")


if __name__ == "__main__":
    main()
