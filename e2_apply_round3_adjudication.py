#!/usr/bin/env python3
"""Apply PI adjudication to the Round-3 recode inclusion set.

The agreement gate failed (kappa = .177), so the pre-registered pipeline wrote only
PROVISIONAL_* files. This script records the PI's deviation decision: disputed sample
pairs follow the adjudication table; undisputed pairs keep the two-coder rule.

Inputs
  e2_validation/PROVISIONAL_G1_inclusion_round3_recode_pairs.csv
  e2_validation/PI_only/G1_inclusion_round3_recode_adjudication.csv

Outputs
  e2_validation/G1_inclusion_round3_recode_adjudicated_base.csv
  e2_validation/G1_inclusion_round3_recode_adjudicated_base_complete_pairs.csv

The prior-round exclusion (PI decision A) is applied by
e2_apply_prior_exclusions.py, which writes the final main and sensitivity sets.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provisional", default=str(OUT / "PROVISIONAL_G1_inclusion_round3_recode_pairs.csv"))
    parser.add_argument("--adjudication", default=str(OUT / "PI_only/G1_inclusion_round3_recode_adjudication.csv"))
    parser.add_argument("--tag", default="_round3_recode")
    args = parser.parse_args()

    prov = load_csv(Path(args.provisional))
    adj = {r["blind_id"]: (int(r["PI_valid"]), r.get("source", "coder_disagreement"))
           for r in load_csv(Path(args.adjudication))}

    rows = []
    for r in prov:
        r = dict(r)
        if r["source"] == "human_sample":
            a, b = int(r["coder_A_valid"]), int(r["coder_B_valid"])
            if r["blind_id"] in adj:
                r["included"], adj_source = adj[r["blind_id"]]
                prefix = "pi_override" if adj_source == "PI_override" else "pi_adjudicated"
                basis = f"{prefix}_valid" if r["included"] == 1 else f"{prefix}_invalid"
            elif a == b == 1:
                r["included"], basis = 1, "human_both_valid"
            elif a == b == 0:
                r["included"], basis = 0, "human_both_zero"
            else:
                raise KeyError(f"disputed pair missing from adjudication: {r['blind_id']}")
        else:
            r["included"], basis = 1, "auto_hard_validation"
        r["inclusion_basis"] = basis
        rows.append(r)

    fields = list(prov[0].keys()) + ["inclusion_basis"]
    final_path = OUT / f"G1_inclusion{args.tag}_adjudicated_base.csv"
    with open(final_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    seed_conds = defaultdict(set)
    for r in rows:
        if r["included"] == 1:
            seed_conds[r["seed_id"]].add(r["condition"])
    complete = sorted(s for s, cs in seed_conds.items() if all(c in cs for c in CONDS))
    comp_path = OUT / f"G1_inclusion{args.tag}_adjudicated_base_complete_pairs.csv"
    with open(comp_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_id"])
        for s in complete:
            w.writerow([s])

    inc = Counter(r["condition"] for r in rows if r["included"] == 1)
    print(f"total={len(rows)} included={sum(1 for r in rows if r['included'] == 1)} complete={len(complete)}")
    print("sample_basis=" + str(Counter(r["inclusion_basis"] for r in rows if r["source"] == "human_sample")))
    print("per_condition=" + str({c: inc[c] for c in CONDS}))


if __name__ == "__main__":
    main()
