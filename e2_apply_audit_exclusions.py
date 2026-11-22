#!/usr/bin/env python3
"""Apply the 65-pair audit exclusions to a Round-3 inclusion set.

Reads coder C's filled audit sheet + the PI key, flips `included` to 0 for
audited-invalid pairs, and regenerates the complete-pairs list.

Usage (per variant):
  python e2_apply_audit_exclusions.py \
    --final e2_validation/G1_inclusion_round3_recode_final.csv \
    --suffix _audit_adjusted

  # sensitivity / prereg-literal variants use their own --final and suffix
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=str(OUT / "G1_inclusion_round3_recode_final.csv"))
    ap.add_argument("--filled", default=str(OUT / "audit_round3_coderC/G1_audit_65_pairs_filled.csv"))
    ap.add_argument("--key", default=str(OUT / "PI_only/G1_inclusion_round3_audit_clean_key.csv"))
    ap.add_argument("--verdicts", default=str(OUT / "PI_only/G1_inclusion_round3_audit_verdicts.csv"))
    ap.add_argument("--suffix", default="_audit_adjusted")
    ap.add_argument("--basis", default="audit_invalid_excluded")
    args = ap.parse_args()

    key = {r["blind_id"]: r for r in load(Path(args.key))}
    verdicts_path = Path(args.verdicts)
    if verdicts_path.exists():
        verdicts = {r["blind_id"]: r["PI_valid"].strip() for r in load(verdicts_path)}
        print(f"using PI verdicts: {verdicts_path.name}")
    else:
        verdicts = {r["blind_id"]: r["valid"].strip() for r in load(Path(args.filled))}
    invalid = {(key[b]["seed_id"], key[b]["condition"])
               for b, v in verdicts.items() if v == "0"}

    rows = load(Path(args.final))
    hit = 0
    for r in rows:
        if r["included"] == "1" and (r["seed_id"], r["condition"]) in invalid:
            r["included"] = "0"
            r["inclusion_basis"] = args.basis
            hit += 1
    if hit > len(invalid):
        raise SystemExit(f"flipped {hit} > flagged {len(invalid)}; check inclusion set")
    already = len(invalid) - hit
    print(f"audit-invalid flagged={len(invalid)} newly excluded={hit} already excluded={already}")

    final_path = OUT / f"{Path(args.final).stem}{args.suffix}.csv"
    with open(final_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    seed_conds = defaultdict(set)
    for r in rows:
        if r["included"] == "1":
            seed_conds[r["seed_id"]].add(r["condition"])
    complete = sorted(s for s, cs in seed_conds.items() if all(c in cs for c in CONDS))
    comp_path = OUT / f"{final_path.stem}_complete_pairs.csv"
    with open(comp_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_id"])
        for s in complete:
            w.writerow([s])

    inc = Counter(r["condition"] for r in rows if r["included"] == "1")
    print(f"input={Path(args.final).name} excluded={hit} included={sum(inc.values())} "
          f"complete={len(complete)} per_condition={ {c: inc[c] for c in CONDS} }")
    print(f"wrote {final_path.name} and {comp_path.name}")


if __name__ == "__main__":
    main()
