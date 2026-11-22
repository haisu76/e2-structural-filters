#!/usr/bin/env python3
"""Build the G1 analysis dataset from the final inclusion set.

Joins the screen-adjusted inclusion table (857 included pairs / 115 complete
seeds) with the representative rewrite for each (seed, condition): the first
validator-pass candidate by SHA256(seed|candidate), per prereg section 4.3.

Output
  e2_analysis/e2_pairs_G1.csv
"""
import argparse
import csv
from pathlib import Path

from e2_prepare_inclusion import load_latest, representative_pairs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_analysis"
FINAL = ROOT / "e2_validation/G1_inclusion_round3_recode_final_audit_adjusted_screen_adjusted.csv"
COMPLETE = ROOT / "e2_validation/G1_inclusion_round3_recode_final_audit_adjusted_screen_adjusted_complete_pairs.csv"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=str(FINAL))
    ap.add_argument("--complete", default=str(COMPLETE))
    ap.add_argument("--out", default=str(OUT / "e2_pairs_G1.csv"))
    ap.add_argument("--expect-pairs", type=int, default=857)
    ap.add_argument("--expect-complete", type=int, default=115)
    ap.add_argument("--gen", default=str(ROOT / "e2_generation/wave1_deepseek.jsonl"))
    args = ap.parse_args()

    reps = representative_pairs(load_latest(Path(args.gen)))
    complete = {r["seed_id"] for r in load(args.complete)}
    rows = [r for r in load(args.final) if r["included"] == "1"]

    out, mismatched = [], []
    for r in rows:
        key = (r["seed_id"], r["condition"])
        rep = reps[key]
        if str(rep["candidate"]) != str(r["candidate"]):
            mismatched.append((key, r["candidate"], rep["candidate"]))
        out.append({
            "seed_id": r["seed_id"],
            "condition": r["condition"],
            "candidate": r["candidate"],
            "complete_pair": int(r["seed_id"] in complete),
            "inclusion_basis": r["inclusion_basis"],
            "subject_original": rep["subject_original"],
            "body_original": rep["original"],
            "subject_rewrite": rep["subject_paraphrase"],
            "body_rewrite": rep["paraphrase"],
            "length_ratio": rep.get("length_ratio", ""),
            "validator_warnings": rep.get("validator_warnings", ""),
        })

    if mismatched:
        raise SystemExit(f"candidate mismatch between inclusion set and representative: {mismatched[:5]}")

    assert len(out) == len({(r["seed_id"], r["condition"]) for r in out}), "duplicate pair"
    assert len(out) == args.expect_pairs, len(out)
    for seed in complete:
        conds = {r["condition"] for r in out if r["seed_id"] == seed}
        assert len(conds) == 5, (seed, conds)
    assert sum(1 for s in complete if any(r["seed_id"] == s for r in out)) == args.expect_complete

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    inc_seeds = {r["seed_id"] for r in out}
    print(f"rows={len(out)} included_seeds={len(inc_seeds)} complete_seeds={len(complete)}")
    print(f"conditions={ {c: sum(1 for r in out if r['condition'] == c) for c in ['neutral','warmer','colder','more_urgent','less_urgent']} }")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
