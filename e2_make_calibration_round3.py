#!/usr/bin/env python3
"""Build the 11-pair targeted calibration sheet for the G1 round-3 redo.

Round 2 failed the kappa gate (kappa=.503, PABAK=.927) with 11 disagreements.
All 11 are used here so both coders can align on the two remaining boundaries
before recoding a fresh 300-pair sample.

Coder-facing sheet is blind to condition and shows no prior labels; the key
stays in PI_only/.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A_PATH = ROOT / "annotation/FILL_ME_G1_inclusion_round2_sample_300_filled.csv"
B_PATH = ROOT / "annotation/FILL_ME_G1_inclusion_round2_sample_300_coder2_filled.csv"
KEY_PATH = ROOT / "e2_validation/PI_only/G1_inclusion_round2_key.csv"
OUT_SHEET = ROOT / "e2_validation/calibration_round3/G1_calibration_11.csv"
OUT_KEY = ROOT / "e2_validation/PI_only/G1_calibration_11_key.csv"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    a_rows = load(A_PATH)
    b_by_id = {r["blind_id"]: r for r in load(B_PATH)}
    key = {r["blind_id"]: r for r in load(KEY_PATH)}

    picked = []
    for a in a_rows:
        b = b_by_id[a["blind_id"]]
        if a["valid"] != b["valid"]:
            picked.append((a, b, key[a["blind_id"]]))

    OUT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SHEET, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "subject_original", "body_original",
                    "subject_rewrite", "body_rewrite", "valid", "notes"])
        for a, _b, _k in picked:
            w.writerow([a["blind_id"], a["subject_original"], a["body_original"],
                        a["subject_rewrite"], a["body_rewrite"], "", ""])

    with open(OUT_KEY, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "seed_id", "condition", "candidate",
                    "round2_A_valid", "round2_B_valid", "round2_A_notes", "round2_B_notes"])
        for a, b, k in picked:
            w.writerow([a["blind_id"], k["seed_id"], k["condition"], k["candidate"],
                        a["valid"], b["valid"], a["notes"].strip(), b["notes"].strip()])

    print(f"wrote {OUT_SHEET} ({len(picked)} rows)")
    print(f"wrote {OUT_KEY}")


if __name__ == "__main__":
    main()
