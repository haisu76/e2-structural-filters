#!/usr/bin/env python3
"""Build the 20-pair calibration sheet for the G1 inclusion coding redo.

Round 1 failed the kappa gate, so its 300 pairs are void as formal data. They
are still useful as calibration material: the 79 disagreements concentrate on
exactly the two boundaries that broke agreement (urgency added in the subject;
urgency softened in the body). Select a fixed, docstring-documented mix:

  less_urgent  A=1/B=0   7   (B penalised intended softening)
  more_urgent  A=0/B=1   6   (A called added urgency a new request)
  colder       disagree  3
  neutral      disagree  2
  warmer       disagree  1
  both=0       any       1   (clear invalid anchor)

Coder-facing sheet is blind to condition; the key stays in PI_only/.
"""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A_PATH = ROOT / "annotation/FILL_ME_G1_inclusion_sample_300_filled.csv"
B_PATH = ROOT / "annotation/FILL_ME_G1_inclusion_sample_300_coder2_filled.csv"
KEY_PATH = ROOT / "e2_validation/PI_only/G1_inclusion_key.csv"
CAL_DIR = ROOT / "e2_validation/calibration"
PI_DIR = ROOT / "e2_validation/PI_only"

QUOTA = {
    ("less_urgent", "10"): 7,
    ("more_urgent", "01"): 6,
    ("colder", "10"): 3,
    ("neutral", "10"): 2,
    ("warmer", "10"): 1,
    ("*", "both0"): 1,
}


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def direction(a, b):
    if a == b == "0":
        return "both0"
    return f"{a}{b}"


def main():
    a_rows = load(A_PATH)
    b_rows = load(B_PATH)
    key = {r["blind_id"]: r for r in load(KEY_PATH)}

    pool = []
    for a, b in zip(a_rows, b_rows):
        bid = a["blind_id"]
        pool.append({
            "blind_id": bid,
            "seed_id": key[bid]["seed_id"],
            "condition": key[bid]["condition"],
            "candidate": key[bid]["candidate"],
            "a": a["valid"],
            "b": b["valid"],
            "a_notes": a["notes"].strip(),
            "b_notes": b["notes"].strip(),
            "subject_original": a["subject_original"],
            "body_original": a["body_original"],
            "subject_rewrite": a["subject_rewrite"],
            "body_rewrite": a["body_rewrite"],
        })

    picked = []
    used = set()

    def take(condition, direction_key, n):
        candidates = [r for r in pool if r["blind_id"] not in used
                      and (condition == "*" or r["condition"] == condition)
                      and (direction_key == "*" or direction(r["a"], r["b"]) == direction_key)]
        candidates.sort(key=lambda r: h(r["blind_id"]))
        for r in candidates[:n]:
            picked.append(r)
            used.add(r["blind_id"])

    for (cond, direction_key), n in QUOTA.items():
        take(cond, direction_key, n)

    picked.sort(key=lambda r: h("calibration|" + r["blind_id"]))

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    sheet = CAL_DIR / "G1_calibration_20.csv"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "subject_original", "body_original",
                    "subject_rewrite", "body_rewrite", "valid", "notes"])
        for r in picked:
            w.writerow([r["blind_id"], r["subject_original"], r["body_original"],
                        r["subject_rewrite"], r["body_rewrite"], "", ""])

    key_path = PI_DIR / "G1_calibration_20_key.csv"
    with open(key_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "seed_id", "condition", "candidate",
                    "coder_A_valid", "coder_B_valid", "coder_A_notes", "coder_B_notes"])
        for r in picked:
            w.writerow([r["blind_id"], r["seed_id"], r["condition"], r["candidate"],
                        r["a"], r["b"], r["a_notes"], r["b_notes"]])

    print(f"wrote {sheet} ({len(picked)} rows)")
    print(f"wrote {key_path}")
    from collections import Counter
    print("by condition:", dict(Counter(r["condition"] for r in picked)))
    print("by direction:", dict(Counter(direction(r["a"], r["b"]) for r in picked)))


if __name__ == "__main__":
    main()
