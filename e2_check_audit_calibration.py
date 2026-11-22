#!/usr/bin/env python3
"""Check coder C's 8-pair audit calibration sheet against the PI key.

Usage:
  python e2_check_audit_calibration.py \
    --filled annotation/CALIBRATION_G1_audit_08_filled.csv \
    --key e2_validation/PI_only/G1_calibration_audit_08_key.csv

Gate: at least 7/8 correct. Below that, discuss the missed rules before
releasing the 65-pair audit sheet.
"""
import argparse
import csv
import sys


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filled", default="annotation/CALIBRATION_G1_audit_08_filled.csv")
    ap.add_argument("--key", default="e2_validation/PI_only/G1_calibration_audit_08_key.csv")
    args = ap.parse_args()

    filled = load(args.filled)
    key = {r["blind_id"]: r for r in load(args.key)}
    if len(filled) != len(key):
        sys.exit(f"row count mismatch: filled={len(filled)} key={len(key)}")
    if {r["blind_id"] for r in filled} != set(key):
        sys.exit("blind_id set mismatch")

    bad_values = [r["blind_id"] for r in filled if r["valid"].strip() not in {"0", "1"}]
    if bad_values:
        sys.exit(f"valid must be 0/1, bad rows: {bad_values}")

    correct = 0
    print(f"{'id':6} {'C':>2} {'key':>3}  rule")
    for r in filled:
        bid = r["blind_id"]
        exp = key[bid]["expected_valid"]
        got = r["valid"].strip()
        ok = got == exp
        correct += ok
        mark = "OK" if ok else "MISS"
        print(f"{bid:6} {got:>2} {exp:>3}  {mark}  {key[bid]['rule']}")
        if not ok and r["notes"].strip():
            print(f"       notes: {r['notes'].strip()[:120]}")

    print(f"\ncorrect={correct}/{len(filled)}  gate(>=7/8): {'PASS' if correct >= 7 else 'FAIL'}")
    sys.exit(0 if correct >= 7 else 1)


if __name__ == "__main__":
    main()
