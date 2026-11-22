#!/usr/bin/env python3
"""Validate coder C's 211-pair screen review and write PI verdicts.

Checks
  - 211 rows, same blind_ids and order as the sent sheet, unchanged text columns
  - valid in {1, 0}; every 0 has notes in `code: "orig" -> "rew"` form
  - both quoted fragments occur (whitespace/case-normalized) in the pair's texts

Output
  e2_validation/PI_only/G1_audit_screen_verdicts.csv
    blind_id, seed_id, condition, candidate, pool, screen_flags,
    coder_C_valid, PI_valid, PI_note
"""
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_validation"
PKG = OUT / "audit_round3_coderC_screen"

# PI reviewed these in full: same-sentence citations, but close to the 3.3/3.4 line.
BORDERLINE = {
    "2715747c6b0e",  # "cannot hit you" -> "does not have to hit you"
    "9d572f15f114",  # "so that ... cannot hit you" -> "from a major expense arriving"
    "87ce08f5ebbf",  # "we say YES" -> "we look for a way to say yes"
    "e18730022617",  # "best ... Period." -> "leading ... That is our view."
}


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(text):
    text = unicodedata.normalize("NFKC", text)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip().casefold()


def main():
    src = load(PKG / "G1_audit_screen_pairs.csv")
    filled = load(PKG / "G1_audit_screen_pairs_filled.csv")
    key = {r["blind_id"]: r for r in load(OUT / "PI_only/G1_audit_screen_key.csv")}

    assert len(src) == len(filled) == len(key) == 211, (len(src), len(filled), len(key))
    assert [r["blind_id"] for r in src] == [r["blind_id"] for r in filled], "row order changed"
    assert set(key) == {r["blind_id"] for r in filled}, "key/sheet id mismatch"

    text_cols = ["subject_original", "body_original", "subject_rewrite", "body_rewrite"]
    changed = [(a["blind_id"], c) for a, b in zip(src, filled) for c in text_cols if a[c] != b[c]]
    assert not changed, f"text columns changed: {changed[:3]}"

    verdicts, problems = [], []
    for row in filled:
        bid = row["blind_id"]
        valid = row["valid"].strip()
        assert valid in {"1", "0"}, (bid, valid)
        note = ""
        if valid == "0":
            m = re.match(r'^\s*([a-z_]+)\s*:\s*"(.*?)"\s*→\s*"(.*?)"\s*$', row["notes"], re.S)
            if not m:
                problems.append((bid, "unparsed note"))
                continue
            code, orig_frag, rew_frag = m.groups()
            orig = norm(row["subject_original"] + "\n" + row["body_original"])
            rew = norm(row["subject_rewrite"] + "\n" + row["body_rewrite"])
            if norm(orig_frag) not in orig:
                problems.append((bid, f"orig fragment missing: {orig_frag[:60]}"))
            if norm(rew_frag) not in rew:
                problems.append((bid, f"rewrite fragment missing: {rew_frag[:60]}"))
            note = "citation_verified"
            if bid in BORDERLINE:
                note += "; borderline (PI reviewed, confirmed 0)"
            note += f"; {code}"
        k = key[bid]
        verdicts.append({
            "blind_id": bid,
            "seed_id": k["seed_id"],
            "condition": k["condition"],
            "candidate": k["candidate"],
            "pool": k["pool"],
            "screen_flags": k["screen_flags"],
            "coder_C_valid": valid,
            "PI_valid": valid,
            "PI_note": note,
        })

    if problems:
        for p in problems:
            print("PROBLEM", p)
        raise SystemExit(f"{len(problems)} citation problems; fix before applying")

    path = OUT / "PI_only/G1_audit_screen_verdicts.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(verdicts[0].keys()))
        w.writeheader()
        w.writerows(verdicts)

    zeros = [v for v in verdicts if v["PI_valid"] == "0"]
    print(f"rows={len(verdicts)} valid={len(verdicts) - len(zeros)} invalid={len(zeros)}")
    print(f"invalid by pool={dict(Counter(v['pool'] for v in zeros))}")
    print(f"invalid by condition={dict(Counter(v['condition'] for v in zeros))}")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
