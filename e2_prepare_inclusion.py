#!/usr/bin/env python3
"""Build blind human-coding sheets for E2 inclusion (prereg section 4.2).

Sampling frame: one representative rewrite per (seed_id, condition) that has at
least one validator-pass candidate. The representative is the first passing
candidate by SHA256(seed|candidate), matching the analysis-set rule in 4.3.

From that frame, take the first 60 (seed, condition) pairs per condition by
SHA256(seed|condition). Assign blind IDs by SHA256("blind|generator|seed|condition")
so neither condition nor seed order is visible. The remaining pairs are sampled
at 10% per condition for the post-hoc audit.

Usage:
  python e2_prepare_inclusion.py --generator G1 \
      --input e2_generation/wave1_deepseek.jsonl --out-dir e2_validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def h(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def load_latest(path: Path) -> dict[tuple[str, str, int], dict]:
    latest: dict[tuple[str, str, int], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["seed_id"]), str(row["condition"]), int(row["candidate"]))
            latest[key] = row
    return latest


def representative_pairs(latest: dict[tuple[str, str, int], dict]) -> dict[tuple[str, str], dict]:
    """First validator-pass candidate per (seed, condition), by hash order."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (seed, condition, _candidate), row in latest.items():
        if row.get("parsed_ok") and row.get("validator_pass") is True:
            grouped[(seed, condition)].append(row)

    reps: dict[tuple[str, str], dict] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: h(key[0], str(r["candidate"])))
        reps[key] = rows[0]
    return reps


def write_sheet(path: Path, rows: list[dict], with_labels: bool) -> None:
    fieldnames = ["blind_id", "subject_original", "body_original",
                  "subject_rewrite", "body_rewrite"]
    if with_labels:
        fieldnames += ["valid", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row[k] for k in fieldnames if k in row}
            if with_labels:
                out.setdefault("valid", "")
                out.setdefault("notes", "")
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, help="G1 or G2")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="e2_validation")
    parser.add_argument("--n-per-condition", type=int, default=60)
    parser.add_argument("--audit-frac", type=float, default=0.10)
    parser.add_argument("--salt", default="",
                        help="extra hash salt for a fresh resample (e.g. round2)")
    parser.add_argument("--tag", default="",
                        help="suffix for output filenames (e.g. _round2)")
    parser.add_argument("--exclude-key", action="append", default=[],
                        help="key CSV whose sample pairs are excluded from this draw "
                             "(repeat to exclude several previous rounds)")
    args = parser.parse_args()

    latest = load_latest(Path(args.input))
    reps = representative_pairs(latest)
    excluded = set()
    for exclude_path in args.exclude_key:
        with Path(exclude_path).open(encoding="utf-8-sig", newline="") as handle:
            excluded |= {(row["seed_id"], row["condition"])
                         for row in csv.DictReader(handle)
                         if row.get("sheet") == "sample"}
    if excluded:
        reps = {key: row for key, row in reps.items() if key not in excluded}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Blind ID depends only on generator/seed/condition (and the optional salt),
    # not on condition order. A non-empty salt produces an independent resample.
    id_parts = ["blind", args.generator]
    if args.salt:
        id_parts.append(args.salt)
    blind = {key: h(*id_parts, key[0], key[1])[:12] for key in reps}
    order_salt = args.salt or ""
    ordered = sorted(reps, key=lambda key: h(order_salt, key[0], key[1]))

    sample_keys: list[tuple[str, str]] = []
    audit_keys: list[tuple[str, str]] = []
    by_condition: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in ordered:
        by_condition[key[1]].append(key)

    for condition, keys in sorted(by_condition.items()):
        take = keys[:args.n_per_condition]
        rest = keys[args.n_per_condition:]
        sample_keys.extend(take)
        n_audit = max(1, round(len(rest) * args.audit_frac))
        audit_keys.extend(rest[:n_audit])

    sample_rows, key_rows = [], []
    for key in sorted(sample_keys, key=lambda k: blind[k]):
        row = reps[key]
        sample_rows.append({
            "blind_id": blind[key],
            "subject_original": row.get("subject_original", ""),
            "body_original": row.get("original", ""),
            "subject_rewrite": row.get("subject_paraphrase", ""),
            "body_rewrite": row.get("paraphrase", ""),
        })
        key_rows.append({
            "blind_id": blind[key], "seed_id": key[0], "condition": key[1],
            "candidate": row.get("candidate"), "provider": row.get("provider", ""),
            "model": row.get("model", ""),
        })

    audit_rows = []
    for key in sorted(audit_keys, key=lambda k: blind[k]):
        row = reps[key]
        audit_rows.append({
            "blind_id": blind[key],
            "subject_original": row.get("subject_original", ""),
            "body_original": row.get("original", ""),
            "subject_rewrite": row.get("subject_paraphrase", ""),
            "body_rewrite": row.get("paraphrase", ""),
        })
        key_rows.append({
            "blind_id": blind[key], "seed_id": key[0], "condition": key[1],
            "candidate": row.get("candidate"), "provider": row.get("provider", ""),
            "model": row.get("model", ""), "sheet": "audit",
        })

    for row in key_rows:
        row.setdefault("sheet", "sample")

    prefix = f"{args.generator}_inclusion{args.tag}"
    write_sheet(out_dir / f"{prefix}_sample_300.csv", sample_rows, with_labels=True)
    write_sheet(out_dir / f"{prefix}_sample_300_coder2.csv", sample_rows, with_labels=True)
    write_sheet(out_dir / f"{prefix}_audit_10pct.csv", audit_rows, with_labels=True)
    with (out_dir / f"{prefix}_key.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    counts = {c: len(v) for c, v in sorted(by_condition.items())}
    sampled = {c: sum(1 for k in sample_keys if k[1] == c) for c in by_condition}
    lines = [
        f"# {args.generator} inclusion sampling",
        "",
        f"- input: `{args.input}`",
        f"- unique generated keys: {len(latest)}",
        f"- validator-pass representative pairs: {len(reps)}",
        (f"- excluded pairs from {', '.join(f'`{p}`' for p in args.exclude_key)}: "
         f"{len(excluded)}" if args.exclude_key else "- excluded pairs: 0"),
        f"- representative pairs per condition: {counts}",
        f"- sampled per condition: {sampled}",
        f"- audit rows: {len(audit_rows)}",
        "",
        "Sampling frame keeps one passing candidate per (seed, condition), chosen by",
        "SHA256(seed|candidate); the 60 per condition are the first by SHA256(seed|condition).",
        "Blind IDs hide condition and seed order. Coders judge whether content and",
        "maliciousness are unchanged and whether any request/fact was added.",
    ]
    (out_dir / f"{prefix}_sampling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
