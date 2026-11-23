"""Export text-free G1 condition counts from the restricted raw generation log.

Usage: python paper/compute_g1_condition_funnel.py --log /path/to/wave1_deepseek.jsonl
The script reads message fields but never writes or prints them. It retains the
last physical record for each (seed_id, condition, candidate) slot, matching
the final 3,935-record log used by the released G1 analysis index.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("neutral", "warmer", "colder", "more_urgent", "less_urgent")

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True, type=Path)
args = parser.parse_args()

slots = {}
records = 0
with args.log.open(encoding="utf-8-sig") as source:
    for line in source:
        row = json.loads(line)
        records += 1
        key = (str(row["seed_id"]), str(row["condition"]), str(row["candidate"]))
        slots[key] = bool(row["validator_pass"])

assert records == 3935 and len(slots) == 3000, (records, len(slots))
final = pd.read_csv(ROOT / "e2_analysis/pairs_index_G1.csv", encoding="utf-8-sig")
final_keys = {(str(r.seed_id), str(r.condition), str(r.candidate)) for r in final.itertuples()}
assert len(final_keys) == len(final) == 857
assert final_keys <= slots.keys()
assert all(slots[key] for key in final_keys), "Final index contains a hard-validation failure"

attempted = Counter(condition for _, condition, _ in slots)
passed = Counter(condition for (_, condition, _), ok in slots.items() if ok)
representative = defaultdict(set)
for (seed, condition, _), ok in slots.items():
    if ok:
        representative[condition].add(seed)
included = Counter(final["condition"])

output = ROOT / "paper/submission/supplementary/D8_G1_condition_funnel.csv"
with output.open("w", encoding="utf-8-sig", newline="") as target:
    writer = csv.writer(target)
    writer.writerow(("condition", "attempted_slots", "hard_valid_slots", "hard_valid_rate",
                     "seed_conditions_with_pass", "final_included_pairs"))
    for condition in ORDER:
        assert attempted[condition] == 600
        writer.writerow((condition, attempted[condition], passed[condition],
                         f"{passed[condition] / 600:.6f}", len(representative[condition]),
                         included[condition]))
    writer.writerow(("TOTAL", sum(attempted.values()), sum(passed.values()),
                     f"{sum(passed.values()) / sum(attempted.values()):.6f}",
                     sum(len(v) for v in representative.values()), sum(included.values())))
assert sum(passed.values()) == 2630
assert [len(representative[c]) for c in ORDER] == [192, 194, 191, 194, 194]
print(f"{output}: {records} records, {len(slots)} slots, {sum(passed.values())} hard-valid, {len(final)} included")
