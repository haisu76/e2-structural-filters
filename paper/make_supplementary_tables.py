#!/usr/bin/env python3
"""Export supplementary Tables D1-D6 for the Elsevier submission.

Inputs (all relative to the project root)
  e2_analysis/h4_temperature_flip.csv, h4_pooled.csv        -> D1a, D1b
  e2_analysis/family_drift.csv                              -> D2
  e2_analysis/detector_family_order.csv                     -> D3
  e2_analysis/detector_metrics.csv                          -> D4
  e2_ratings/qc_e2_600*/qc_verdict.json                     -> D5a
  e2_ratings/e2_600_R*_filled_v2.csv + annotation map       -> D5b
  e2_generation/wave1_qwen.jsonl, G2 inclusion/audit files  -> D6a
  e2_analysis/detector_metrics_G2.csv                       -> D6b
  e2_analysis/detector_family_order_G2.csv                  -> D6c

Output  paper/submission/supplementary/
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "submission" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "annotation"))
import analyze_calibration as ac  # noqa: E402
from diagnose_400 import consistency_icc  # noqa: E402

CONDITIONS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]


def load(rel, **kw):
    return pd.read_csv(ROOT / rel, encoding="utf-8-sig", **kw)


def write(df, name):
    path = OUT / name
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL,
              lineterminator="\n", float_format="%.6g")
    print(f"wrote {path.relative_to(ROOT)}  ({len(df)} rows)")


def wilson(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def export_d1():
    d1 = load("e2_analysis/h4_temperature_flip.csv")
    d1["note"] = np.where(d1["note"].notna(),
                          "fewer than 10 flips; descriptive only", "")
    d1 = d1.rename(columns={"dimension": "temperature_dimension"})
    write(d1, "D1a_full_H4_20_cell_table.csv")

    d1b = load("e2_analysis/h4_pooled.csv")
    write(d1b, "D1b_H4_pooled_partial_correlations.csv")


def export_d2_d4():
    write(load("e2_analysis/family_drift.csv"), "D2_feature_family_drift.csv")
    write(load("e2_analysis/detector_family_order.csv"),
          "D3_pairwise_family_contrasts.csv")
    write(load("e2_analysis/detector_metrics.csv"), "D4_detector_metrics.csv")


def export_d5():
    rows = []
    for key, label in [("qc_e2_600", "v1"), ("qc_e2_600_v2", "v2 (official)")]:
        with open(ROOT / f"e2_ratings/{key}/qc_verdict.json", encoding="utf-8") as f:
            q = json.load(f)
        c = q["counts"]
        rows.append({
            "pass": label,
            "verdict": q["verdict"],
            "generated_at": q["generated_at"],
            "n_rows": q["n_rows"],
            "n_unique_items": q["n_unique_items"],
            "n_anchor_pairs_per_rater": q["n_anchor_pairs_per_rater"],
            "scale_flags": c["scale_flags"],
            "quantization_flags": c["quantization_flags"],
            "quantization_blocks": c["quantization_blocks"],
            "block_flags": c["block_flags"],
            "anchor_flags": c["anchor_flags"],
            "rejected_blocks": "; ".join(
                f"{b['rater_id']} block {b['block']}" for b in q["rejected_blocks"]),
        })
    write(pd.DataFrame(rows), "D5a_rating_QC_v1_v2.csv")

    amp = load("e2_ratings/e2_annotation_600_map.csv")
    keep = set(amp.loc[amp["is_anchor"] == 0, "sheet_id"])
    raters = []
    for r in ("R1", "R2", "R3"):
        df = load(f"e2_ratings/e2_600_{r}_filled_v2.csv")
        df["rater_id"] = r
        raters.append(df)
    df = pd.concat(raters, ignore_index=True)
    df = df[df["sheet_id"].isin(keep)]
    df = df.assign(
        valence=(8 - df["V1_negative_tone"] + df["V2_goodwill"]) / 2,
        urgency=df[["A1_time_pressure", "A2_intensity", "A3_act_now"]].mean(axis=1),
        warmth=df[["W1_care", "W2_friendly", "W3_person_not_ticket"]].mean(axis=1),
        distance=df[["D1_institutional", "D2_impersonal", "D3_procedural"]].mean(axis=1),
    )
    expected = {"valence": .391, "urgency": .791, "warmth": .670, "distance": .694}
    rows = []
    for dim, items in [("valence", 2), ("urgency", 3), ("warmth", 3), ("distance", 3)]:
        wide = df.pivot(index="sheet_id", columns="rater_id", values=dim).dropna()
        mat = wide.to_numpy(dtype=float)
        cons = consistency_icc(mat)
        assert abs(cons - expected[dim]) < 5e-4, (dim, cons)
        rows.append({
            "dimension": dim,
            "n_items": items,
            "n_cases": len(wide),
            "icc_consistency_2_3": cons,
            "icc_absolute_2_3": ac.icc_2k(mat),
            "krippendorff_alpha_interval": ac.krippendorff_alpha(mat, level="interval"),
            "use": "exploratory" if dim == "distance" else "primary",
        })
    write(pd.DataFrame(rows), "D5b_rating_reliability_ICC.csv")


def export_d6():
    recs = [json.loads(line) for line in
            open(ROOT / "e2_generation/wave1_qwen.jsonl", encoding="utf-8")]
    last = {(r["seed_id"], r["condition"], r["candidate"]): r for r in recs}
    valid = [r for r in last.values() if r.get("parsed_ok") and r.get("validator_pass")]
    assert len(last) == 3000 and len(valid) == 844

    inclusion = load("e2_validation/G2_inclusion.csv")
    assert len(inclusion) == 481
    audit_key = load("e2_validation/PI_only/G2_audit_65_key.csv")
    audit = audit_key.merge(load("e2_validation/PI_only/G2_audit_verdicts.csv"),
                           on="blind_id", how="left")
    screen_key = load("e2_validation/PI_only/G2_screen_key.csv")
    screen = screen_key.merge(load("e2_validation/PI_only/G2_screen_verdicts.csv"),
                              on="blind_id", how="left")
    final = load("e2_analysis/e2_pairs_G2.csv")
    assert len(final) == 421 and final["seed_id"].nunique() == 163
    assert int(load("e2_validation/G2_inclusion_audit_adjusted.csv")
               .eval("included == 1").sum()) == 455
    assert final.loc[final["complete_pair"] == 1, "seed_id"].nunique() == 18

    rows = []

    def add(stage, cond, metric, value="", denom="", est="", lo="", hi="", note=""):
        rows.append({"stage": stage, "condition": cond, "metric": metric,
                     "value": value, "denominator": denom, "estimate": est,
                     "ci_low": lo, "ci_high": hi, "note": note})

    add("generation", "all", "slots_attempted", len(last))
    add("generation", "all", "log_records", len(recs))
    p, lo, hi = wilson(len(valid), len(last))
    add("generation", "all", "hard_valid_slots", len(valid), len(last), p, lo, hi,
        "parsed_ok and validator_pass; last record per slot")
    for cond in CONDITIONS:
        k = sum(1 for r in valid if r["condition"] == cond)
        n = sum(1 for r in last.values() if r["condition"] == cond)
        add("generation", cond, "hard_valid_slots", k, n, k / n)
    add("generation", "all", "seeds_with_all_five_valid", 23)
    add("generation", "all", "representative_pairs", len(inclusion))
    for cond in CONDITIONS:
        add("generation", cond, "representative_pairs",
            int((inclusion["condition"] == cond).sum()))

    add("audit", "all", "pairs_audited", len(audit))
    inv = int((audit["PI_valid"] == 0).sum())
    p, lo, hi = wilson(inv, len(audit))
    add("audit", "all", "invalid_pairs", inv, len(audit), p, lo, hi,
        "Wilson 95% CI; PI-verified")
    for cond in CONDITIONS:
        sub = audit[audit["condition"] == cond]
        k = int((sub["PI_valid"] == 0).sum())
        add("audit", cond, "invalid_pairs", k, len(sub), k / len(sub))
    add("audit", "all", "screen_recall_among_audit_invalid", 8, inv, 8 / inv,
        note="frozen six-rule screen")

    add("audit_adjusted", "all", "included_pairs", 455)
    add("audit_adjusted", "all", "complete_seeds", 20)

    add("screen", "all", "pairs_screened", len(screen))
    inv_s = int((screen["PI_valid"] == 0).sum())
    p, lo, hi = wilson(inv_s, len(screen))
    add("screen", "all", "invalid_pairs", inv_s, len(screen), p, lo, hi,
        "Wilson 95% CI; PI-verified")
    for cond in CONDITIONS:
        sub = screen[screen["condition"] == cond]
        k = int((sub["PI_valid"] == 0).sum())
        add("screen", cond, "invalid_pairs", k, len(sub), k / len(sub))

    add("final", "all", "included_pairs", len(final))
    for cond in CONDITIONS:
        add("final", cond, "included_pairs",
            int((final["condition"] == cond).sum()))
    add("final", "all", "complete_seeds", 18)
    add("final", "all", "complete_pairs", 90)

    d6a = pd.DataFrame(rows)
    overall = d6a[(d6a["metric"] == "invalid_pairs") & (d6a["condition"] == "all")]
    assert overall["value"].astype(int).sum() == 60
    write(d6a, "D6a_G2_generation_inclusion_counts.csv")

    m = load("e2_analysis/detector_metrics_G2.csv")
    cols = ["detector", "condition", "n_pairs", "n_seeds",
            "original_detection_rate", "detection_rate", "evasion_rate",
            "detection_delta", "detection_delta_lo", "detection_delta_hi",
            "score_drift_z", "drift_lo", "drift_hi", "threshold", "test_fpr"]
    d6b = m[cols].copy()
    d6b["condition"] = pd.Categorical(d6b["condition"], CONDITIONS, ordered=True)
    d6b = d6b.sort_values(["detector", "condition"]).reset_index(drop=True)
    write(d6b, "D6b_G2_evasion_matrix.csv")

    f = load("e2_analysis/detector_family_order_G2.csv")
    d6c = f.rename(columns={"lo": "ci_low", "hi": "ci_high"})
    d6c["condition"] = d6c["condition"].replace({"all": "pooled"})
    d6c = d6c[["condition", "contrast", "n_seeds", "mean_diff", "ci_low", "ci_high"]]
    d6c["note"] = "descriptive only (amendment 2, clause F3)"
    write(d6c, "D6c_G2_paired_contrasts.csv")


def main():
    export_d1()
    export_d2_d4()
    export_d5()
    export_d6()


if __name__ == "__main__":
    main()
