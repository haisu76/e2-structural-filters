# Supplementary Tables D1–D8

Tables D1–D6 were generated 2026-11-19 by `paper/make_supplementary_tables.py`;
D7 was generated 2026-11-23 by `paper/compute_h1_prespecified.py`; D8 was
generated from the restricted G1 log by `paper/compute_g1_condition_funnel.py`.
All files are
UTF-8 with BOM, comma-separated, one header row, and no index column. Numeric
values are written with six significant digits (`%.6g`); rates recomputed from
integer counts may differ in the last digit.

| ID | File | Rows | Content |
|---|---|---|---|
| D1a | `D1a_full_H4_20_cell_table.csv` | 20 | Full exploratory H4 table: detector × rewrite condition, with flip count, flip rate, Pearson r between within-rater temperature change and detection, p, and BH q. |
| D1b | `D1b_H4_pooled_partial_correlations.csv` | 5 | Pooled within-condition partial correlations by detector. |
| D2 | `D2_feature_family_drift.csv` | 25 | Feature-family drift by detector × condition: mean absolute z shift, its bootstrap CI, proportion of features with \|Δz\| > 0.5, original–rewrite feature Spearman ρ, and drift difference versus neutral with CI, p, and q. |
| D3 | `D3_pairwise_family_contrasts.csv` | 42 | All pairwise detector-family contrasts of paired evasion, by condition and pooled: mean difference, 95% bootstrap CI, p, and BH q. |
| D4 | `D4_detector_metrics.csv` | 25 | Full detector metrics by detector × condition, including baseline and rewrite detection rates, evasion, paired detection change and CI, flip rate, standardized score drift, selected C, validation and test AUC, threshold, FPR/TPR at the shared operating point, and q values. |
| D5a | `D5a_rating_QC_v1_v2.csv` | 2 | First-pass and official QC verdicts: flag counts for scale, quantization, block, and anchor checks; rejected blocks. |
| D5b | `D5b_rating_reliability_ICC.csv` | 4 | Dimension reliability on the 600 non-anchor items: consistency ICC(2,3) (primary), absolute-agreement ICC(2,3), and Krippendorff interval α. |
| D6a | `D6a_G2_generation_inclusion_counts.csv` | 40 | G2 generation, audit, screen, and final inclusion counts by stage and condition, with Wilson 95% CIs for the two overall invalid rates. |
| D6b | `D6b_G2_evasion_matrix.csv` | 25 | G2 evasion matrix by detector × condition with paired detection change and CI, score drift and CI, threshold, and test FPR. |
| D6c | `D6c_G2_paired_contrasts.csv` | 42 | G2 pairwise family contrasts by condition and pooled: mean difference and 95% bootstrap CI. |
| D7 | `D7_H1_prespecified_aggregate_contrasts.csv` | 2 | The two prespecified H1 aggregate evasion contrasts on 115 complete seeds, computed for the manuscript revision from released text-free scores. |
| D8 | `D8_G1_condition_funnel.csv` | 6 | Text-free G1 condition funnel: attempted slots, hard-valid slots, seed-conditions with at least one pass, and final included pairs. The sixth row gives totals. |

## Sources and definitions

- **D1** — `e2_analysis/h4_temperature_flip.csv`, `e2_analysis/h4_pooled.csv` (G1). Cells with fewer than 10 flips are flagged in `note` and are descriptive only.
- **D2** — `e2_analysis/family_drift.csv` (G1). Drift is computed on standardized features; sparse families use non-zero features with training document frequency ≥ 1%.
- **D3** — `e2_analysis/detector_family_order.csv` (G1). Contrasts are paired evasion differences on complete-pair seeds; `condition = all` is the pooled estimate across conditions.
- **D4** — `e2_analysis/detector_metrics.csv` (G1). `test_fpr` is the false-positive rate on the held-out test set at the threshold selected for TPR = .90 on validation data.
- **D5a** — `e2_ratings/qc_e2_600/qc_verdict.json` (v1, REVISE) and `e2_ratings/qc_e2_600_v2/qc_verdict.json` (v2, official, PASS). The v2 pass has nine non-blocking quantization warnings.
- **D5b** — computed from `e2_ratings/e2_600_R{1,2,3}_filled_v2.csv` on the 600 non-anchor items, with valence reverse-scored as mean(8 − V1, V2). Consistency ICC is the value reported in the manuscript (Section 3.10 and Section 7); absolute-agreement ICC and interval α are reported for reference. Distance is exploratory. The raters are AI systems.
- **D6a** — generation counts from `e2_generation/wave1_qwen.jsonl` (last record per seed × condition × candidate; 5,586 log records → 3,000 slots); audit and screen counts from the PI-verified verdict files in `e2_validation/PI_only/`; final counts from `e2_analysis/e2_pairs_G2.csv`.
- **D6b–D6c** — `e2_analysis/detector_metrics_G2.csv` and `e2_analysis/detector_family_order_G2.csv`. Thresholds are identical to G1 (amendment 2, clause F6). Per clause F3 (18 complete seeds < 30), G2 results are descriptive only: D6c intentionally omits p and q values, and no significance claim is made.
- **D7** — `e2_analysis/detector_pair_scores.csv` joined to complete seeds in `e2_analysis/pairs_index_G1.csv`. The middle tier is the equal-weight mean of D1, D4 and D5 evasion, exactly as defined for H1; 10,000 seed-clustered percentile resamples use RNG seed 43. This is a calculation on frozen scores, not a new detector run.
- **D8** — restricted `e2_generation/wave1_deepseek.jsonl` contains 3,935 physical records for 3,000 seed × condition × candidate slots; 935 slots have two attempts. The last physical record per slot is retained. All 857 keys in the released final G1 index are hard-valid under that rule. `e2_validation/G1_wave1_summary.md` describes an earlier 3,925-record snapshot, before ten additional attempts for seed E2-200. It gives the same overall 2,630 passes but assigns one more to `more_urgent` and one fewer to `less_urgent`. D8 uses the final log consistent with the analysis index. The script exports only counts and never publishes the message text.

## Reproduce

```bash
python paper/make_supplementary_tables.py
python paper/compute_h1_prespecified.py
python paper/compute_g1_condition_funnel.py --log /private/path/wave1_deepseek.jsonl
```
