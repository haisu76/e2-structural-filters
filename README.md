# Structural Filters Are the Least Robust — replication package

Version 1.0.3 · 2026-11-23

Code, prompts, detector configurations, derived tables, and figure scripts for
the preregistered stress test of spam-detector feature families under
content-preserving LLM rewrites.

- Preregistration and amendments 1–2 (frozen documents, SHA-256 manifest):
  https://doi.org/10.5281/zenodo.24363906
- This package: https://github.com/haisu76/e2-structural-filters/tree/v1.0.3
  (no software DOI has been assigned)

The two records serve different purposes. The Zenodo record above archives the
protocol documents and proves what was frozen before data collection. This
package contains the materials behind the reported results: analysis code,
prompts, detector definitions, derived tables, and figure scripts.

## Contents

| Path | Contents |
|---|---|
| `*.py` (root) | Full analysis pipeline: corpus audit, sample construction, rewrite generation, validator, inclusion coding, detector training and scoring, H4 analysis, reliability |
| `annotation/` | Rating schema (v1, v2, v2.1), rating templates, ICC / Krippendorff reliability code |
| `prompts/rewrite_prompts.md` | Shared system prompt, the five condition prompts, and L2/L3 templates |
| `e2_analysis/` | Derived tables behind every reported number, plus `G1_VERDICT.md` and the three `DETECTOR_DESIGN*.md` files |
| `audit_tables/` | Corpus audit aggregates (98,609 documents, 7 corpora) |
| `paper/make_fig*.py` | Figure scripts (Matplotlib) |
| `paper/figures/` | Rendered figures: 600 dpi PNG and vector PDF |
| `paper/make_supplementary_tables.py` | Builds supplementary Tables D1–D6 |
| `paper/submission/supplementary/` | Supplementary Tables D1–D6 (CSV) with a README |
| `docs/` | Corpus audit report and model provenance (API endpoints, model versions, dates) |

## What is not included

| Material | Reason | Access |
|---|---|---|
| Raw seed messages (`samples/`) | Third-party corpus; redistribution subject to the original corpus license | Original corpus source |
| Generated rewrite text (`e2_generation/`) | LLM output containing spam content; license and content review | Corresponding author on request |
| Human/AI ratings and adjudication files (`e2_ratings/`, `e2_validation/`) | Annotation materials; de-identification review | Corresponding author on request |
| Full pair tables with message text | Same as above | `pairs_index_*.csv` provides the same rows without text |

`e2_analysis/pairs_index_G1.csv`, `pairs_index_G2.csv`, and
`pairs_index_G1_sensitivity.csv` are text-free versions of the full pair tables
(857, 421, and 789 rows). They retain seed, condition, candidate, completeness,
inclusion basis, length ratio, and validator warnings, so inclusion counts can
be checked without redistributing message text.

## Environment

Python 3.11 or newer (tested on 3.13). Install dependencies with:

```
pip install -r requirements.txt
```

## Reproducing the figures

Figures 1–3 and 5 regenerate from the derived tables in `e2_analysis/`:

```
cd paper
python make_fig1_design_flow.py
python make_fig2_evasion_heatmap.py
python make_fig3_operating_point.py
python make_fig2_evasion_heatmap.py \
  --metrics ../e2_analysis/detector_metrics_G2.csv \
  --out fig5_g2_heatmap \
  --title "G2 (Qwen2.5-7B): evasion by detector family and condition"
```

Outputs are written to `paper/figures/` as 600 dpi PNG and vector PDF. Figure
counts in Fig. 1 are transcribed from `e2_analysis/G1_VERDICT.md` and are
hard-coded in the script.

Figure 4 additionally reads the rating files
`e2_ratings/e2_600_R{1,2,3}_filled.csv` and
`e2_ratings/e2_annotation_600_map.csv`, which are not included. The figure
script and its inputs are documented in its docstring; the rendered figure is
in `paper/figures/`.

## Reproducing the analysis tables

`e2_detectors.py` trains D1–D5 and writes the detector tables:

```
python e2_detectors.py
```

It requires two inputs that are not in this package:
`samples/clean_spamassassin.csv` (third-party corpus) and
`e2_analysis/e2_pairs_G1.csv` (the pair table with message text). Add `--tag
_G2` with the G2 pair table to reproduce the G2 tables.

Other entry points:

| Script | Purpose |
|---|---|
| `audit_corpora.py` | Rebuild `CORPUS_AUDIT.md` and `audit_tables/` |
| `build_samples.py` | Build cleaned corpus and fixed E1/E2 samples |
| `generate_rewrites.py` | Generate rewrites (mock or OpenAI-compatible endpoint) |
| `validate_rewrites.py` | Rule-based validator for content preservation |
| `e2_prepare_inclusion.py` | Build representative pairs and inclusion tables |
| `e2_analyze_ratings.py` | Manipulation check from the 600-item rating set |
| `e2_inclusion_agreement.py` | Inter-coder agreement and audit calibration |
| `e2_h4_temperature_flip.py` | Exploratory H4 analysis |
| `paper/make_supplementary_tables.py` | Build Tables D1–D6 |

The full pipeline is: corpus audit → sample construction → rewriting →
validation → inclusion coding and adjudication → analysis set → detector
training and scoring → H4 and reliability → figures and supplementary tables.
Later stages consume the restricted inputs listed above; see each script's
docstring for exact paths and outputs.

## Derived tables

All CSVs are UTF-8 with BOM.

| File | Rows | Content |
|---|---|---|
| `detector_metrics.csv` | 25 | Detector × condition: detection and evasion rates, paired change and CI, flip rate, score drift, selected C, validation/test AUC, threshold, FPR, TPR, q values (G1) |
| `detector_pair_scores.csv` | 4,285 | Included pair × detector: original and rewrite scores, decisions, standardized drift (text-free) |
| `detector_split.csv` | 6,148 | Corpus row → group, label, and train/validation/test split |
| `family_drift.csv` | 25 | Feature-family drift by detector × condition with bootstrap CIs (G1) |
| `detector_family_order.csv` | 42 | Pairwise detector-family contrasts of paired evasion (G1) |
| `h4_temperature_flip.csv` | 20 | Exploratory H4: detector × condition flip counts, rates, correlations, q values |
| `h4_pooled.csv` | 5 | Pooled within-condition partial correlations by detector |
| `pairs_index_G1.csv` | 857 | Text-free pair index (G1 analysis set) |
| `pairs_index_G2.csv` | 421 | Text-free pair index (G2) |
| `*_G2.csv`, `*_sensitivity.csv` | — | The same tables for the G2 run and the sensitivity run |
| `G1_VERDICT.md` | — | Frozen G1 counts and verdicts cited in the manuscript |
| `DETECTOR_DESIGN*.md` | — | Exact detector configurations, hyperparameter grids, and thresholds |

## Licenses

- Code (`*.py`): MIT, see `LICENSE_CODE_MIT.txt`.
- Derived tables, figures, and documentation: CC BY 4.0, see
  `LICENSE_DATA_CC-BY-4.0.txt`.
- Third-party corpora and generated message text are not covered by these
  licenses and are not redistributed here.

## Citation

If you use this code or the derived tables, cite the manuscript and this
package using its versioned GitHub URL. Add a software DOI to `CITATION.cff`
only if this package is later deposited and a DOI is actually assigned.

## Contact

Corresponding author: Xin Ge (gexin@usst.edu.cn). Restricted materials
are available on reasonable request, subject to license review.
