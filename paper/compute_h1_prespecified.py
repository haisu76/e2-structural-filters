"""Compute the two prespecified H1 aggregate evasion contrasts.

Uses only the released text-free complete-seed pair scores. No model fitting
or additional message-level data are required.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'e2_analysis'
score = pd.read_csv(DATA / 'detector_pair_scores.csv', encoding='utf-8-sig')
index = pd.read_csv(DATA / 'pairs_index_G1.csv', encoding='utf-8-sig')
complete = set(index.loc[index['complete_pair'].eq(1), 'seed_id'])
score = score.loc[score['seed_id'].isin(complete)].copy()
score['evasion'] = 1 - score['detect_rewrite']
wide = score.pivot(index=['seed_id', 'condition'], columns='detector', values='evasion')
assert len(complete) == 115 and len(wide) == 575 and wide.notna().all().all()

middle = wide[['D1', 'D4', 'D5']].mean(axis=1)
contrasts = {'D3-minus-middle': wide['D3'] - middle,
             'middle-minus-D2': middle - wide['D2']}
rng = np.random.default_rng(43)
records = []
for name, values in contrasts.items():
    per_seed = values.groupby(level='seed_id').mean().to_numpy()
    draws = rng.integers(0, len(per_seed), size=(10_000, len(per_seed)))
    bootstrap = per_seed[draws].mean(axis=1)
    lo, hi = np.quantile(bootstrap, [.025, .975])
    records.append({'contrast': name, 'n_seeds': len(per_seed),
                    'mean_diff': per_seed.mean(), 'lo': lo, 'hi': hi,
                    'scale': 'evasion-rate difference',
                    'ci_method': '10000 seed-clustered percentile bootstrap resamples; RNG seed 43'})

out = ROOT / 'paper/submission/supplementary/D7_H1_prespecified_aggregate_contrasts.csv'
pd.DataFrame(records).round(6).to_csv(out, index=False, encoding='utf-8-sig')
print(out)
