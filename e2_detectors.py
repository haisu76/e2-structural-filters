#!/usr/bin/env python3
"""Train D1-D5 spam detectors and apply them to the G1 rewrite pairs.

Implements PREREGISTRATION_E2_RQ-B.md section 6.

Split (6.1)
  Training corpus is samples/clean_spamassassin.csv. Groups are `group_id`.
  Every corpus row whose group contains one of the 200 E2 seeds is forced into
  test. The remaining groups are split 60/20/20 by group, stratified on the
  row-level spam/ham ratio, seed 42. Group assignment is greedy by row quota
  within each label stratum.

Models (6.2)
  D1 word TF-IDF 1-2, min_df=2, sublinear -> LogisticRegression
  D2 char_wb TF-IDF 3-5, min_df=2 -> LinearSVC
  D3 structural counts -> HistGradientBoosting(max_iter=300)
  D4 sentiment/urgency/warmth lexicon counts -> LogisticRegression
  D5 word TF-IDF -> TruncatedSVD(200) -> LogisticRegression
  C selected on validation AUC: D1/D4/D5 {0.1, 1, 10}; D2 {0.1, 1}.
  No rewrite text is used for training or tuning.

Operating point (6.3)
  Threshold = 10th percentile of validation-spam decision scores (TPR .90),
  shared by test and rewrite scoring.

Outputs
  e2_analysis/detector_metrics.csv
  e2_analysis/family_drift.csv
  e2_analysis/detector_pair_scores.csv
  e2_analysis/detector_split.csv
  e2_analysis/DETECTOR_DESIGN.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from validate_rewrites import (  # noqa: E402
    THREAT_CONCEPTS,
    URGENCY,
    WARMTH,
    char_count,
    concept_hits,
    extract_entities,
    term_count,
    word_count,
)

CORPUS = ROOT / "samples/clean_spamassassin.csv"
SEEDS = ROOT / "samples/e2_seeds.csv"
PAIRS = ROOT / "e2_analysis/e2_pairs_G1.csv"
OUT = ROOT / "e2_analysis"
SEED = 42
CONDS = ["neutral", "warmer", "colder", "more_urgent", "less_urgent"]
N_BOOT = 10_000

HTML_RE = re.compile(
    r"(?is)<\s*(?:html|body|div|p|br|a|table|span|font|img|td|tr|strong|em)\b"
    r"|</\s*(?:html|body|div|p|a|table|span|font|td|tr|strong|em)\s*>"
)

POSITIVE = [
    "good", "great", "best", "better", "free", "win", "winner", "won", "prize",
    "reward", "bonus", "gift", "offer", "discount", "save", "saving", "savings",
    "guarantee", "guaranteed", "opportunity", "benefit", "success", "successful",
    "happy", "glad", "pleased", "thank", "thanks", "appreciate", "welcome",
    "friend", "love", "enjoy", "excited", "excellent", "amazing", "fantastic",
    "special", "valuable", "profit", "earn", "cheap", "lowest", "exclusive",
    "congratulations", "selected", "approved", "approved", "safe", "secure",
]
NEGATIVE = [
    "bad", "worst", "worse", "problem", "issue", "error", "fail", "failed",
    "failure", "warning", "alert", "suspend", "suspended", "suspension",
    "terminate", "terminated", "penalty", "legal", "fraud", "scam", "risk",
    "danger", "dangerous", "lose", "lost", "loss", "closed", "deactivate",
    "deactivated", "expire", "expired", "deadline", "critical", "debt", "bill",
    "overdue", "charge", "fee", "complaint", "sorry", "apologize",
    "unfortunately", "unable", "denied", "declined", "blocked", "locked",
    "restricted", "violation", "illegal", "arrest", "court", "lawsuit",
    "damage", "threat", "virus", "hacked", "compromised", "stolen",
    "unauthorized", "suspicious", "final",
]
NEGATIONS = ["not", "no", "never", "none", "nobody", "nothing", "neither",
             "nor", "cannot", "can't", "don't", "doesn't", "didn't", "won't",
             "isn't", "aren't", "wasn't", "weren't"]
GREETING_RE = re.compile(
    r"^\s*(?:dear|hi|hello|hey|greetings|good morning|good afternoon)\b", re.I
)
SECOND_PERSON = ["you", "your", "yours", "yourself", "yourselves"]
THREAT_NAMES = list(THREAT_CONCEPTS)


# ---------------------------------------------------------------------------
# feature builders
# ---------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def d3_features(text: str) -> dict[str, float]:
    text = str(text)
    body = text
    ent = extract_entities(body)
    counts = {k: sum(v.values()) for k, v in ent.items()}
    letters = re.findall(r"[A-Za-z]", body)
    upper = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    return {
        "word_count": word_count(body),
        "char_count": char_count(body),
        "url_count": counts.get("urls", 0),
        "email_count": counts.get("emails", 0),
        "phone_count": counts.get("phones", 0),
        "money_count": counts.get("money", 0),
        "date_count": counts.get("dates", 0),
        "time_window_count": counts.get("time_windows", 0),
        "id_count": counts.get("ids", 0),
        "attachment_count": counts.get("attachments", 0),
        "has_url": float(counts.get("urls", 0) > 0),
        "html_markup": float(bool(HTML_RE.search(body))),
        "uppercase_ratio": upper,
        "punct_count": sum(1 for c in body if c in ".,;:!?-()[]\"'$%&/*+=<>"),
        "exclaim_density": body.count("!") / max(len(body), 1),
        "question_density": body.count("?") / max(len(body), 1),
        "digit_ratio": sum(c.isdigit() for c in body) / max(len(body), 1),
    }


def d4_features(text: str) -> dict[str, float]:
    text = str(text)
    toks = _tokens(text)
    pos = sum(t in POSITIVE for t in toks)
    neg = sum(t in NEGATIVE for t in toks)
    negated_pos = 0
    for i, tok in enumerate(toks):
        if tok in NEGATIONS and any(t in POSITIVE for t in toks[i + 1:i + 4]):
            negated_pos += 1
    return {
        "positive_count": pos,
        "negative_count": neg,
        "negated_positive": negated_pos,
        "polarity": (pos - neg) / (pos + neg + 1),
        "urgency_terms": term_count(text, URGENCY),
        "threat_concepts": len(concept_hits(text, THREAT_CONCEPTS)),
        "warmth_terms": term_count(text, WARMTH),
        "greeting": float(bool(GREETING_RE.search(text))),
        "please_count": sum(t == "please" for t in toks),
        "thanks_count": sum(t in {"thank", "thanks"} for t in toks),
        "second_person": sum(t in SECOND_PERSON for t in toks),
        "exclaim_count": text.count("!"),
        "question_count": text.count("?"),
    }


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def split_groups(corpus: pd.DataFrame, seed_groups: set[str], rng) -> pd.DataFrame:
    corpus = corpus.copy()
    corpus["is_seed_group"] = corpus["group_id"].isin(seed_groups)
    pool = corpus.loc[~corpus["is_seed_group"]]
    split = pd.Series("test", index=corpus.index)
    assigned: dict[str, str] = {}

    primary = (pool.groupby("group_id")["label"].mean() >= 0.5).astype(int)
    sizes = pool.groupby(["group_id", "label"]).size().unstack(fill_value=0)
    for label in (0, 1):
        groups = primary.index[primary == label].to_numpy()
        rng.shuffle(groups)
        n = int(sizes[label].loc[groups].sum())
        quota_train, quota_val = int(round(0.60 * n)), int(round(0.80 * n))
        acc = 0
        for gid in groups:
            if acc < quota_train:
                assigned[gid] = "train"
            elif acc < quota_val:
                assigned[gid] = "val"
            else:
                assigned[gid] = "test"
            acc += int(sizes[label].loc[gid])

    for idx, gid in corpus["group_id"].items():
        split.loc[idx] = "test" if corpus.loc[idx, "is_seed_group"] else assigned[gid]
    corpus["split"] = split
    return corpus


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class Detector:
    def __init__(self, name=None):
        self.name = name or type(self).__name__

    def fit_select(self, train, val):
        raise NotImplementedError

    def score(self, frame):
        return self.apply(frame)


def make_detector(name):
    return {
        "D1": D1(), "D2": D2(), "D3": D3(), "D4": D4(), "D5": D5(),
    }[name]


class TextModel(Detector):
    def build_vec(self):
        raise NotImplementedError

    def build_model(self, C):
        raise NotImplementedError

    def grid(self):
        return [0.1, 1.0, 10.0]

    def fit_select(self, train, val):
        self.vec = self.build_vec()
        Xtr = self.vec.fit_transform(train["text"])
        Xva = self.vec.transform(val["text"])
        best = None
        for C in self.grid():
            model = self.build_model(C)
            model.fit(Xtr, train["label"].to_numpy())
            auc = roc_auc_score(val["label"], model.decision_function(Xva))
            if best is None or auc > best[0]:
                best = (auc, C, model)
        self.val_auc, self.C, self.model = best
        return self

    def score(self, frame):
        return self.model.decision_function(self.vec.transform(frame["text"]))


class D1(TextModel):
    def build_vec(self):
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    def build_model(self, C):
        return LogisticRegression(C=C, max_iter=2000, random_state=SEED)


class D2(TextModel):
    def build_vec(self):
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)

    def grid(self):
        return [0.1, 1.0]

    def build_model(self, C):
        return LinearSVC(C=C, random_state=SEED)


class D5(TextModel):
    def build_vec(self):
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    def build_model(self, C):
        return LogisticRegression(C=C, max_iter=2000, random_state=SEED)

    def fit_select(self, train, val):
        self.vec = self.build_vec()
        Xtr = self.vec.fit_transform(train["text"])
        self.svd = TruncatedSVD(n_components=200, random_state=SEED)
        Ztr = self.svd.fit_transform(Xtr)
        Zva = self.svd.transform(self.vec.transform(val["text"]))
        best = None
        for C in self.grid():
            model = self.build_model(C)
            model.fit(Ztr, train["label"].to_numpy())
            auc = roc_auc_score(val["label"], model.predict_proba(Zva)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, C, model)
        self.val_auc, self.C, self.model = best
        return self

    def score(self, frame):
        return self.model.predict_proba(
            self.svd.transform(self.vec.transform(frame["text"])))[:, 1]


class D3(Detector):
    def fit_select(self, train, val):
        self.cols = list(d3_features("").keys())
        Xtr = np.array([[r[k] for k in self.cols] for r in train["d3"]])
        Xva = np.array([[r[k] for k in self.cols] for r in val["d3"]])
        self.model = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
        self.model.fit(Xtr, train["label"].to_numpy())
        self.val_auc = roc_auc_score(val["label"], self.model.predict_proba(Xva)[:, 1])
        self.C = "max_iter=300"
        return self

    def score(self, frame):
        X = np.array([[r[k] for k in self.cols] for r in frame["d3"]])
        return self.model.predict_proba(X)[:, 1]


class D4(Detector):
    def fit_select(self, train, val):
        self.cols = list(d4_features("").keys())
        Xtr = np.array([[r[k] for k in self.cols] for r in train["d4"]])
        Xva = np.array([[r[k] for k in self.cols] for r in val["d4"]])
        best = None
        for C in [0.1, 1.0, 10.0]:
            model = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
            model.fit(Xtr, train["label"].to_numpy())
            auc = roc_auc_score(val["label"], model.decision_function(Xva))
            if best is None or auc > best[0]:
                best = (auc, C, model)
        self.val_auc, self.C, self.model = best
        return self

    def score(self, frame):
        X = np.array([[r[k] for k in self.cols] for r in frame["d4"]])
        return self.model.decision_function(X)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def bh_q(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * n / (rank + 1))
        q[idx] = running
    return q


def boot_mean(values, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = values.mean()
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return mean, lo, hi, min(p, 1.0)


def sparse_stats(X, min_df_frac=0.01):
    """Per-feature sd and stability mask (document frequency >= 1% of train)."""
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    sd = np.maximum(np.sqrt(np.maximum(sq - mean ** 2, 0)), 1e-9)
    df = np.asarray((X > 0).mean(axis=0)).ravel()
    return sd, df >= min_df_frac


def drift_for_rows(a, b, sd, keep):
    union = (a + b).tocsr()
    out = []
    for row in range(union.shape[0]):
        idx = union.indices[union.indptr[row]:union.indptr[row + 1]]
        idx = idx[keep[idx]]
        if len(idx) == 0:
            out.append((0.0, 0.0, np.nan))
            continue
        va = np.asarray(a[row, idx].todense()).ravel()
        vb = np.asarray(b[row, idx].todense()).ravel()
        dz = (vb - va) / sd[idx]
        rho = (spearmanr(va, vb).statistic
               if len(idx) > 2 and np.ptp(va) > 0 and np.ptp(vb) > 0 else np.nan)
        out.append((np.nanmean(np.abs(dz)), np.nanmean(np.abs(dz) > 0.5), rho))
    return out


def dense_drift(f_orig, f_rewrite, sd):
    valid = sd > 1e-9
    dz = (f_rewrite[:, valid] - f_orig[:, valid]) / sd[valid]
    out = []
    for row in range(len(dz)):
        a, b = f_orig[row, valid], f_rewrite[row, valid]
        rho = (spearmanr(a, b).statistic
               if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan)
        out.append((np.mean(np.abs(dz[row])),
                    np.mean(np.abs(dz[row]) > 0.5), rho))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    corpus = pd.read_csv(CORPUS, encoding="utf-8-sig")
    corpus["subject"] = corpus["subject"].fillna("").astype(str)
    corpus["body"] = corpus["body"].fillna("").astype(str)
    corpus["text"] = corpus["subject"] + "\n\n" + corpus["body"]
    corpus["label"] = corpus["label"].astype(int)

    seeds = pd.read_csv(SEEDS, encoding="utf-8-sig")
    seed_groups = set(seeds["group_id"].astype(str))
    rng = np.random.default_rng(SEED)
    corpus = split_groups(corpus, seed_groups, rng)

    train = corpus.loc[corpus["split"] == "train"].reset_index(drop=True)
    val = corpus.loc[corpus["split"] == "val"].reset_index(drop=True)
    test = corpus.loc[corpus["split"] == "test"].reset_index(drop=True)
    assert not (set(train["group_id"]) & seed_groups)
    assert not (set(val["group_id"]) & seed_groups)

    for frame in (train, val, test):
        frame["d3"] = [d3_features(t) for t in frame["text"]]
        frame["d4"] = [d4_features(t) for t in frame["text"]]

    pairs = pd.read_csv(args.pairs, encoding="utf-8-sig")
    pairs["text_original"] = pairs["subject_original"].fillna("") + "\n\n" + pairs["body_original"].fillna("")
    pairs["text_rewrite"] = pairs["subject_rewrite"].fillna("") + "\n\n" + pairs["body_rewrite"].fillna("")
    work = pd.DataFrame({
        "text": pairs["text_original"], "d3": [d3_features(t) for t in pairs["text_original"]],
        "d4": [d4_features(t) for t in pairs["text_original"]],
    })
    work_rw = pd.DataFrame({
        "text": pairs["text_rewrite"], "d3": [d3_features(t) for t in pairs["text_rewrite"]],
        "d4": [d4_features(t) for t in pairs["text_rewrite"]],
    })

    detectors = {name: make_detector(name).fit_select(train, val) for name in ["D1", "D2", "D3", "D4", "D5"]}

    metrics, drift_rows, seed_drift_rows, pair_rows = [], [], [], []
    boot_rng = np.random.default_rng(SEED + 1)

    for name, det in detectors.items():
        s_val = det.score(val)
        thr = np.quantile(s_val[val["label"].to_numpy() == 1], 0.10)
        pred_val = s_val >= thr
        val_tpr = pred_val[val["label"].to_numpy() == 1].mean()
        val_fpr = pred_val[val["label"].to_numpy() == 0].mean()

        s_test = det.score(test)
        y_test = test["label"].to_numpy()
        test_auc = roc_auc_score(y_test, s_test)
        pred_test = s_test >= thr
        test_tpr = pred_test[y_test == 1].mean()
        test_fpr = pred_test[y_test == 0].mean()
        z_sd = s_test[y_test == 1].std(ddof=1)

        s_orig = det.score(work)
        s_rw = det.score(work_rw)
        p_orig, p_rw = s_orig >= thr, s_rw >= thr
        d_orig = (s_rw - s_orig) / z_sd
        pair_rows.append(pd.DataFrame({
            "seed_id": pairs["seed_id"], "condition": pairs["condition"],
            "detector": name, "score_original": s_orig, "score_rewrite": s_rw,
            "detect_original": p_orig.astype(int), "detect_rewrite": p_rw.astype(int),
            "score_drift_z": d_orig,
        }))

        # feature drift (z-scored on train features; rare sparse features dropped)
        fam = pairs.copy()
        if name in ("D1", "D2"):
            Xtr = det.vec.transform(train["text"])
            sd, keep = sparse_stats(Xtr)
            drift = drift_for_rows(det.vec.transform(work["text"]),
                                   det.vec.transform(work_rw["text"]), sd, keep)
        elif name == "D5":
            Xtr = det.vec.transform(train["text"])
            Ztr = det.svd.transform(Xtr)
            sd = np.maximum(Ztr.std(axis=0, ddof=1), 1e-9)
            drift = dense_drift(det.svd.transform(det.vec.transform(work["text"])),
                                det.svd.transform(det.vec.transform(work_rw["text"])), sd)
        else:
            key = "d3" if name == "D3" else "d4"
            cols = det.cols
            Ftr = np.array([[r[k] for k in cols] for r in train[key]])
            Fo = np.array([[r[k] for k in cols] for r in work[key]])
            Fr = np.array([[r[k] for k in cols] for r in work_rw[key]])
            sd = np.maximum(Ftr.std(axis=0, ddof=1), 1e-9)
            drift = dense_drift(Fo, Fr, sd)
        fam["abs_dz"] = [d[0] for d in drift]
        fam["prop_gt05"] = [d[1] for d in drift]
        fam["rho"] = [d[2] for d in drift]

        for cond in CONDS:
            mask = pairs["condition"] == cond
            sub = pairs.loc[mask]
            seeds_c = sub["seed_id"].to_numpy()
            det_delta = p_rw[mask].astype(float) - p_orig[mask].astype(float)
            flip = (p_orig[mask] & ~p_rw[mask]).astype(float)
            dz = d_orig[mask]

            per_seed = (pd.DataFrame({"seed": seeds_c, "det_delta": det_delta,
                                      "flip": flip, "dz": dz})
                        .groupby("seed").mean())
            m_d, lo_d, hi_d, p_d = boot_mean(per_seed["det_delta"], boot_rng)
            m_f, lo_f, hi_f, p_f = boot_mean(per_seed["flip"], boot_rng)
            m_z, lo_z, hi_z, p_z = boot_mean(per_seed["dz"], boot_rng)
            metrics.append({
                "detector": name, "condition": cond,
                "n_pairs": int(mask.sum()), "n_seeds": int(per_seed.shape[0]),
                "original_detection_rate": p_orig[mask].mean(),
                "detection_rate": p_rw[mask].mean(),
                "evasion_rate": 1 - p_rw[mask].mean(),
                "detection_delta": m_d, "detection_delta_lo": lo_d, "detection_delta_hi": hi_d,
                "detection_delta_p": p_d,
                "flip_rate": m_f, "flip_lo": lo_f, "flip_hi": hi_f, "flip_p": p_f,
                "score_drift_z": m_z, "drift_lo": lo_z, "drift_hi": hi_z, "drift_p": p_z,
                "selected_C": det.C, "val_auc": det.val_auc,
                "threshold": thr, "val_fpr_at_tpr90": val_fpr, "val_tpr": val_tpr,
                "test_auc": test_auc, "test_fpr": test_fpr, "test_tpr": test_tpr,
                "test_spam_score_sd": z_sd,
            })

            fam_c = fam.loc[mask]
            per_seed_d = (pd.DataFrame({"seed": fam_c["seed_id"],
                                        "abs_dz": fam_c["abs_dz"],
                                        "prop": fam_c["prop_gt05"],
                                        "rho": fam_c["rho"]})
                          .groupby("seed").mean())
            m_a, lo_a, hi_a, _ = boot_mean(per_seed_d["abs_dz"], boot_rng)
            drift_rows.append({
                "detector": name, "condition": cond,
                "n_pairs": int(mask.sum()), "n_seeds": int(per_seed_d.shape[0]),
                "mean_abs_dz": m_a, "abs_dz_lo": lo_a, "abs_dz_hi": hi_a,
                "prop_gt_0_5sd": per_seed_d["prop"].mean(),
                "spearman_rho": np.nanmean(per_seed_d["rho"]),
            })
            seed_drift_rows.extend({
                "detector": name, "condition": cond, "seed": seed, "abs_dz": value,
            } for seed, value in per_seed_d["abs_dz"].items())

    metrics = pd.DataFrame(metrics)
    for col, pcol in [("detection_delta", "detection_delta_p"),
                      ("flip_rate", "flip_p"), ("score_drift_z", "drift_p")]:
        metrics[f"q_{col}"] = bh_q(metrics[pcol].to_numpy())

    drift = pd.DataFrame(drift_rows)
    seed_drift = pd.DataFrame(seed_drift_rows)
    neut = (seed_drift.loc[seed_drift["condition"] == "neutral"]
            .set_index(["detector", "seed"])["abs_dz"])
    deltas = []
    for row in drift.itertuples():
        vals = (seed_drift.loc[(seed_drift["detector"] == row.detector)
                               & (seed_drift["condition"] == row.condition)]
                .set_index("seed")["abs_dz"])
        base = neut.loc[row.detector]
        common = vals.index.intersection(base.index)
        m, lo, hi, p = boot_mean(vals.loc[common] - base.loc[common], boot_rng)
        deltas.append({"delta_vs_neutral": m, "delta_lo": lo,
                       "delta_hi": hi, "delta_p": p})
    drift = pd.concat([drift, pd.DataFrame(deltas)], axis=1)
    drift["q_delta_vs_neutral"] = bh_q(drift["delta_p"].to_numpy())
    pairs_out = pd.concat(pair_rows, ignore_index=True)

    # paired family ordering: D3 vs D2 evasion difference, complete pairs only
    complete_seeds = set(pairs.loc[pairs["complete_pair"] == 1, "seed_id"])
    base = pairs_out.loc[pairs_out["seed_id"].isin(complete_seeds)]
    base = base.groupby(["seed_id", "condition", "detector"]).agg(
        evasion=("detect_rewrite", lambda s: 1 - s.mean())).reset_index()
    wide = base.pivot_table(index=["seed_id", "condition"], columns="detector",
                            values="evasion")
    order_rows = []
    for cond in CONDS + ["all"]:
        sub = wide if cond == "all" else wide.xs(cond, level="condition")
        for hi, lo in [("D3", "D2"), ("D3", "D1"), ("D3", "D4"), ("D3", "D5"),
                       ("D1", "D2"), ("D4", "D2"), ("D5", "D2")]:
            diff = (sub[hi] - sub[lo]).dropna().groupby(level=0).mean()
            m, l, h, p = boot_mean(diff, boot_rng)
            order_rows.append({"condition": cond, "contrast": f"{hi}-{lo}",
                               "n_seeds": len(diff), "mean_diff": m,
                               "lo": l, "hi": h, "p": p})
    order = pd.DataFrame(order_rows)
    order["q"] = bh_q(order["p"].to_numpy())

    OUT.mkdir(exist_ok=True)
    tag = args.tag
    metrics.round(6).to_csv(OUT / f"detector_metrics{tag}.csv", index=False, encoding="utf-8-sig")
    drift.round(6).to_csv(OUT / f"family_drift{tag}.csv", index=False, encoding="utf-8-sig")
    pairs_out.round(6).to_csv(OUT / f"detector_pair_scores{tag}.csv", index=False, encoding="utf-8-sig")
    order.round(6).to_csv(OUT / f"detector_family_order{tag}.csv", index=False, encoding="utf-8-sig")
    if not tag:
        corpus[["doc_id", "group_id", "label", "split"]].to_csv(
            OUT / "detector_split.csv", index=False, encoding="utf-8-sig")

    design = f"""# 检测器实现（预注册 §6）

- 脚本：`e2_detectors.py`，seed={SEED}，bootstrap={N_BOOT}，按 seed 聚类
- 训练语料：`samples/clean_spamassassin.csv`（spam {int((corpus['label']==1).sum())} / ham {int((corpus['label']==0).sum())}）
- 切分：`group_id` 分组；200 个 E2 种子所在 {len(seed_groups)} 组（{int(corpus['is_seed_group'].sum())} 行）强制进测试集；其余组按标签分层、按行数配额 60/20/20；train={len(train)}、val={len(val)}、test={len(test)}
- D4 词典：环境无 nltk/vader/textblob，使用脚本内冻结的正/负情感词表 + 项目既有 URGENCY/WARMTH/threat 词表；该实现细节在任何检测器结果产出前确定
- 超参数：验证集 AUC 选 C；阈值 = 验证集 spam 分数第 10 百分位（TPR=.90），测试集与改写共用
- 漂移：特征在训练集上 z 化；mean |Δz|、|Δz|>0.5 SD 比例、原文-改写 Spearman ρ；稀疏特征在非零并集上计算，训练文档频率 <1% 的特征因 z 分不稳定而排除；D5 用其 200 维 SVD 特征
- 漂移的显著性检验用「条件 − neutral」的 seed 内配对差；检测率/flip/分数漂移用「改写 − 原文」配对差
- 族间排序：完整配对 seed 上的 evasion 配对差，bootstrap 95% CI；BH-FDR 在每个结局族内（5 条件 × 5 检测器）
"""
    (OUT / f"DETECTOR_DESIGN{tag}.md").write_text(design, encoding="utf-8")

    print(metrics[["detector", "condition", "detection_rate", "evasion_rate",
                   "flip_rate", "score_drift_z", "test_fpr"]].round(3).to_string(index=False))
    print(f"\nsplits: train={len(train)} val={len(val)} test={len(test)} "
          f"(seed-group rows={int(corpus['is_seed_group'].sum())})")
    print(f"wrote {OUT.relative_to(ROOT)}/detector_*{tag}.csv, family_drift{tag}.csv, DETECTOR_DESIGN{tag}.md")


if __name__ == "__main__":
    main()
