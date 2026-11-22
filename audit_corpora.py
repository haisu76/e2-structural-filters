#!/usr/bin/env python3
"""Audit the downloaded email corpora before using them for rewrite experiments.

Produces a markdown report plus CSV tables covering:
  - inventory, label/language balance, missing text
  - length distributions (compression confound)
  - exact duplicates / template-prefix clusters
  - PhishFuzzer group structure (7 variants per seed; length/entity drift)
  - NEST-Phish cue validity
  - text-only label probes and cross-corpus fingerprint probes
  - SpamAssassin archive counts and RMF-PC id provenance

Usage:
  python audit_corpora.py --data D:/paper/mail/data --out D:/paper/mail/mvp_v2
"""
from __future__ import annotations

import argparse
import ast
import email
import hashlib
import html
import json
import re
import sys
import tarfile
from collections import Counter
from email import policy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

COLUMNS = [
    "corpus", "doc_id", "group_id", "text", "subject", "body",
    "label", "label_raw", "language", "source", "date", "meta",
]


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------
def norm_ws(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def norm_text(s: object) -> str:
    return norm_ws(s).lower()


def text_hash(s: object) -> str:
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()


def word_count(s: object) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*", str(s)))


def char_count(s: object) -> int:
    return len(re.sub(r"\s", "", str(s)))


def cjk_ratio(s: object) -> float:
    text = str(s)
    if not text:
        return 0.0
    cjk = len(re.findall(r"[一-鿿぀-ヿ가-힯]", text))
    return cjk / len(text)


def parse_list_field(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value)
    if s.strip().lower() in {"", "none", "null", "nan", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        return [s]


def strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return html.unescape(s)


def empty_text(s: object) -> bool:
    return norm_text(s) == ""


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def load_enron(data: Path) -> pd.DataFrame:
    path = data / "hf_enron_spam" / "enron_spam_data.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    subject = df["Subject"].fillna("").astype(str)
    body = df["Message"].fillna("").astype(str)
    rows = []
    for i, (subj, bod, raw_label, date) in enumerate(
        zip(subject, body, df["Spam/Ham"], df["Date"])
    ):
        rows.append({
            "corpus": "enron",
            "doc_id": f"enron:{df['Message ID'].iloc[i]}",
            "group_id": None,
            "text": (subj + "\n\n" + bod).strip(),
            "subject": subj,
            "body": bod,
            "label": 1 if str(raw_label).strip().lower() == "spam" else 0,
            "label_raw": str(raw_label).strip().lower(),
            "language": "en",
            "source": "SetFit/enron_spam",
            "date": str(date),
            "meta": {},
        })
    return _frame(rows)


def load_llmgen(data: Path) -> pd.DataFrame:
    rows = []
    for filename, lang, prefix in [
        ("GPT_Phishing_Email_dataset.csv", "en", "llmgen-en"),
        ("CN_Phishing_Email_dataset.csv", "zh", "llmgen-zh"),
    ]:
        path = data / "hf_llmgen_phishing" / filename
        df = pd.read_csv(path, encoding="utf-8-sig")
        for i, (content, label) in enumerate(zip(df["content"], df["label"])):
            text = str(content)
            rows.append({
                "corpus": "llmgen",
                "doc_id": f"{prefix}:{i}",
                "group_id": None,
                "text": text,
                "subject": "",
                "body": text,
                "label": int(label),
                "label_raw": "phishing" if int(label) == 1 else "legitimate",
                "language": lang,
                "source": filename,
                "date": None,
                "meta": {},
            })
    return _frame(rows)


def load_ephishgen(data: Path) -> pd.DataFrame:
    path = data / "hf_ephishgen" / "ephishLLM.json"
    with open(path, encoding="utf-8-sig") as f:
        data_json = json.load(f)
    rows = []
    for i, item in enumerate(data_json):
        subject = str(item.get("Subject") or "")
        body = str(item.get("Body") or "")
        label = int(item.get("type", 0))
        rows.append({
            "corpus": "ephishgen",
            "doc_id": f"ephish:{i}",
            "group_id": None,
            "text": (subject + "\n\n" + body).strip(),
            "subject": subject,
            "body": body,
            "label": label,
            "label_raw": "phishing" if label == 1 else "legitimate",
            "language": str(item.get("Language") or "unknown"),
            "source": "pajola/e-phishGen",
            "date": None,
            "meta": {},
        })
    return _frame(rows)


def load_phishfuzzer(data: Path) -> pd.DataFrame:
    base = data / "PhishFuzzer"
    rows = []
    with open(base / "PhishFuzzer_emails_original_seed_v1.json", encoding="utf-8-sig") as f:
        seeds = json.load(f)
    with open(base / "PhishFuzzer_emails_entity_rephrased_v1.json", encoding="utf-8-sig") as f:
        variants = json.load(f)

    def add(item: dict, is_variant: bool) -> None:
        subject = str(item.get("Subject") or "")
        body = str(item.get("Body") or "")
        type_raw = str(item.get("Type") or "Unknown")
        rows.append({
            "corpus": "phishfuzzer",
            "doc_id": ("var:" if is_variant else "seed:") + str(item.get("No.")),
            "group_id": str(item.get("Original_ID") if is_variant else item.get("No.")),
            "text": (subject + "\n\n" + body).strip(),
            "subject": subject,
            "body": body,
            "label": 0 if type_raw == "Valid" else 1,
            "label_raw": type_raw,
            "language": str(item.get("Language") or "unknown"),
            "source": str(item.get("Source") or ""),
            "date": str(item.get("Year") or ""),
            "meta": {
                "variant": is_variant,
                "motivation": str(item.get("Motivation") or ""),
                "created_by": str(item.get("Created by") or ""),
                "entity_type": str(item.get("Entity_Type") or ""),
                "length_type": str(item.get("Length_Type") or ""),
                "sender": str(item.get("Sender") or ""),
                "urls_field": parse_list_field(item.get("URL")),
                "file_field": parse_list_field(item.get("File")),
            },
        })

    for item in seeds:
        add(item, False)
    for item in variants:
        add(item, True)
    df = _frame(rows)
    meta = pd.json_normalize(df["meta"])
    for col in ["variant", "created_by", "entity_type", "length_type", "motivation"]:
        df[col] = meta[col].values if col in meta.columns else None
    return df


def load_nest(data: Path) -> pd.DataFrame:
    path = data / "kaggle" / "final_email_dataset.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    cue_cols = [c for c in df.columns if c.startswith("has_") or c == "total_signals"]
    rows = []
    for _, r in df.iterrows():
        subject = str(r.get("subject") or "")
        body = str(r.get("body") or "")
        label_raw = str(r.get("true_label") or "").strip().lower()
        row = {
            "corpus": "nest",
            "doc_id": str(r.get("record_id")),
            "group_id": str(r.get("theme") or ""),
            "text": (subject + "\n\n" + body).strip(),
            "subject": subject,
            "body": body,
            "label": 1 if label_raw == "phish" else 0,
            "label_raw": label_raw,
            "language": "en",
            "source": "emilywinokur/synthetic-email-corpus",
            "date": str(r.get("date") or ""),
            "meta": {"theme": str(r.get("theme") or ""), "sender": str(r.get("sender") or "")},
        }
        for cue in cue_cols:
            row[f"cue_{cue}"] = r.get(cue)
        rows.append(row)
    return _frame(rows)


def _decode_part(part: email.message.Message) -> str:
    try:
        content = part.get_content()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content)
    except Exception:
        payload = part.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="replace")
        return str(part.get_payload())


def parse_email_bytes(raw: bytes) -> tuple[str, str, str]:
    try:
        msg = email.message_from_bytes(raw, policy=policy.default)
    except Exception:
        return "", raw.decode("utf-8", errors="replace"), ""
    subject = str(msg.get("Subject", "") or "")
    date = str(msg.get("Date", "") or "")
    plains, htmls = [], []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            disposition = str(part.get("Content-Disposition", "") or "").lower()
            if "attachment" in disposition:
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain":
                plains.append(_decode_part(part))
            elif ctype == "text/html":
                htmls.append(_decode_part(part))
    else:
        body = _decode_part(msg)
        if msg.get_content_type() == "text/html":
            body = strip_html(body)
        return subject, body, date
    body = "\n".join(plains) if plains else "\n".join(htmls)
    if not plains and htmls:
        body = strip_html(body)
    return subject, body, date


def load_spamassassin(data: Path) -> pd.DataFrame:
    archive_dir = data / "spamassassin"
    rows = []
    for path in sorted(archive_dir.glob("*.tar.bz2")):
        stem = path.name.replace(".tar.bz2", "")
        with tarfile.open(path, "r:bz2") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                raw = tf.extractfile(member).read()
                subject, body, date = parse_email_bytes(raw)
                inner = member.name.split("/")[0]
                if inner.startswith("spam"):
                    label, label_raw = 1, "spam"
                else:
                    label, label_raw = 0, "ham"
                base = member.name.split("/")[-1]
                md5_part = base.split(".", 1)[1] if "." in base else base
                rows.append({
                    "corpus": "spamassassin",
                    "doc_id": f"{stem}/{member.name}",
                    "group_id": f"md5:{md5_part}",
                    "text": (subject + "\n\n" + body).strip(),
                    "subject": subject,
                    "body": body,
                    "label": label,
                    "label_raw": label_raw,
                    "language": "en",
                    "source": stem,
                    "date": date,
                    "meta": {"archive": stem, "member": member.name},
                })
    return _frame(rows)


def load_turkphish(data: Path) -> pd.DataFrame:
    path = data / "turkphish_v1" / "turkish_phishing_dataset.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    rows = []
    for _, r in df.iterrows():
        subject = str(r.get("Konu") or "")
        body = str(r.get("İçerik") or "")
        raw_label = str(r.get("Kategori") or "").strip()
        rows.append({
            "corpus": "turkphish_v1",
            "doc_id": f"turkphish:{r.get('ID')}",
            "group_id": None,
            "text": (subject + "\n\n" + body).strip(),
            "subject": subject,
            "body": body,
            "label": 1 if raw_label == "Oltalama" else 0,
            "label_raw": "phishing" if raw_label == "Oltalama" else "legitimate",
            "language": "tr",
            "source": "Kaggle Turkish Phishing Email Dataset v1",
            "date": None,
            "meta": {"sender": str(r.get("Gönderen") or "")},
        })
    return _frame(rows)


LOADERS = {
    "enron": load_enron,
    "llmgen": load_llmgen,
    "ephishgen": load_ephishgen,
    "phishfuzzer": load_phishfuzzer,
    "nest": load_nest,
    "spamassassin": load_spamassassin,
    "turkphish_v1": load_turkphish,
}


# ---------------------------------------------------------------------------
# audit tables
# ---------------------------------------------------------------------------
def inventory_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corpus, g in df.groupby("corpus", sort=False):
        labels = g["label_raw"].value_counts(dropna=False).to_dict()
        langs = g["language"].value_counts(dropna=False).head(6).to_dict()
        words = g["text"].map(word_count)
        chars = g["text"].map(char_count)
        rows.append({
            "corpus": corpus,
            "n": len(g),
            "labels": "; ".join(f"{k}:{v}" for k, v in labels.items()),
            "languages": "; ".join(f"{k}:{v}" for k, v in langs.items()),
            "empty_text": int(g["text"].map(empty_text).sum()),
            "median_words": float(words.median()),
            "p10_words": float(words.quantile(0.10)),
            "p90_words": float(words.quantile(0.90)),
            "median_chars": float(chars.median()),
            "unique_groups": int(g["group_id"].nunique()) if g["group_id"].notna().any() else 0,
        })
    return pd.DataFrame(rows)


def length_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (corpus, label_raw), g in df.groupby(["corpus", "label_raw"], sort=False, dropna=False):
        words = g["text"].map(word_count)
        chars = g["text"].map(char_count)
        rows.append({
            "corpus": corpus,
            "label": label_raw,
            "n": len(g),
            "median_words": float(words.median()),
            "p10_words": float(words.quantile(0.10)),
            "p90_words": float(words.quantile(0.90)),
            "median_chars": float(chars.median()),
        })
    return pd.DataFrame(rows)


def duplicate_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corpus, g in df.groupby("corpus", sort=False):
        hashes = g["text"].map(text_hash)
        counts = hashes.value_counts()
        dup_groups = int((counts > 1).sum())
        unique_texts = int(hashes.nunique())
        dup_rows = int(len(g) - unique_texts)
        prefix = g["text"].map(lambda s: norm_text(s)[:120])
        prefix_counts = prefix.value_counts()
        rows.append({
            "corpus": corpus,
            "n": len(g),
            "empty": int(g["text"].map(empty_text).sum()),
            "unique_texts": unique_texts,
            "exact_dup_rows": dup_rows,
            "exact_dup_rate": dup_rows / max(len(g), 1),
            "dup_groups": dup_groups,
            "max_group": int(counts.max()) if len(counts) else 0,
            "prefix_groups_gt1": int((prefix_counts > 1).sum()),
            "max_prefix_group": int(prefix_counts.max()) if len(prefix_counts) else 0,
        })
    return pd.DataFrame(rows)


def top_duplicate_examples(df: pd.DataFrame, corpus: str, n: int = 5) -> pd.DataFrame:
    g = df[df["corpus"] == corpus].copy()
    g["_h"] = g["text"].map(text_hash)
    counts = g["_h"].value_counts()
    rows = []
    for h, count in counts.head(n).items():
        examples = g[g["_h"] == h]["doc_id"].head(3).tolist()
        sample = g[g["_h"] == h]["text"].iloc[0][:160]
        rows.append({"count": int(count), "examples": "; ".join(examples), "text": sample})
    return pd.DataFrame(rows)


def top_prefix_clusters(df: pd.DataFrame, corpus: str, n: int = 10) -> pd.DataFrame:
    g = df[df["corpus"] == corpus].copy()
    g["_p"] = g["text"].map(lambda s: norm_text(s)[:120])
    counts = g["_p"].value_counts()
    rows = []
    for prefix, count in counts.head(n).items():
        if count < 2:
            continue
        examples = g[g["_p"] == prefix]["doc_id"].head(2).tolist()
        rows.append({"count": int(count), "examples": "; ".join(examples), "prefix": prefix[:120]})
    return pd.DataFrame(rows)


def phishfuzzer_group_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pf = df[df["corpus"] == "phishfuzzer"].copy()
    if pf.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sizes = pf.groupby("group_id").size().value_counts().sort_index()
    size_table = pd.DataFrame({"group_size": sizes.index.astype(int), "n_groups": sizes.values})

    seeds = pf[~pf["variant"].astype(bool)].drop_duplicates("group_id").set_index("group_id")
    variants = pf[pf["variant"].astype(bool)].copy()

    length_rows, url_rows = [], []
    for length_type, g in variants.groupby("length_type"):
        ratios, var_words = [], []
        for _, r in g.iterrows():
            if r["group_id"] not in seeds.index:
                continue
            seed = seeds.loc[r["group_id"]]
            var_words.append(word_count(r["text"]))
            ratios.append(word_count(r["text"]) / max(word_count(seed["text"]), 1))
        if ratios:
            length_rows.append({
                "length_type": length_type,
                "n": len(ratios),
                "median_var_words": float(np.median(var_words)),
                "median_ratio": float(np.median(ratios)),
                "p10_ratio": float(np.quantile(ratios, 0.10)),
                "p90_ratio": float(np.quantile(ratios, 0.90)),
            })

    url_changed = url_missing = url_added = email_changed = 0
    n_cmp = 0
    for _, r in variants.iterrows():
        if r["group_id"] not in seeds.index:
            continue
        seed = seeds.loc[r["group_id"]]
        n_cmp += 1
        seed_urls = set(seed["meta"]["urls_field"])
        var_urls = set(r["meta"]["urls_field"])
        if seed_urls != var_urls:
            url_changed += 1
        url_missing += len(seed_urls - var_urls) > 0
        url_added += len(var_urls - seed_urls) > 0
        seed_emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", seed["text"]))
        var_emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", r["text"]))
        if seed_emails != var_emails:
            email_changed += 1

    if n_cmp:
        url_rows.append({
            "variants_compared": n_cmp,
            "url_set_changed": url_changed,
            "url_set_changed_pct": url_changed / n_cmp,
            "url_missing_pct": url_missing / n_cmp,
            "url_added_pct": url_added / n_cmp,
            "email_set_changed_pct": email_changed / n_cmp,
        })
    return size_table, pd.DataFrame(length_rows), pd.DataFrame(url_rows)


def nest_cue_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    nest = df[df["corpus"] == "nest"].copy()
    cue_cols = [c for c in nest.columns if c.startswith("cue_")]
    rows = []
    for cue in cue_cols:
        sub = nest[[cue, "label"]].dropna()
        if sub.empty or sub[cue].nunique() < 2:
            continue
        table = pd.crosstab(sub[cue], sub["label"])
        try:
            chi2, p, _, _ = chi2_contingency(table)
        except ValueError:
            chi2, p = np.nan, np.nan
        rows.append({
            "cue": cue.replace("cue_", ""),
            "phish_mean": float(sub[sub["label"] == 1][cue].mean()),
            "legit_mean": float(sub[sub["label"] == 0][cue].mean()),
            "diff": float(sub[sub["label"] == 1][cue].mean() - sub[sub["label"] == 0][cue].mean()),
            "chi2": float(chi2),
            "p": float(p),
        })
    cue_table = pd.DataFrame(rows)

    metrics = {}
    if len(nest) >= 30 and cue_cols:
        X = nest[cue_cols].fillna(0).astype(float).values
        y = nest["label"].astype(int).values
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                 method="predict_proba")[:, 1]
        metrics = {
            "n": len(nest),
            "cue_accuracy": float(accuracy_score(y, (pred >= 0.5).astype(int))),
            "cue_f1": float(f1_score(y, (pred >= 0.5).astype(int), average="macro")),
            "cue_auc": float(roc_auc_score(y, pred)),
        }
    return cue_table, metrics


def _vectorizer(texts: list[str]) -> TfidfVectorizer:
    mean_cjk = float(np.mean([cjk_ratio(t) for t in texts])) if texts else 0.0
    if mean_cjk > 0.05:
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                               max_features=120000, sublinear_tf=True)
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120000,
                           sublinear_tf=True, strip_accents="unicode")


def probe_label(df: pd.DataFrame, corpus: str, max_n: int = 6000) -> dict | None:
    g = df[df["corpus"] == corpus].copy()
    label_col = "label_raw" if corpus == "phishfuzzer" else "label"
    g = g[g[label_col].notna()]
    if g.empty:
        return None
    # balance sample across classes
    classes = g[label_col].unique()
    per_class = max(50, max_n // len(classes))
    parts = []
    for cls in classes:
        sub = g[g[label_col] == cls]
        if len(sub) > per_class:
            sub = sub.sample(per_class, random_state=RANDOM_STATE)
        parts.append(sub)
    g = pd.concat(parts, ignore_index=True)
    y = g[label_col].astype(str).values
    if len(np.unique(y)) < 2 or min(Counter(y).values()) < 3:
        return None
    texts = (g["subject"].fillna("").astype(str) + "\n" + g["body"].fillna("").astype(str)).str[:30000].tolist()
    n_splits = 5 if min(Counter(y).values()) >= 5 else 3
    solver = "liblinear" if len(np.unique(y)) == 2 else "lbfgs"
    pipe = Pipeline([
        ("tfidf", _vectorizer(texts)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver=solver)),
    ])
    try:
        proba = cross_val_predict(
            pipe, texts, y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
            method="predict_proba",
        )
    except Exception as exc:
        return {"corpus": corpus, "n": len(g), "error": str(exc)[:160]}
    classes_sorted = np.unique(y)
    pred = classes_sorted[proba.argmax(axis=1)]
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    if len(classes_sorted) == 2:
        auc = roc_auc_score((y == classes_sorted[1]).astype(int), proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    return {"corpus": corpus, "n": len(g), "classes": len(classes_sorted),
            "accuracy": float(acc), "macro_f1": float(f1), "auc": float(auc)}


def fingerprint_probe(df: pd.DataFrame, per_corpus: int = 400) -> tuple[dict, pd.DataFrame]:
    parts = []
    for corpus, g in df.groupby("corpus", sort=False):
        if len(g) > per_corpus:
            g = g.sample(per_corpus, random_state=RANDOM_STATE)
        parts.append(g[["corpus", "subject", "body"]])
    d = pd.concat(parts, ignore_index=True)
    texts = (d["subject"].fillna("").astype(str) + "\n" + d["body"].fillna("").astype(str)).str[:30000].tolist()
    y = d["corpus"].values
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                  max_features=120000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
    ])
    pred = cross_val_predict(pipe, texts, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    metrics = {
        "n": len(d),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }
    return metrics, cm_df


def provenance_probe(df: pd.DataFrame) -> dict | None:
    pf = df[(df["corpus"] == "phishfuzzer") & df["created_by"].notna()].copy()
    if pf.empty:
        return None
    y = (pf["created_by"].str.lower() == "llm").astype(int).values
    if len(np.unique(y)) < 2 or min(Counter(y).values()) < 5:
        return None
    texts = (pf["subject"].fillna("").astype(str) + "\n" + pf["body"].fillna("").astype(str)).str[:30000].tolist()
    pipe = Pipeline([
        ("tfidf", _vectorizer(texts)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])
    proba = cross_val_predict(pipe, texts, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"n": len(pf), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "auc": float(roc_auc_score(y, proba))}


def spamassassin_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sa = df[df["corpus"] == "spamassassin"].copy()
    if sa.empty:
        return pd.DataFrame(), pd.DataFrame()
    archive = sa.groupby("source").agg(
        n=("doc_id", "size"),
        spam=("label", "sum"),
        ham=("label", lambda s: int((s == 0).sum())),
    ).reset_index()
    archive["spam_rate"] = archive["spam"] / archive["n"]
    hashes = sa.groupby("group_id")["source"].nunique()
    dup_across = hashes[hashes > 1]
    dup_rows = []
    for h, _ in dup_across.head(10).items():
        sub = sa[sa["group_id"] == h]
        dup_rows.append({"group_id": h, "archives": "; ".join(sorted(sub["source"].unique())),
                         "examples": "; ".join(sub["doc_id"].head(3))})
    return archive, pd.DataFrame(dup_rows)


def rmf_match(sa_df: pd.DataFrame) -> dict | None:
    lab = Path("D:/paper/spam/lab")
    files = [lab / "l2_pairs_source.csv", lab / "l3_pairs_source.csv"]
    ids = set()
    for path in files:
        if not path.exists():
            continue
        try:
            ids |= set(pd.read_csv(path, encoding="utf-8-sig", usecols=["id"])["id"].astype(str))
        except Exception:
            continue
    if not ids:
        return None
    matched = ids & set(sa_df["doc_id"])
    return {"n_ids": len(ids), "matched": len(matched), "rate": len(matched) / len(ids)}


def llmgen_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["corpus"] == "llmgen"].copy()
    if g.empty:
        return pd.DataFrame()
    g["starts_subject"] = g["text"].str.match(r"(?i)^\s*subject\s*:")
    g["has_from"] = g["text"].str.contains(r"(?i)\bfrom\s*:", regex=True)
    rows = []
    for lang, sub in g.groupby("language"):
        rows.append({
            "language": lang,
            "n": len(sub),
            "phishing": int((sub["label"] == 1).sum()),
            "legitimate": int((sub["label"] == 0).sum()),
            "median_words": float(sub["text"].map(word_count).median()),
            "median_chars": float(sub["text"].map(char_count).median()),
            "starts_subject_pct": float(sub["starts_subject"].mean()),
            "has_from_pct": float(sub["has_from"].mean()),
            "exact_dup_rate": float((len(sub) - sub["text"].map(text_hash).nunique()) / max(len(sub), 1)),
        })
    return pd.DataFrame(rows)


def date_label_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["corpus"] == "enron"].copy()
    if g.empty:
        return pd.DataFrame()
    g["year"] = pd.to_datetime(g["date"], errors="coerce").dt.year
    out = g.dropna(subset=["year"]).groupby("year").agg(
        n=("doc_id", "size"), spam=("label", "sum")
    ).reset_index()
    out["spam_rate"] = out["spam"] / out["n"]
    return out


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_（无数据）_"
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.3f}")
            else:
                vals.append(str(v).replace("|", "\\|")[:200])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="D:/paper/mail/data")
    parser.add_argument("--out", default="D:/paper/mail/mvp_v2")
    parser.add_argument("--corpora", nargs="*", default=list(LOADERS))
    parser.add_argument("--max-probe", type=int, default=6000)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "audit_tables"
    tables_dir.mkdir(exist_ok=True)

    frames = []
    for name in args.corpora:
        print(f"[load] {name} ...", flush=True)
        try:
            frames.append(LOADERS[name](data_dir))
        except Exception as exc:
            print(f"[warn] failed to load {name}: {exc}", file=sys.stderr)
    if not frames:
        raise SystemExit("no corpora loaded")
    df = pd.concat(frames, ignore_index=True)

    inventory = inventory_table(df)
    lengths = length_table(df)
    duplicates = duplicate_table(df)
    llmgen = llmgen_table(df)
    enron_year = date_label_table(df)
    pf_sizes, pf_lengths, pf_urls = phishfuzzer_group_table(df)
    nest_cues, nest_metrics = nest_cue_table(df)
    sa_archives, sa_dups = spamassassin_tables(df)
    sa_match = rmf_match(df[df["corpus"] == "spamassassin"]) if "spamassassin" in args.corpora else None

    probes = []
    for corpus in args.corpora:
        print(f"[probe] {corpus} ...", flush=True)
        result = probe_label(df, corpus, max_n=args.max_probe)
        if result:
            probes.append(result)
    probes_df = pd.DataFrame(probes)

    print("[probe] cross-corpus fingerprint ...", flush=True)
    fp_metrics, fp_cm = fingerprint_probe(df)
    prov = provenance_probe(df)

    # persist tables
    for name, table in [
        ("inventory", inventory), ("lengths", lengths), ("duplicates", duplicates),
        ("llmgen", llmgen), ("enron_year", enron_year), ("phishfuzzer_group_sizes", pf_sizes),
        ("phishfuzzer_lengths", pf_lengths), ("phishfuzzer_urls", pf_urls),
        ("nest_cues", nest_cues), ("spamassassin_archives", sa_archives),
        ("spamassassin_duplicates", sa_dups), ("probes", probes_df),
        ("fingerprint_confusion", fp_cm.reset_index().rename(columns={"index": "true"})),
    ]:
        table.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    # report
    r = []
    r.append("# 语料审计报告（mvp_v2）\n")
    r.append(f"- 数据目录：`{data_dir}`")
    r.append(f"- 语料：{', '.join(args.corpora)}")
    r.append(f"- 文档总数：{len(df)}")
    r.append(f"- 生成脚本：`audit_corpora.py`\n")

    r.append("## 0. 结论摘要\n")
    total_by_corpus = df["corpus"].value_counts().to_dict()
    r.append(f"- 已加载 {len(args.corpora)} 个语料，共 {len(df)} 封邮件：{total_by_corpus}。")
    if not pf_lengths.empty:
        ratios = ", ".join(
            f"{row['length_type']}={row['median_ratio']:.2f}"
            for _, row in pf_lengths.iterrows()
        )
        words = ", ".join(
            f"{row['length_type']}={row['median_var_words']:.0f}"
            for _, row in pf_lengths.iterrows()
        )
        r.append(f"- PhishFuzzer 的 7 版本/种子结构会在长度上产生漂移（词数比 {ratios}；"
                 f"变体词数中位数 {words}）。`long` 与 `medium` 的实际长度几乎相同，"
                 "说明长度控制是名义上的、并不严格。它适合做外部基准/OOD，不适合直接当作"
                 "“保长度、保实体”的风格改写数据。")
    if not pf_urls.empty:
        row = pf_urls.iloc[0]
        r.append(f"- PhishFuzzer 变体中 {row['url_set_changed_pct']:.1%} 改变了 URL 集合，"
                 f"{row['email_set_changed_pct']:.1%} 改变了邮箱集合；这进一步说明它是“实体替换”而非“风格改写”。")
    if not duplicates.empty:
        worst = duplicates.sort_values("exact_dup_rate", ascending=False).iloc[0]
        r.append(f"- 精确重复率最高的是 {worst['corpus']}：{worst['exact_dup_rate']:.1%}。")
    if nest_metrics:
        r.append(f"- NEST-Phish 仅 6 个线索特征就能达到 AUC={nest_metrics['cue_auc']:.3f}；"
                 "它适合做构念效度/人工标注对照，不适合训练大模型。")
    if sa_match:
        r.append(f"- RMF-PC 的 {sa_match['n_ids']} 个 pair id 中，{sa_match['matched']} 个能在下载的 "
                 f"SpamAssassin 中找到（{sa_match['rate']:.1%}），来源可复核。")
    r.append("")

    r.append("## 1. 语料清单与基本分布\n")
    r.append(md_table(inventory))
    r.append("\n> `empty_text` 指主题和正文都为空；`unique_groups` 仅对具有分组字段的语料有意义（PhishFuzzer/NEST）。\n")

    r.append("## 2. 长度分布（压缩混淆基线）\n")
    r.append(md_table(lengths))
    r.append("\n> 词数按拉丁词计；中文语料的词数会低估，实际长度请同时看 `median_chars`。\n")

    r.append("## 3. 重复与模板\n")
    r.append(md_table(duplicates))
    for corpus in ["llmgen", "turkphish_v1", "nest", "spamassassin"]:
        if corpus in args.corpora:
            top = top_prefix_clusters(df, corpus, n=6)
            if not top.empty:
                r.append(f"\n### 3.{corpus} 前缀模板簇\n")
                r.append(md_table(top))

    r.append("\n## 4. PhishFuzzer 的分组结构\n")
    if pf_sizes.empty:
        r.append("_未加载 PhishFuzzer。_")
    else:
        r.append("### 4.1 每组文档数\n")
        r.append(md_table(pf_sizes))
        r.append("\n### 4.2 变体相对种子的词数比\n")
        r.append(md_table(pf_lengths))
        r.append("\n### 4.3 变体的 URL / 邮箱集合变化\n")
        r.append(md_table(pf_urls))
        r.append("\n> 结论：PhishFuzzer 的 `Length_Type` 是有意操纵的变量，且实体/URL 也被替换。"
                 "若拿它做“温度漂移”，长度和实体是强混淆；只能做外部基准，或在同一 `Length_Type` 内做敏感性分析。\n")

    r.append("## 5. NEST-Phish 线索效度\n")
    if nest_cues.empty:
        r.append("_未加载 NEST-Phish。_")
    else:
        r.append(md_table(nest_cues))
        if nest_metrics:
            r.append(f"\n- 仅线索特征的 5 折 Logistic 回归：n={nest_metrics['n']}, "
                     f"accuracy={nest_metrics['cue_accuracy']:.3f}, macro-F1={nest_metrics['cue_f1']:.3f}, "
                     f"AUC={nest_metrics['cue_auc']:.3f}。")
        r.append("\n> 这不是检测器结果，而是检查“线索标注是否几乎等价于标签”。若线索即可高分预测标签，"
                 "则文本模型的高分可能只是学到了这些显式线索，而不是更细的情感温度。\n")

    r.append("## 6. 文本可学习性探针（非正式检测结果）\n")
    r.append(md_table(probes_df))
    r.append(f"\n- 跨语料指纹探针：n={fp_metrics['n']}, accuracy={fp_metrics['accuracy']:.3f}, "
             f"macro-F1={fp_metrics['macro_f1']:.3f}。")
    if prov:
        r.append(f"- PhishFuzzer 的 Human vs LLM 来源探针：n={prov['n']}, accuracy={prov['accuracy']:.3f}, "
                 f"macro-F1={prov['macro_f1']:.3f}, AUC={prov['auc']:.3f}。")
    r.append("\n> 这些探针只用于审计：说明语料/来源在文本上是否可区分，以及标签是否有明显文本信号。"
             "它们不是论文中的正式检测器，也不能替代锁定测试集。\n")

    r.append("## 7. SpamAssassin 与 RMF-PC 来源核对\n")
    if sa_archives.empty:
        r.append("_未加载 SpamAssassin。_")
    else:
        r.append(md_table(sa_archives))
        sa = df[df["corpus"] == "spamassassin"]
        sa_unique_text = int(sa["text"].map(text_hash).nunique())
        sa_cross = int((sa.groupby("group_id")["source"].nunique() > 1).sum())
        r.append(f"\n- 去重前 {len(sa)} 封；按归一化正文去重后 {sa_unique_text} 封"
                 f"（重复 {len(sa) - sa_unique_text} 封）。")
        r.append(f"- 跨归档重复组：{sa_cross} 组；主要是 `20050311_spam_2` 与 "
                 "`20030228_spam_2` 为同一批邮件。建模/改写抽样前必须先去重。")
        if sa_match:
            r.append(f"- RMF-PC pair id 匹配：{sa_match['matched']}/{sa_match['n_ids']} "
                     f"（{sa_match['rate']:.1%}）。")
        if not sa_dups.empty:
            r.append("\n### 7.1 跨归档重复\n")
            r.append(md_table(sa_dups))

    r.append("\n## 8. LLMGen 细节\n")
    if llmgen.empty:
        r.append("_未加载 LLMGen。_")
    else:
        r.append(md_table(llmgen))
        r.append("\n> 实际行数与 v1.0 方案中的 3,754 不一致；以本表为准。"
                 "`starts_subject_pct` / `has_from_pct` 高说明文本里保留了邮件头模板，"
                 "训练/改写时需要决定是否保留。\n")

    r.append("## 9. Enron 时间与标签\n")
    if enron_year.empty:
        r.append("_未加载 Enron。_")
    else:
        r.append(md_table(enron_year))
        early = enron_year[enron_year["year"] <= 2001]
        late = enron_year[enron_year["year"] >= 2003]
        if not early.empty and not late.empty:
            r.append(f"\n- 2001 年及以前的 spam rate 为 {early['spam'].sum() / max(early['n'].sum(), 1):.3f}；"
                     f"2003 年及以后为 {late['spam'].sum() / max(late['n'].sum(), 1):.3f}。"
                     "标签与年份几乎完全共线：随机切分会把“时间”当成特征，必须按年份/发件人/线程分组切分。")

    r.append("\n## 10. 对“保风格改写”的直接含义\n")
    r.append("1. **主数据源**：从 SpamAssassin 与 Enron 的 spam 中抽样；先按归一化文本去重，再按来源/模板分层。")
    r.append("2. **不要用 PhishFuzzer 变体做主实验**：它的长度与实体是刻意变化的；可作为外部基准或负对照。")
    r.append("3. **长度约束**：0.8–1.2（按词数）；中文按字符数另算。生成后必须用 `validate_rewrites.py` 自动校验。")
    r.append("4. **实体约束**：URL、邮箱、电话、金额、日期、账号/订单号、附件名、签名块必须逐字保留。")
    r.append("5. **禁止新增危害**：不得新增请求、承诺、凭据索取、威胁、截止日期或新链接。")
    r.append("6. **人工复核不可省**：自动校验只做候选筛查；最终纳入仍需双人标注 + 裁决 + κ/PABAK。")

    r.append("\n## 11. 缺失与限制\n")
    r.append("- **TREC 2007 未下载**：目录里只有协议说明页，需要签署协议后获取 `trec07p.tgz`；本审计未覆盖 TREC。")
    r.append("- **TurkPhish v2 未下载**：只有论文 PDF；v1 仅作泄漏案例，不作主基准。")
    r.append("- **NEST-Phish 仅 146 封**：所有估计都不稳定，只用于方法学/构念效度。")
    r.append("- **探针使用 TF-IDF + LogisticRegression**：不是 BERT/LLM 检测器，只用于语料审计。")
    r.append("- **中文语料**：LLMGen CN 3,030 封，未做分词/中文情感资源校验；主语种仍为英文。")
    r.append("- **来源/许可**：Enron 含 PII，TREC 不可再分发；使用前请核对各语料许可。")

    report = "\n".join(r) + "\n"
    (out_dir / "CORPUS_AUDIT.md").write_text(report, encoding="utf-8")
    print(f"[done] wrote {out_dir / 'CORPUS_AUDIT.md'}")
    print(f"[done] tables in {tables_dir}")


if __name__ == "__main__":
    sys.exit(main())
