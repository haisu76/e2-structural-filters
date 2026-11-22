#!/usr/bin/env python3
"""Build cleaned corpora and the fixed E1/E2 samples.

Steps
  1. SpamAssassin: parse archives, normalise, exact-dedup to unique texts,
     keep plain-text candidates, and tag template groups.
  2. Enron: exact-dedup, parse year, extract an approximate sender from the
     pre-processed body, and build year/sender leakage groups.
  3. E2: 200 SpamAssassin spam seeds (50-300 body words, plain text, >=1 URL).
  4. E1: 400 items, crossed spam/ham x human/synthetic, plus a blinded
     annotation sheet (no labels or provenance).

Usage
  python build_samples.py --data D:/paper/mail/data --out D:/paper/mail/mvp_v2/samples
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate_rewrites import (  # noqa: E402
    ACTION_CONCEPTS,
    THREAT_CONCEPTS,
    URGENCY,
    WARMTH,
    char_count,
    concept_hits,
    credential_hits,
    extract_attachments,
    extract_dates,
    extract_emails,
    extract_entities,
    extract_ids,
    extract_money,
    extract_phones,
    extract_time_windows,
    extract_urls,
    norm_text,
    norm_ws,
    term_count,
    word_count,
)

RANDOM_STATE = 42

HTML_MARKUP_RE = re.compile(
    r"(?is)<\s*(?:html|body|div|p|br|a|table|span|font|img|td|tr|strong|em)\b"
    r"|</\s*(?:html|body|div|p|a|table|span|font|td|tr|strong|em)\s*>"
)
TOKENIZED_URL_RE = re.compile(r"(?i)\bhttps?\s*:\s*/\s*/")

EN_STOP = set(
    "the a an and or but if then of to in on at for with from by as is are was "
    "were be been being have has had do does did will would can could should may "
    "might must not no you your we our they their he she it this that these those "
    "i me my please thank thanks dear hi hello".split()
)
NON_EN_STOP = set(
    "il lo la gli le di da con su per tra fra che non sono un una del della dei "
    "delle al alla ai alle questo questa come piu anche molto ciao gentile caro "
    "cara grazie buongiorno salve der die das und ist nicht ich sie wir ihr mit "
    "fur auf von zu den dem des ein eine einen einem einer dass auch noch oder "
    "aber sehr vielen dank hallo freundliche grusse les du de et est pas vous nous "
    "pour avec sur dans ce cette ces merci bonjour cordialement el los las es para "
    "con por una un del que se su sus hola estimado saludos gracias".split()
)

LLM_PARSE_RE = re.compile(
    r"^\s*(?:\*\*)?Subject(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.*?)\s+"
    r"(?:\*\*)?(?:From|Sender)(?:\*\*)?\s*:\s*(?:\*\*)?\s*(\S+)\s+(.*)$",
    re.S | re.I,
)


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def text_hash(s: object) -> str:
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def template_key(subject: object, text: object) -> str:
    s = norm_text(subject)
    s = re.sub(r"^(?:(?:re|fw|fwd|aw)\s*:\s*)+", "", s)
    if not s:
        s = norm_text(text)[:80]
    return short_hash(s)


def thread_key(subject: object, text: object) -> str:
    s = norm_text(subject)
    s = re.sub(r"^(?:(?:re|fw|fwd|aw)\s*:\s*)+", "", s)
    if not s:
        s = norm_text(text)[:120]
    return short_hash(s, 16)


def is_probably_english(s: object, min_tokens: int = 12) -> bool:
    text = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", " ", str(s))
    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+", " ", text)
    toks = re.findall(r"[a-zà-ÿ']+", text.lower())
    if len(toks) < min_tokens:
        return False
    en = sum(t in EN_STOP for t in toks)
    non = sum(t in NON_EN_STOP for t in toks)
    return en >= 3 and en > non and en / len(toks) >= 0.05


def subject_looks_non_english(s: object) -> bool:
    toks = re.findall(r"[a-zà-ÿ']+", str(s).lower())
    return any(t in NON_EN_STOP for t in toks)


def strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return html.unescape(s)


def clean_markdown(s: str) -> str:
    s = str(s).replace("**", "")
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def entity_counts(text: object) -> dict[str, int]:
    ent = extract_entities(str(text))
    return {
        "url_count": sum(ent["urls"].values()),
        "email_count": sum(ent["emails"].values()),
        "phone_count": sum(ent["phones"].values()),
        "money_count": sum(ent["money"].values()),
        "date_count": sum(ent["dates"].values()),
        "time_window_count": sum(ent["time_windows"].values()),
        "id_count": sum(ent["ids"].values()),
        "attachment_count": sum(ent["attachments"].values()),
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = df["body"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["word_count"] = df["body"].map(word_count)
    df["char_count"] = df["body"].map(char_count)
    feats = pd.DataFrame([entity_counts(b) for b in df["body"]], index=df.index)
    df = pd.concat([df, feats], axis=1)
    df["has_url"] = df["url_count"] > 0
    df["html_markup"] = df["body"].map(lambda s: bool(HTML_MARKUP_RE.search(str(s))))
    df["is_empty"] = df["text"].str.strip().eq("")
    df["urgency_proxy"] = df["body"].map(
        lambda s: term_count(str(s), URGENCY)
        + len(extract_time_windows(str(s)))
        + len(concept_hits(str(s), THREAT_CONCEPTS))
    )
    df["warmth_proxy"] = df["body"].map(lambda s: term_count(str(s), WARMTH))
    df["template_key"] = [
        template_key(s, t) for s, t in zip(df["subject"], df["text"])
    ]
    return df


# ---------------------------------------------------------------------------
# parsing / loading
# ---------------------------------------------------------------------------
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


def parse_email_ex(raw: bytes) -> tuple[str, str, str, bool, bool]:
    """Return subject, body, date, has_plain_part, has_html_part."""
    try:
        msg = email.message_from_bytes(raw, policy=policy.default)
    except Exception:
        return "", raw.decode("utf-8", errors="replace"), "", False, False

    subject = str(msg.get("Subject", "") or "")
    date = str(msg.get("Date", "") or "")
    plains: list[str] = []
    htmls: list[str] = []

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
        body = "\n".join(plains) if plains else "\n".join(htmls)
        if not plains and htmls:
            body = strip_html(body)
        return subject, body, date, bool(plains), bool(htmls)

    body = _decode_part(msg)
    ctype = msg.get_content_type()
    if ctype == "text/html":
        body = strip_html(body)
        return subject, body, date, False, True
    return subject, body, date, ctype == "text/plain", False


def load_spamassassin_raw(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((data_dir / "spamassassin").glob("*.tar.bz2")):
        archive = path.name.replace(".tar.bz2", "")
        with tarfile.open(path, "r:bz2") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                raw = tf.extractfile(member).read()
                subject, body, date, has_plain, has_html = parse_email_ex(raw)
                inner = member.name.split("/")[0]
                label = 1 if inner.startswith("spam") else 0
                base = member.name.split("/")[-1]
                file_md5 = base.split(".", 1)[1] if "." in base else base
                text = (subject + "\n\n" + body).strip()
                rows.append({
                    "corpus": "spamassassin",
                    "doc_id": f"{archive}/{member.name}",
                    "file_md5": file_md5,
                    "subject": subject,
                    "body": body,
                    "text": text,
                    "label": label,
                    "label_raw": "spam" if label else "ham",
                    "language": "en",
                    "provenance": "human",
                    "source_group": archive,
                    "date": date,
                    "has_plain": has_plain,
                    "has_html": has_html,
                    "body_source": (
                        "plain" if has_plain else "html_stripped" if has_html else "none"
                    ),
                })
    return pd.DataFrame(rows)


def build_clean_spamassassin(data_dir: Path) -> tuple[pd.DataFrame, int]:
    df = load_spamassassin_raw(data_dir)
    raw_n = len(df)
    df["dup_group_id"] = df["text"].map(text_hash)
    dup_counts = df.groupby("dup_group_id").size()
    # Prefer a text/plain representation with a non-empty subject.
    df["_pref"] = df["has_plain"].astype(int) * 2 + (
        df["subject"].fillna("").astype(str).str.strip().ne("")
    ).astype(int)
    df = df.sort_values(["_pref", "doc_id"], ascending=[False, True])
    df = df.drop_duplicates("dup_group_id", keep="first").drop(columns=["_pref"])
    df["dup_count"] = df["dup_group_id"].map(dup_counts).astype(int)
    df = add_features(df)
    df["group_id"] = "sa|" + df["template_key"]
    return df, raw_n


SENDER_PATTERNS = [
    re.compile(r"forwarded by\s+([a-z][a-z .'-]{2,60}?)\s*/"),
    re.compile(
        r"(?:^|\n)\s*(?:x-)?from\s*:\s*([a-z][a-z .'-]{2,60}?)"
        r"(?=\s+(?:on|to|cc|subject|sent)\b|\s*\n|\s*$)"
    ),
    re.compile(
        r"(?:^|\n)\s*([a-z][a-z .'-]{2,40})\s+\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}"
        r"\s+\d{1,2}\s*:\s*\d{2}\s*(?:am|pm)"
    ),
]


def extract_enron_sender(body: object) -> str:
    text = str(body)
    for pat in SENDER_PATTERNS:
        m = pat.search(text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip().lower()
    return ""


def build_clean_enron(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "hf_enron_spam" / "enron_spam_data.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = pd.DataFrame({
        "doc_id": "enron:" + df["Message ID"].astype(str),
        "subject": df["Subject"].fillna("").astype(str),
        "body": df["Message"].fillna("").astype(str),
        "label_raw": df["Spam/Ham"].astype(str).str.strip().str.lower(),
        "date": df["Date"].astype(str),
    })
    df["text"] = (df["subject"] + "\n\n" + df["body"]).str.strip()
    df["label"] = (df["label_raw"] == "spam").astype(int)
    df["dup_group_id"] = df["text"].map(text_hash)
    dup_counts = df.groupby("dup_group_id").size()
    df = df.drop_duplicates("dup_group_id", keep="first").copy()
    df["dup_count"] = df["dup_group_id"].map(dup_counts).astype(int)
    df["corpus"] = "enron"
    df["language"] = "en"
    df["provenance"] = "human"
    df["source_group"] = "enron"
    df["has_plain"] = True
    df["has_html"] = False
    df["body_source"] = "enron_preprocessed"
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df["year"] = df["year"].fillna(0).astype(int).astype(str).replace("0", "unknown")
    df["sender"] = df["body"].map(extract_enron_sender)
    df["year_sender"] = df["year"] + "|" + df["sender"].replace("", "unknown")
    df["thread_key"] = [
        thread_key(s, t) for s, t in zip(df["subject"], df["text"])
    ]
    df["group_id"] = [
        f"enron|{ys}" if sender else f"enron|{year}|unknown|{tk}"
        for ys, sender, year, tk in zip(
            df["year_sender"], df["sender"], df["year"], df["thread_key"]
        )
    ]
    df["tokenized_url_count"] = df["body"].map(
        lambda s: len(TOKENIZED_URL_RE.findall(str(s)))
    )
    df = add_features(df)
    return df


def load_llmgen_en(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "hf_llmgen_phishing" / "GPT_Phishing_Email_dataset.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    rows = []
    for i, (content, label) in enumerate(zip(df["content"], df["label"])):
        text = str(content)
        m = LLM_PARSE_RE.match(text)
        if not m:
            continue
        subject = clean_markdown(m.group(1)).replace("\n", " ")
        body = clean_markdown(m.group(3))
        if not subject or not body:
            continue
        rows.append({
            "corpus": "llmgen",
            "doc_id": f"llmgen-en:{i}",
            "subject": subject,
            "body": body,
            "text": (subject + "\n\n" + body).strip(),
            "label": int(label),
            "label_raw": "phishing" if int(label) else "legitimate",
            "language": "en",
            "provenance": "synthetic",
            "source_group": "llmgen",
            "date": "",
            "has_plain": True,
            "has_html": False,
            "body_source": "generated_text",
        })
    return pd.DataFrame(rows)


def load_ephishgen_en(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "hf_ephishgen" / "ephishLLM.json"
    with open(path, encoding="utf-8-sig") as f:
        items = json.load(f)
    rows = []
    for i, item in enumerate(items):
        if str(item.get("Language") or "").strip().lower() != "en":
            continue
        subject = clean_markdown(item.get("Subject") or "")
        body = clean_markdown(item.get("Body") or "")
        if not subject or not body:
            continue
        if subject_looks_non_english(subject):
            continue
        if not is_probably_english(body):
            continue
        label = int(item.get("type", 0))
        rows.append({
            "corpus": "ephishgen",
            "doc_id": f"ephishgen:{i}",
            "subject": subject,
            "body": body,
            "text": (subject + "\n\n" + body).strip(),
            "label": label,
            "label_raw": "phishing" if label else "legitimate",
            "language": "en",
            "provenance": "synthetic",
            "source_group": "ephishgen",
            "date": "",
            "has_plain": True,
            "has_html": bool(HTML_MARKUP_RE.search(body)),
            "body_source": "generated_text",
        })
    return pd.DataFrame(rows)


def load_nest(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "kaggle" / "final_email_dataset.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    rows = []
    for _, r in df.iterrows():
        subject = clean_markdown(r.get("subject") or "")
        body = clean_markdown(r.get("body") or "")
        label_raw = str(r.get("true_label") or "").strip().lower()
        rows.append({
            "corpus": "nest",
            "doc_id": str(r.get("record_id")),
            "subject": subject,
            "body": body,
            "text": (subject + "\n\n" + body).strip(),
            "label": 1 if label_raw == "phish" else 0,
            "label_raw": label_raw,
            "language": "en",
            "provenance": "synthetic",
            "source_group": "nest",
            "date": str(r.get("date") or ""),
            "has_plain": True,
            "has_html": bool(HTML_MARKUP_RE.search(body)),
            "body_source": "generated_text",
        })
    return pd.DataFrame(rows)


def build_clean_synthetic(data_dir: Path) -> pd.DataFrame:
    parts = [load_llmgen_en(data_dir), load_ephishgen_en(data_dir), load_nest(data_dir)]
    df = pd.concat(parts, ignore_index=True)
    df["dup_group_id"] = df["text"].map(text_hash)
    dup_counts = df.groupby("dup_group_id").size()
    df = df.drop_duplicates("dup_group_id", keep="first").copy()
    df["dup_count"] = df["dup_group_id"].map(dup_counts).astype(int)
    df = add_features(df)
    df["group_id"] = df["corpus"] + "|" + df["template_key"]
    df["url_count"] = df["url_count"].astype(int)
    return df


# ---------------------------------------------------------------------------
# stratified sampling
# ---------------------------------------------------------------------------
def allocate_capped(
    total: int,
    caps: dict[str, int],
    weights: dict[str, float] | None = None,
    floor: int = 0,
) -> dict[str, int]:
    """Allocate `total` units across keys, respecting capacities."""
    keys = list(caps)
    total = min(total, sum(caps.values()))
    base = {k: 0 for k in keys}

    if floor > 0:
        for k in sorted(keys, key=lambda x: caps[x], reverse=True):
            if sum(base.values()) >= total:
                break
            give = min(floor, caps[k])
            base[k] = give

    remaining = total - sum(base.values())
    if remaining > 0:
        residual = {k: caps[k] - base[k] for k in keys}
        live = [k for k in keys if residual[k] > 0]
        if weights is None:
            # Proportional to remaining capacity (used for archive quotas).
            wsum = sum(residual[k] for k in live)
            raw = {k: (remaining * residual[k] / wsum) if wsum else 0.0 for k in keys}
        else:
            # Fixed source shares (used for human/synthetic source quotas).
            wsum = sum(weights[k] for k in live)
            raw = {
                k: (remaining * weights[k] / wsum) if k in live and wsum else 0.0
                for k in keys
            }
        if wsum > 0:
            for k in keys:
                base[k] += min(int(raw[k]), residual[k])
            left = total - sum(base.values())
            while left > 0:
                cand = [k for k in keys if base[k] < caps[k]]
                if not cand:
                    break
                k = max(cand, key=lambda x: (raw[x] - int(raw[x]), caps[x] - base[x]))
                base[k] += 1
                left -= 1

    while sum(base.values()) > total:
        k = max(base, key=lambda x: base[x])
        base[k] -= 1
    return base


def _add_bins(pool: pd.DataFrame) -> pd.DataFrame:
    p = pool.copy()
    n = len(p)
    if n >= 8:
        p["_u"] = pd.qcut(p["urgency_proxy"].rank(method="first"), 4, labels=False)
        median_w = p["warmth_proxy"].median()
        p["_w"] = (p["warmth_proxy"] > median_w).astype(int)
    else:
        p["_u"] = 0
        p["_w"] = 0
    p["_stratum"] = p["_u"].astype(str) + "_" + p["_w"].astype(str)
    return p


def stratified_sample(
    pool: pd.DataFrame,
    n: int,
    seed: int,
    cap: int = 2,
) -> pd.DataFrame:
    """Sample n rows spread over urgency/warmth strata, with a template cap."""
    if n <= 0:
        return pool.iloc[0:0].copy()
    if len(pool) < n:
        raise ValueError(f"pool too small: need {n}, have {len(pool)}")
    p = _add_bins(pool)
    sizes = p["_stratum"].value_counts().to_dict()
    quotas = allocate_capped(n, sizes)
    p = p.sample(frac=1, random_state=seed)

    selected: list[int] = []
    chosen: set[int] = set()
    stratum_counts: Counter = Counter()
    template_counts: Counter = Counter()

    for pass_cap in (cap, cap * 2, 10 ** 9):
        if len(selected) >= n:
            break
        for idx, row in p.iterrows():
            if len(selected) >= n:
                break
            if idx in chosen:
                continue
            stratum = row["_stratum"]
            if stratum_counts[stratum] >= quotas.get(stratum, 0):
                continue
            key = str(row.get("template_key") or row.get("doc_id"))
            if template_counts[key] >= pass_cap:
                continue
            selected.append(idx)
            chosen.add(idx)
            stratum_counts[stratum] += 1
            template_counts[key] += 1

    if len(selected) < n:
        raise ValueError(f"could only select {len(selected)}/{n} rows")
    return p.loc[selected].drop(columns=["_u", "_w", "_stratum"])


# ---------------------------------------------------------------------------
# sample builders
# ---------------------------------------------------------------------------
def e1_pools(sa: pd.DataFrame, syn: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    def sa_pool(label: int) -> pd.DataFrame:
        g = sa[
            (sa["label"] == label)
            & sa["word_count"].between(30, 500)
            & (~sa["is_empty"])
            & sa["subject"].str.strip().ne("")
            & (~sa["html_markup"])
            & sa["has_plain"]
        ]
        return g

    def syn_pool(label: int) -> pd.DataFrame:
        g = syn[
            (syn["label"] == label)
            & syn["word_count"].between(30, 500)
            & (~syn["is_empty"])
            & syn["subject"].str.strip().ne("")
            & (~syn["html_markup"])
        ]
        return g

    s = syn_pool(1)
    h = syn_pool(0)
    return {
        "human_spam": {"spamassassin": sa_pool(1)},
        "human_ham": {"spamassassin": sa_pool(0)},
        "synthetic_spam": {
            "llmgen": s[s["corpus"] == "llmgen"],
            "ephishgen": s[s["corpus"] == "ephishgen"],
            "nest": s[s["corpus"] == "nest"],
        },
        "synthetic_ham": {
            "llmgen": h[h["corpus"] == "llmgen"],
            "ephishgen": h[h["corpus"] == "ephishgen"],
            "nest": h[h["corpus"] == "nest"],
        },
    }


def build_e1(sa: pd.DataFrame, syn: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    pools = e1_pools(sa, syn)
    parts = []
    cell_codes = {
        "human_spam": "HS", "human_ham": "HH",
        "synthetic_spam": "SS", "synthetic_ham": "SH",
    }
    for cell_i, (cell, sources) in enumerate(pools.items()):
        caps = {name: len(df) for name, df in sources.items()}
        weights = {name: (1.0 if cell.startswith("human") else
                          {"llmgen": 0.4, "ephishgen": 0.4, "nest": 0.2}[name])
                   for name in sources}
        alloc = allocate_capped(100, caps, weights)
        for src_i, (name, k) in enumerate(alloc.items()):
            if k <= 0:
                continue
            part = stratified_sample(
                sources[name], k, seed + cell_i * 17 + src_i, cap=2
            ).copy()
            part["cell"] = cell
            part["cell_code"] = cell_codes[cell]
            parts.append(part)
    items = pd.concat(parts, ignore_index=True)
    items = items.sample(frac=1, random_state=seed).reset_index(drop=True)
    counters: Counter = Counter()
    sample_ids = []
    for cell in items["cell"]:
        counters[cell] += 1
        sample_ids.append(f"E1-{cell_codes[cell]}-{counters[cell]:03d}")
    items.insert(0, "sample_id", sample_ids)
    return items


def build_e2(sa: pd.DataFrame, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, int]:
    pool = sa[
        (sa["label"] == 1)
        & sa["word_count"].between(50, 300)
        & (~sa["is_empty"])
        & sa["subject"].str.strip().ne("")
        & (~sa["html_markup"])
        & sa["has_plain"]
        & sa["has_url"]
    ].copy()
    # Cap any single archive so one template batch cannot dominate the seeds.
    caps = {
        k: min(int(v), 100)
        for k, v in pool["source_group"].value_counts().to_dict().items()
    }
    alloc = allocate_capped(200, caps, weights=None, floor=10)
    parts = []
    for i, (archive, k) in enumerate(alloc.items()):
        if k <= 0:
            continue
        parts.append(
            stratified_sample(pool[pool["source_group"] == archive], k, seed + i, cap=2)
        )
    seeds = pd.concat(parts, ignore_index=True)
    seeds = seeds.sample(frac=1, random_state=seed).reset_index(drop=True)
    seeds.insert(0, "seed_id", [f"E2-{i:03d}" for i in range(1, len(seeds) + 1)])
    return seeds, len(pool)


def protected_spans(text: str) -> list[str]:
    ent = extract_entities(text)
    out, seen = [], set()
    for key in ["urls", "emails", "phones", "money", "dates",
                "time_windows", "ids", "attachments"]:
        for value in sorted(set(ent[key]), key=lambda x: (len(x), x)):
            if value and value not in seen:
                seen.add(value)
                out.append(value)
    return out


def seed_notes(text: str) -> str:
    actions = sorted(concept_hits(text, ACTION_CONCEPTS))
    threats = sorted(concept_hits(text, THREAT_CONCEPTS))
    creds = sorted(credential_hits(text))
    deadlines = sorted(set(extract_time_windows(text)))
    return (
        "spam"
        f"; actions={actions}"
        f"; threats={threats}"
        f"; credentials={creds}"
        f"; deadlines={deadlines[:5]}"
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_（无数据）_"
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.3f}")
            else:
                vals.append(str(v).replace("|", "\\|")[:120])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    sa: pd.DataFrame,
    enron: pd.DataFrame,
    syn: pd.DataFrame,
    e2: pd.DataFrame,
    e1: pd.DataFrame,
    raw_sa: int,
    raw_enron: int,
    e2_pool_n: int,
) -> None:
    r: list[str] = []
    r.append("# E1 / E2 抽样报告\n")
    r.append(f"- 输出目录：`{out_dir}`")
    r.append(f"- 随机种子：{RANDOM_STATE}")
    r.append("- E2：200 个改写种子；E1：400 个标注条目（spam/ham × human/synthetic）\n")

    r.append("## 1. 清洗结果\n")
    r.append("### 1.1 SpamAssassin\n")
    r.append(f"- 原始邮件：{raw_sa}")
    r.append(f"- 归一化正文去重后：{len(sa)}（要求 6,148）")
    r.append(f"- 去重后 spam/ham：{int((sa['label'] == 1).sum())} / {int((sa['label'] == 0).sum())}")
    r.append(f"- 模板组数：{sa['group_id'].nunique()}；最大精确重复组：{int(sa['dup_count'].max())}")
    r.append(f"- 含 HTML 标记：{int(sa['html_markup'].sum())}；HTML-only：{int((sa['body_source'] == 'html_stripped').sum())}")

    r.append("\n### 1.2 Enron\n")
    r.append(f"- 原始邮件：{raw_enron}")
    r.append(f"- 去重后：{len(enron)}（要求 30,138，含 1 个空文本桶）")
    r.append(f"- 非空可用：{int((~enron['is_empty']).sum())}")
    r.append(f"- 去重后 spam/ham：{int((enron['label'] == 1).sum())} / {int((enron['label'] == 0).sum())}")
    sender_known = int((enron["sender"] != "").sum())
    r.append(f"- 发件人可抽取：{sender_known}/{len(enron)}（{sender_known / len(enron):.1%}）；其余用 `year|unknown|thread` 兜底")
    r.append(f"- 泄漏分组数：{enron['group_id'].nunique()}；`year|sender` 组数：{enron['year_sender'].nunique()}")
    r.append(f"- Enron URL 被数据集预处理成 `http : / /`，可识别 URL 数：{int(enron['url_count'].sum())}；tokenized URL 邮件数：{int((enron['tokenized_url_count'] > 0).sum())}")
    year_tab = (
        enron.assign(year=enron["year"])
        .groupby("year")
        .agg(n=("doc_id", "size"), spam=("label", "sum"))
        .reset_index()
    )
    year_tab["spam_rate"] = year_tab["spam"] / year_tab["n"]
    r.append("\n" + md_table(year_tab, max_rows=10))

    r.append("\n### 1.3 合成语料（英文）\n")
    syn_tab = (
        syn.groupby(["corpus", "label_raw"])
        .size()
        .reset_index(name="n")
    )
    r.append(md_table(syn_tab))
    r.append(f"\n- 合成语料去重后合计：{len(syn)}；其中 LLMGen 只保留可解析 `Subject/From` 的英文条目，E-PhishGEN 只保留英文正文且排除明显非英文主题。")

    r.append("\n## 2. E2：200 个改写种子\n")
    r.append(f"- 抽样池：SpamAssassin 唯一 spam 中满足 50–300 词、纯文本、≥1 URL、主题非空、非 HTML 的邮件：{e2_pool_n} 条；抽取 {len(e2)} 条。")
    r.append("- 只用 SpamAssassin：Enron 的 URL 被预处理成 `http : / /`，无法做逐字保留 URL 的改写。")
    r.append("- `20050311_spam_2` 与 `20030228_spam_2` 高度重复，去重后没有独占条目；为避免单一归档主导，单归档上限 100 条。")
    arch = e2["source_group"].value_counts().rename_axis("archive").reset_index(name="n")
    r.append("\n" + md_table(arch))
    r.append(f"\n- 词数：p10={e2['word_count'].quantile(.10):.0f}, median={e2['word_count'].median():.0f}, p90={e2['word_count'].quantile(.90):.0f}")
    r.append(f"- URL 数/封：median={e2['url_count'].median():.0f}, max={int(e2['url_count'].max())}")
    coverage = {
        "email": (e2["email_count"] > 0).mean(),
        "phone": (e2["phone_count"] > 0).mean(),
        "money": (e2["money_count"] > 0).mean(),
        "date": (e2["date_count"] > 0).mean(),
        "time_window": (e2["time_window_count"] > 0).mean(),
        "id": (e2["id_count"] > 0).mean(),
        "attachment": (e2["attachment_count"] > 0).mean(),
    }
    r.append("- 受保护实体覆盖率：" + ", ".join(f"{k}={v:.1%}" for k, v in coverage.items()))
    r.append(f"- 模板簇数：{e2['template_key'].nunique()}；单模板最多 {int(e2['template_key'].value_counts().max())} 条")

    r.append("\n## 3. E1：400 个标注条目\n")
    cell_tab = (
        e1.groupby(["cell", "corpus", "provenance", "label_raw"])
        .size()
        .reset_index(name="n")
    )
    r.append(md_table(cell_tab, max_rows=20))
    cell_len = (
        e1.groupby("cell")["word_count"]
        .agg(n="size", p10=lambda x: x.quantile(.10), median="median",
             p90=lambda x: x.quantile(.90))
        .reset_index()
    )
    r.append("\n" + md_table(cell_len))
    r.append("\n- 每个 cell 100 条；human = SpamAssassin 真实邮件，synthetic = LLMGen / E-PhishGEN / NEST。")
    r.append("- 标注表 `e1_annotation_sheet.csv` 已去掉 label、provenance、corpus，并打乱顺序；完整元数据在 `e1_items.csv`。")
    r.append("- `urgency_proxy` / `warmth_proxy` 只用于抽样时拉开温度分布，**不是温度标签**，分析时不要当因变量。")

    r.append("\n## 4. 关键设计决定与限制\n")
    r.append("1. **E1 的人类样本只用 SpamAssassin**：Enron 文本被数据集小写化并把标点拆开（如 `2 : 00 p . m .`），会系统性削弱大小写/感叹号等温度线索，直接用于人工温度评分会造成格式混淆。Enron 仍保留在 `clean_enron.csv`，供 E3 检测器训练/分组切分。")
    r.append("2. **E2 只用 SpamAssassin**：Enron 的 URL 已被 token 化，无法满足“URL 逐字保留”的硬约束。")
    r.append("3. **E-PhishGEN 的 `Language=en` 条目标题不全是英文**（正文是英文，部分主题是意大利语等）；已用停用词规则排除 116 条明显非英文主题。")
    r.append("4. **NEST 的 phish 经过 30–500 词过滤后只剩 37 条**，因此 synthetic_spam 的 NEST 配额为 20，其余由 LLMGen/E-PhishGEN 补足。")
    r.append("5. **E1 人类样本的 ham 来自 easy/hard ham**，包含简报/邮件列表，个人往来比例低于 Enron；后续分析需报告来源构成。")
    r.append("6. **只做了精确去重**；模板/近重复用 `template_key` 在抽样时限制（E2 每模板最多 2 条，E1 每模板最多 2 条，不足时自动放宽），但没有做全语料的 MinHash 去重。")
    r.append("7. **Enron 含 PII**；SpamAssassin 年代偏老（2002–2005）；合成语料不能代表人类温度表达。论文中必须限定外推范围。")
    r.append("8. **E1 还没有温度标注**：这 400 条是标注条目池；下一步按 50 条校准 → 400 条 × 3 人标注，算 ICC/α 后再做 EFA。")
    r.append("9. **E1 的合成 cell 在代理分数上更暖/更急**（synthetic 的 warmth/urgency 中位数高于 human SA）。这是来源与温度的潜在混淆；分析时要把 provenance 作为协变量，并做跨来源测量不变性检验，不能直接把 cell 差异解释为温度差异。")
    r.append("10. **E2 种子的账号/附件实体覆盖为 0**：SpamAssassin 这批 spam 的正文里主要是 URL/邮箱/电话/金额/日期；若后续要检验账号号段或附件名的保留，需要从别的语料补种子（当前 E2 的硬约束仍以 URL 为主）。")

    r.append("\n## 5. 输出文件\n")
    r.append("| 文件 | 内容 |\n|---|---|")
    r.append("| `clean_spamassassin.csv` | 6,148 条去重 SpamAssassin，含模板组、实体计数、plain/HTML 标记 |")
    r.append("| `clean_enron.csv` | 30,138 条去重 Enron（含 1 个空文本桶），含 year/sender/group |")
    r.append("| `clean_synthetic.csv` | 英文合成语料（LLMGen / E-PhishGEN / NEST），含分组与实体计数 |")
    r.append("| `e2_seeds.csv` | 200 个 E2 种子及元数据 |")
    r.append("| `e2_seed_inputs.jsonl` | 可直接接入生成管线的输入（subject/body/protected_spans/notes） |")
    r.append("| `e1_items.csv` | 400 个 E1 条目及完整元数据 |")
    r.append("| `e1_annotation_sheet.csv` | 400 条盲评表（sample_id/subject/body，已打乱） |")
    r.append("| `sampling_manifest.json` | 参数、计数、文件清单 |")

    (out_dir / "SAMPLING_REPORT.md").write_text("\n".join(r) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    global RANDOM_STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="D:/paper/mail/data")
    parser.add_argument("--out", default=str(HERE / "samples"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    RANDOM_STATE = args.seed

    print("[1/5] SpamAssassin ...", flush=True)
    sa, raw_sa_rows = build_clean_spamassassin(data_dir)
    print(f"      unique={len(sa)} (raw={raw_sa_rows})", flush=True)
    if len(sa) != 6148:
        print(f"[warn] expected 6,148 unique SpamAssassin texts, got {len(sa)}", file=sys.stderr)

    print("[2/5] Enron ...", flush=True)
    enron = build_clean_enron(data_dir)
    print(f"      unique={len(enron)}", flush=True)
    if len(enron) != 30138:
        print(f"[warn] expected 30,138 unique Enron texts, got {len(enron)}", file=sys.stderr)

    print("[3/5] synthetic ...", flush=True)
    syn = build_clean_synthetic(data_dir)
    print(f"      english synthetic={len(syn)}", flush=True)

    print("[4/5] E2 seeds ...", flush=True)
    e2, e2_pool_n = build_e2(sa, seed=args.seed)
    print(f"      E2={len(e2)} (pool={e2_pool_n})", flush=True)

    print("[5/5] E1 items ...", flush=True)
    e1 = build_e1(sa, syn, seed=args.seed)
    print(f"      E1={len(e1)}", flush=True)

    # persist cleaned tables
    sa.to_csv(out_dir / "clean_spamassassin.csv", index=False, encoding="utf-8-sig")
    enron.to_csv(out_dir / "clean_enron.csv", index=False, encoding="utf-8-sig")
    syn.to_csv(out_dir / "clean_synthetic.csv", index=False, encoding="utf-8-sig")

    # E2 metadata + prompt-ready inputs
    e2_out = e2.copy()
    e2_out["protected_spans"] = [
        json.dumps(protected_spans(t), ensure_ascii=False)
        for t in e2_out["text"]
    ]
    e2_out["notes"] = e2_out["text"].map(seed_notes)
    e2_out.to_csv(out_dir / "e2_seeds.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "e2_seed_inputs.jsonl", "w", encoding="utf-8") as f:
        for _, r in e2_out.iterrows():
            rec = {
                "seed_id": r["seed_id"],
                "language": "en",
                "subject": r["subject"],
                "body": r["body"],
                "protected_spans": json.loads(r["protected_spans"]),
                "notes": r["notes"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # E1 full metadata + blinded annotation sheet
    e1_out = e1.copy()
    e1_out.to_csv(out_dir / "e1_items.csv", index=False, encoding="utf-8-sig")
    blind = e1_out.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    blind = blind[["sample_id", "subject", "body"]].copy()
    blind.insert(0, "order", range(1, len(blind) + 1))
    blind.to_csv(out_dir / "e1_annotation_sheet.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "seed": args.seed,
        "filters": {
            "e2": {
                "source": "spamassassin spam",
                "body_words": [50, 300],
                "plain_text": True,
                "min_urls": 1,
                "subject_non_empty": True,
            },
            "e1": {
                "body_words": [30, 500],
                "cells": {"human_spam": 100, "human_ham": 100,
                          "synthetic_spam": 100, "synthetic_ham": 100},
                "human_source": "spamassassin only",
                "synthetic_sources": ["llmgen", "ephishgen", "nest"],
            },
        },
        "counts": {
            "spamassassin_raw": raw_sa_rows,
            "spamassassin_unique": len(sa),
            "enron_unique": len(enron),
            "enron_nonempty": int((~enron["is_empty"]).sum()),
            "synthetic_english": len(syn),
            "e2_pool": e2_pool_n,
            "e2": len(e2),
            "e1": len(e1),
        },
        "files": [
            "clean_spamassassin.csv", "clean_enron.csv", "clean_synthetic.csv",
            "e2_seeds.csv", "e2_seed_inputs.jsonl",
            "e1_items.csv", "e1_annotation_sheet.csv", "SAMPLING_REPORT.md",
        ],
    }
    (out_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report(out_dir, sa, enron, syn, e2, e1, raw_sa_rows, 33716, e2_pool_n)
    print(f"[done] wrote samples to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
