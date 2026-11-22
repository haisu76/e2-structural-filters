#!/usr/bin/env python3
"""Validate style-controlled email rewrites before human review.

Checks (hard failures):
  - empty output / unchanged text / near-copy
  - word (or character) length ratio outside the configured band
  - missing or added URLs, emails, phones, money, dates, time windows,
    account/order IDs, attachment filenames
  - meta-preamble ("Certainly", "Here is the rewritten email", ...)
  - new action types (call/reply/pay/download/submit/...)
  - new credential requests, threats, promises, or deadlines

Also reports condition-aware tone diagnostics (urgency/warmth lexicon deltas)
as warnings, not hard failures.

Usage:
  python validate_rewrites.py --pairs pairs.csv --out validation
  python validate_rewrites.py --pairs pairs.jsonl --out validation --show 50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# lexicons (same families as mvp_audit/audit_pairs.py, but word-boundary based)
# ---------------------------------------------------------------------------
URGENCY = [
    "urgent", "immediately", "immediate", "now", "today", "deadline", "expire",
    "expires", "suspend", "suspended", "suspension", "verify", "confirm",
    "alert", "warning", "final", "last chance", "act now", "limited time",
    "hurry", "asap", "important", "attention", "critical",
]
WARMTH = [
    "dear", "hi", "hello", "please", "thank", "thanks", "appreciate", "sorry",
    "apologize", "regards", "sincerely", "kindly", "hope", "welcome", "friend",
    "happy", "help", "assist", "glad", "care",
]
CREDENTIAL_TERMS = [
    r"\bpassword\b", r"\bpasscode\b", r"\bpin\b", r"\bssn\b",
    r"\bsocial security\b", r"\bcredit card\b", r"\bcard number\b",
    r"\bcvv\b", r"\bcvc\b", r"\botp\b", r"\bone[- ]time (?:code|password)\b",
    r"\bseed phrase\b", r"\bwallet\b", r"\bbank account\b",
    r"\brouting number\b", r"\blogin credentials\b",
]
THREAT_CONCEPTS = {
    "suspend": r"\bsuspen(?:d(?:ed|ing|s)?|sion)\b",
    "terminate": r"\bterminat(?:e|ed|ing|ion)\b",
    "legal": r"\blegal action\b",
    "penalty": r"\bpenalt(?:y|ies)\b",
    "lock": r"\block(?:ed|ing|s)?\b",
    "close_account": r"\bclos(?:e|ed|ing) (?:your |the )?account\b",
    "deactivate": r"\bdeactivat(?:e|ed|ing|ion)\b",
}
ACTION_CONCEPTS = {
    "click": r"\b(?:click|visit|follow (?:this|the) link|go to|open (?:the )?link)\b",
    "reply": r"\b(?:reply|respond|write back|email (?:us|me) back)\b",
    "call": r"\b(?:call|phone|dial)\b",
    "pay": r"\b(?:pay|payment|transfer|wire|send money)\b",
    "download": r"\b(?:download|open (?:the )?attach(?:ment|ed file)|open (?:the )?file)\b",
    "verify": r"\b(?:verify|verification|confirm|validate|authenticate)\b",
    "submit": r"\b(?:submit|fill (?:out|in)|complete (?:the )?(?:\w+ ){0,2}form|provide (?:your )?(?:information|details))\b",
}
PROMISE_CONCEPTS = {
    "free": r"(?<!feel )\bfree\b",
    "guarantee": r"\bguarantee(?:d|s)?\b",
    "refund": r"\brefund(?:ed|s)?\b",
    "discount": r"\bdiscount(?:ed|s)?\b",
    "prize": r"\bprize(?:s)?\b",
    "bonus": r"\bbonus(?:es)?\b",
    "reward": r"\breward(?:ed|s)?\b",
    "gift": r"\bgift(?:s)?\b",
}

META_FATAL = [
    r"^\s*(?:certainly|sure|of course|absolutely)\b",
    r"^\s*(?:here is|here’s|here's|below is)\s+(?:the|a|an|your|my)?\s*"
    r"(?:(?:rewritten|revised|paraphrased|updated)\s+)?"
    r"(?:email|version|text|paraphrase|rewrite|output|message)\b\s*(?:[:：]|$)",
    r"^\s*(?:rewritten|revised|paraphrased)\s+(?:email|version|text)\b",
    r"^\s*as an ai\b",
    r"\bi (?:have|’ve|'ve) rewritten\b",
    r"^\s*(?:subject|body)\s*:",
    r"^\s*```",
]
META_WARN = [
    r"\bi hope this helps\b",
    r"\blet me know if\b",
    r"\bnote:\s*(?:i|this|the)\b",
]
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|nowhere|neither|nor|"
    r"can(?:not|['’]t)|won['’]t|don['’]t|doesn['’]t|isn['’]t|aren['’]t|"
    r"wasn['’]t|weren['’]t|shouldn['’]t|mustn['’]t|couldn['’]t|wouldn['’]t)\b",
    re.I,
)

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"')\]]+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?"
    r"\d{3,4}[\s.-]?\d{3,4}(?!\d)"
)
MONEY_RE = re.compile(
    r"(?i)(?:[$€£¥]\s?\d[\d,.]*|\b\d[\d,.]*\s?"
    r"(?:USD|EUR|GBP|JPY|CAD|AUD|dollars?|euros?|pounds?)\b)"
)
DATE_PATTERNS = [
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
    r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|"
    r"May|June|July|August|September|October|November|December)\b",
    r"\b(?:today|tomorrow|tonight|yesterday)\b",
]
TIME_WINDOW_RE = re.compile(
    r"(?i)\b(?:within|in|after|before|by|until)\s+"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?:minutes?|hours?|days?|weeks?|months?)\b"
    r"|\b(?:by|before|until|on)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday)\b"
    r"|\b\d+[- ](?:hour|day|week|month)\b"
)
ID_RE = re.compile(
    r"\b(?i:order|invoice|account|ticket|reference|ref|confirmation|"
    r"transaction|tracking|case|id)\b\s*"
    r"(?:(?i:number|no\.?|#|:))?\s*"
    r"([A-Za-z0-9][A-Za-z0-9-]{3,})\b"
)
HASH_ID_RE = re.compile(r"#\s?([A-Za-z0-9][A-Za-z0-9-]{3,})\b")
ATTACH_RE = re.compile(
    r"(?i)\b[\w][\w .-]*\.(?:pdf|docx?|xlsx?|pptx?|zip|rar|7z|html?|txt|csv|eml|msg)\b"
)


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------
def norm_ws(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def norm_text(s: object) -> str:
    return norm_ws(s).lower()


def word_count(s: object) -> int:
    # count URLs and email addresses as one token instead of splitting them
    text = URL_RE.sub(" url ", str(s))
    text = EMAIL_RE.sub(" email ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*", text))


def char_count(s: object) -> int:
    return len(re.sub(r"\s", "", str(s)))


def cjk_ratio(s: object) -> float:
    text = str(s)
    if not text:
        return 0.0
    return len(re.findall(r"[一-鿿぀-ヿ가-힯]", text)) / len(text)


def sentence_count(s: object) -> int:
    return max(len(re.findall(r"[.!?。！？]+", str(s))), 1)


def paragraph_count(s: object) -> int:
    return max(len(re.split(r"\n\s*\n", str(s))), 1)


def tokens(s: object) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['’\-][a-z0-9]+)*", str(s).lower())


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b), autojunk=False).ratio()


def term_count(text: str, terms: list[str]) -> int:
    low = lexicon_text(text)
    total = 0
    for term in terms:
        t = term.strip()
        if not t:
            continue
        total += len(re.findall(r"\b" + re.escape(t) + r"\b", low))
    return total


def clean_url(url: str) -> str:
    return url.rstrip(".,;:!?)]}\"'")


def clean_span(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().rstrip(".,;:!?)]}\"'")


# ---------------------------------------------------------------------------
# entity extraction
# ---------------------------------------------------------------------------
def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def extract_urls(text: str) -> list[str]:
    return [clean_url(m.group(0)) for m in URL_RE.finditer(text)]


def extract_emails(text: str) -> list[str]:
    return [m.group(0).lower() for m in EMAIL_RE.finditer(text)]


def extract_phones(text: str) -> list[str]:
    out = []
    for m in PHONE_RE.finditer(text):
        value = m.group(0)
        digits = re.sub(r"\D", "", value)
        if len(digits) >= 10:
            out.append(re.sub(r"\s+", " ", value.strip()))
    return out


def extract_money(text: str) -> list[str]:
    return [clean_span(m.group(0)).lower() for m in MONEY_RE.finditer(text)]


def plausible_numeric_date(value: str) -> bool:
    parts = re.split(r"[-/.]", value)
    if len(parts) != 3:
        return True
    try:
        a, b, c = (int(p) for p in parts)
    except ValueError:
        return True
    if len(parts[0]) == 4:
        return 1900 <= a <= 2100 and 1 <= b <= 12 and 1 <= c <= 31
    if len(parts[2]) == 4 and not (1900 <= c <= 2100):
        return False
    return (1 <= a <= 12 and 1 <= b <= 31) or (1 <= b <= 12 and 1 <= a <= 31)


def extract_dates(text: str) -> list[str]:
    out = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            raw = m.group(0)
            if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}", raw) \
                    and not plausible_numeric_date(raw):
                continue
            out.append(norm_text(clean_span(raw)))
    return out


def extract_time_windows(text: str) -> list[str]:
    return [norm_text(clean_span(m.group(0))) for m in TIME_WINDOW_RE.finditer(text)]


def valid_id_token(token: str, from_hash: bool = False) -> bool:
    """Reject ordinary words and CSS hex colors captured as IDs."""
    token = token.strip()
    if from_hash and re.fullmatch(r"(?i)(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})", token):
        return False
    # account/order IDs in this corpus are alphanumeric; requiring a digit
    # avoids treating all-caps words ("WILL") as IDs
    return any(ch.isdigit() for ch in token)


def extract_ids(text: str) -> list[str]:
    out = []
    for m in ID_RE.finditer(text):
        token = m.group(1)
        if valid_id_token(token):
            out.append(token.upper())
    for m in HASH_ID_RE.finditer(text):
        token = m.group(1)
        if valid_id_token(token, from_hash=True):
            out.append(token.upper())
    return out


def extract_attachments(text: str) -> list[str]:
    return [norm_text(clean_span(m.group(0))) for m in ATTACH_RE.finditer(strip_urls(text))]


def extract_entities(text: str) -> dict[str, Counter]:
    return {
        "urls": Counter(extract_urls(text)),
        "emails": Counter(extract_emails(text)),
        "phones": Counter(extract_phones(text)),
        "money": Counter(extract_money(text)),
        "dates": Counter(extract_dates(text)),
        "time_windows": Counter(extract_time_windows(text)),
        "ids": Counter(extract_ids(text)),
        "attachments": Counter(extract_attachments(text)),
    }


def counter_diff(a: Counter, b: Counter) -> tuple[list[str], list[str]]:
    missing = sorted((a - b).elements())
    added = sorted((b - a).elements())
    return missing, added


def lexicon_text(text: str) -> str:
    """Strip URLs and email addresses so paths/domains do not trigger lexicons."""
    return EMAIL_RE.sub(" ", URL_RE.sub(" ", str(text))).lower()


def concept_hits(text: str, concepts: dict[str, str]) -> set[str]:
    low = lexicon_text(text)
    return {name for name, pattern in concepts.items() if re.search(pattern, low)}


def credential_hits(text: str) -> set[str]:
    low = lexicon_text(text)
    return {p for p in CREDENTIAL_TERMS if re.search(p, low)}


# ---------------------------------------------------------------------------
# condition handling
# ---------------------------------------------------------------------------
CONDITION_ALIASES = {
    "neutral": "neutral", "control": "neutral", "paraphrase": "neutral",
    "none": "neutral", "same": "neutral",
    "warm": "warmer", "warmer": "warmer", "warmth": "warmer",
    "cold": "colder", "colder": "colder", "formal": "colder", "distant": "colder",
    "urgent": "more_urgent", "more_urgent": "more_urgent",
    "high_urgency": "more_urgent", "increase_urgency": "more_urgent",
    "calm": "less_urgent", "less_urgent": "less_urgent",
    "low_urgency": "less_urgent", "decrease_urgency": "less_urgent",
}


def normalize_condition(value: object) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return CONDITION_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_pair(
    original: str,
    paraphrase: str,
    subject_original: str = "",
    subject_paraphrase: str = "",
    condition: str = "",
    min_ratio: float = 0.80,
    max_ratio: float = 1.20,
    check_proper_nouns: bool = False,
    check_numbers: bool = False,
) -> dict:
    failures: list[str] = []
    failure_codes: list[str] = []
    warnings: list[str] = []

    def fail(code: str, detail: str) -> None:
        failure_codes.append(code)
        failures.append(detail)

    original = str(original or "")
    paraphrase = str(paraphrase or "")
    condition = normalize_condition(condition)

    if not paraphrase.strip():
        fail("empty", "paraphrase is empty")
        return _result(original, paraphrase, condition, failures, failure_codes, warnings, {})

    # HTML markup makes length and entity checks unreliable; convert to text first
    html_pat = r"<\s*/?\s*[a-z][a-z0-9]*(?:\s[^>]*)?>"
    if re.search(html_pat, original, flags=re.I) or re.search(html_pat, paraphrase, flags=re.I):
        warnings.append("html_markup_detected")

    # unchanged / near-copy
    if norm_text(original) == norm_text(paraphrase):
        fail("unchanged", "paraphrase is identical to the original")
    sim = similarity(original, paraphrase)
    jac = jaccard(original, paraphrase)
    if sim >= 0.95 or jac >= 0.95:
        fail("near_copy", f"too similar (similarity={sim:.3f}, jaccard={jac:.3f})")
    elif sim >= 0.90 or jac >= 0.90:
        warnings.append(f"borderline_copy (similarity={sim:.3f}, jaccard={jac:.3f})")

    # language / script sanity (do this before length so a translated output
    # does not also produce a meaningless word-count failure)
    orig_cjk, para_cjk = cjk_ratio(original), cjk_ratio(paraphrase)
    language_detail = ""
    if orig_cjk > 0.5 and para_cjk < 0.3:
        language_detail = f"CJK ratio dropped {orig_cjk:.2f} -> {para_cjk:.2f}"
    elif orig_cjk < 0.1 and para_cjk > 0.4:
        language_detail = f"CJK ratio rose {orig_cjk:.2f} -> {para_cjk:.2f}"

    # length
    orig_words = word_count(original)
    para_words = word_count(paraphrase)
    use_chars = cjk_ratio(original) > 0.20 or orig_words < 20
    if use_chars:
        metric = "chars"
        orig_len = max(char_count(original), 1)
        para_len = char_count(paraphrase)
    else:
        metric = "words"
        orig_len = max(orig_words, 1)
        para_len = para_words
    length_ratio = para_len / orig_len
    if language_detail:
        warnings.append("length_ratio_not_enforced_due_to_language_mismatch")
    elif length_ratio < min_ratio or length_ratio > max_ratio:
        fail("length_ratio",
             f"{metric} ratio {length_ratio:.3f} outside [{min_ratio:.2f}, {max_ratio:.2f}] "
             f"({orig_len} -> {para_len})")

    sent_ratio = sentence_count(paraphrase) / sentence_count(original)
    if sent_ratio < 0.60 or sent_ratio > 1.60:
        warnings.append(f"sentence_ratio {sent_ratio:.2f}")

    para_diff = abs(paragraph_count(paraphrase) - paragraph_count(original))
    if para_diff > 2:
        warnings.append(f"paragraph_count_changed by {para_diff}")

    # entities
    ent_o = extract_entities(original)
    ent_p = extract_entities(paraphrase)
    entity_details: dict[str, str] = {}
    for name in ent_o:
        missing, added = counter_diff(ent_o[name], ent_p[name])
        if missing:
            fail(f"{name}_missing", f"{name} missing: {'; '.join(missing[:5])}")
        if added:
            fail(f"{name}_added", f"{name} added: {'; '.join(added[:5])}")
        entity_details[f"{name}_missing"] = "; ".join(missing)
        entity_details[f"{name}_added"] = "; ".join(added)

    # meta-preamble
    head = paraphrase.lstrip()[:500]
    for pat in META_FATAL:
        if re.search(pat, head, flags=re.I):
            fail("meta_preamble", f"meta-preamble matched: {pat}")
            break
    if "```" in paraphrase:
        fail("meta_preamble", "markdown fence in output")
    for pat in META_WARN:
        if re.search(pat, paraphrase, flags=re.I):
            warnings.append(f"meta_warning: {pat}")

    # new actions / credentials / threats / promises
    orig_actions = concept_hits(original, ACTION_CONCEPTS)
    para_actions = concept_hits(paraphrase, ACTION_CONCEPTS)
    new_actions = para_actions - orig_actions
    # "click" is a channel for an existing URL action; "download" for an existing attachment
    if "click" in new_actions and ent_o["urls"]:
        new_actions.discard("click")
    if "download" in new_actions and ent_o["attachments"]:
        new_actions.discard("download")
    if new_actions:
        fail("new_action", f"new action(s): {', '.join(sorted(new_actions))}")

    new_creds = credential_hits(paraphrase) - credential_hits(original)
    if new_creds:
        fail("new_credential", "new credential term(s): " + "; ".join(sorted(new_creds)))

    new_threats = concept_hits(paraphrase, THREAT_CONCEPTS) - concept_hits(original, THREAT_CONCEPTS)
    if new_threats:
        fail("new_threat", "new threat concept(s): " + ", ".join(sorted(new_threats)))

    new_promises = concept_hits(paraphrase, PROMISE_CONCEPTS) - concept_hits(original, PROMISE_CONCEPTS)
    if new_promises:
        fail("new_promise", "new promise/offer concept(s): " + ", ".join(sorted(new_promises)))

    # removals are warnings, not hard failures: the lexicons are intentionally
    # coarse, so a synonym can look like a missing concept
    missing_actions = orig_actions - para_actions
    if missing_actions:
        warnings.append("action_maybe_removed: " + ", ".join(sorted(missing_actions)))
    missing_creds = credential_hits(original) - credential_hits(paraphrase)
    if missing_creds:
        warnings.append("credential_request_maybe_removed: " + "; ".join(sorted(missing_creds)))
    missing_threats = concept_hits(original, THREAT_CONCEPTS) - concept_hits(paraphrase, THREAT_CONCEPTS)
    if missing_threats:
        warnings.append("threat_maybe_removed: " + ", ".join(sorted(missing_threats)))
    missing_promises = concept_hits(original, PROMISE_CONCEPTS) - concept_hits(paraphrase, PROMISE_CONCEPTS)
    if missing_promises:
        warnings.append("promise_maybe_removed: " + ", ".join(sorted(missing_promises)))

    orig_neg = len(NEGATION_RE.findall(original))
    para_neg = len(NEGATION_RE.findall(paraphrase))
    if orig_neg != para_neg:
        warnings.append(f"negation_count_changed {orig_neg}->{para_neg}")

    # explicit new deadlines: dates/time windows are already checked above,
    # so this is a semantic summary warning for the human reviewer.
    if ent_p["dates"] - ent_o["dates"] or ent_p["time_windows"] - ent_o["time_windows"]:
        warnings.append("review_new_deadline (already flagged as date/time_window added)")

    # subject checks
    subject_similarity = None
    if subject_original and not subject_paraphrase:
        fail("subject_missing", "subject is missing")
    if subject_original and subject_paraphrase:
        subject_similarity = similarity(subject_original, subject_paraphrase)
        if norm_text(subject_original) == norm_text(subject_paraphrase):
            warnings.append("subject_unchanged")
        if subject_similarity < 0.30:
            warnings.append(f"subject_low_similarity {subject_similarity:.2f}")
        for label, value in [("subject", subject_paraphrase)]:
            subj_meta = [p for p in META_FATAL if re.search(p, value.lstrip()[:300], flags=re.I)]
            if subj_meta:
                fail("meta_preamble_subject", f"{label} starts with meta-preamble")
        so = extract_entities(subject_original)
        sp = extract_entities(subject_paraphrase)
        for name in ["urls", "emails", "phones", "money", "dates",
                     "time_windows", "ids", "attachments"]:
            missing, added = counter_diff(so[name], sp[name])
            if missing:
                fail(f"subject_{name}_missing", f"subject {name} missing: {'; '.join(missing[:5])}")
            if added:
                fail(f"subject_{name}_added", f"subject {name} added: {'; '.join(added[:5])}")

    # language / script sanity
    if language_detail:
        fail("language_mismatch", language_detail)

    # optional warnings
    if check_numbers:
        num_o = Counter(re.findall(r"\b\d[\d,.]*\b", original))
        num_p = Counter(re.findall(r"\b\d[\d,.]*\b", paraphrase))
        missing, added = counter_diff(num_o, num_p)
        if missing:
            warnings.append("numbers_missing: " + "; ".join(missing[:8]))
        if added:
            warnings.append("numbers_added: " + "; ".join(added[:8]))
    if check_proper_nouns:
        prop_o = set(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", original))
        prop_p = set(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", paraphrase))
        missing = sorted(prop_o - prop_p)
        if missing:
            warnings.append("proper_nouns_missing: " + "; ".join(missing[:8]))

    # tone diagnostics
    urgency_orig = term_count(original, URGENCY)
    urgency_para = term_count(paraphrase, URGENCY)
    warmth_orig = term_count(original, WARMTH)
    warmth_para = term_count(paraphrase, WARMTH)
    urgency_delta = urgency_para - urgency_orig
    warmth_delta = warmth_para - warmth_orig

    if language_detail:
        pass  # lexicon-based tone diagnostics are invalid on translated text
    elif condition == "neutral":
        if abs(warmth_delta) > 2 or abs(urgency_delta) > 2:
            warnings.append(f"neutral_tone_shift (warmth={warmth_delta}, urgency={urgency_delta})")
    elif condition == "warmer" and warmth_delta < 1:
        warnings.append(f"warmer_direction_weak (warmth delta={warmth_delta})")
    elif condition == "colder" and warmth_delta > -1:
        warnings.append(f"colder_direction_weak (warmth delta={warmth_delta})")
    elif condition == "more_urgent" and urgency_delta < 1:
        warnings.append(f"more_urgent_direction_weak (urgency delta={urgency_delta})")
    elif condition == "less_urgent" and urgency_delta > -1:
        warnings.append(f"less_urgent_direction_weak (urgency delta={urgency_delta})")

    details = {
        "metric": metric,
        "orig_len": orig_len,
        "para_len": para_len,
        "length_ratio": round(length_ratio, 4),
        "similarity": round(sim, 4),
        "jaccard": round(jac, 4),
        "sentence_ratio": round(sent_ratio, 4),
        "subject_similarity": round(subject_similarity, 4) if subject_similarity is not None else None,
        "urgency_orig": urgency_orig,
        "urgency_para": urgency_para,
        "urgency_delta": urgency_delta,
        "warmth_orig": warmth_orig,
        "warmth_para": warmth_para,
        "warmth_delta": warmth_delta,
        **entity_details,
    }
    return _result(original, paraphrase, condition, failures, failure_codes, warnings, details)


def _result(original, paraphrase, condition, failures, failure_codes, warnings, details) -> dict:
    row = {
        "condition": condition,
        "pass": len(failures) == 0,
        "failure_codes": ";".join(sorted(set(failure_codes))),
        "failure_detail": " | ".join(failures),
        "warnings": " | ".join(warnings),
        **details,
    }
    if not details:
        row.update({
            "metric": "chars" if cjk_ratio(original) > 0.20 else "words",
            "orig_len": 0, "para_len": 0, "length_ratio": 0.0,
            "similarity": 0.0, "jaccard": 0.0, "sentence_ratio": 0.0,
            "subject_similarity": None, "urgency_orig": 0, "urgency_para": 0,
            "urgency_delta": 0, "warmth_orig": 0, "warmth_para": 0, "warmth_delta": 0,
        })
    return row


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------
ALIASES = {
    "id": ["id", "pair_id", "doc_id", "sample_id", "source_sample_id"],
    "original": ["original", "original_body", "source_body", "source", "src", "input"],
    "paraphrase": ["paraphrase", "paraphrased", "rewrite", "rewritten", "target", "output", "body"],
    "subject_original": ["subject_original", "original_subject", "source_subject"],
    "subject_paraphrase": ["subject_paraphrase", "paraphrase_subject", "rewritten_subject"],
    "condition": ["condition", "style", "variant", "tone", "setting"],
}


def pick(row: dict, key: str) -> str:
    for alias in ALIASES[key]:
        if alias in row and pd.notna(row[alias]):
            value = row[alias]
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
    return ""


def maybe_json_text(value: str, field: str) -> str:
    s = str(value or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and field in obj:
                return str(obj[field])
        except Exception:
            pass
    return s


def load_pairs(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    raise ValueError(f"unsupported input format: {suffix}")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, record in enumerate(df.to_dict(orient="records")):
        original = maybe_json_text(pick(record, "original"), "body")
        paraphrase = maybe_json_text(pick(record, "paraphrase"), "body")
        subject_original = pick(record, "subject_original")
        subject_paraphrase = pick(record, "subject_paraphrase")
        # if paraphrase is a JSON object, recover subject too
        para_raw = pick(record, "paraphrase")
        if para_raw.strip().startswith("{"):
            try:
                obj = json.loads(para_raw)
                paraphrase = str(obj.get("body", paraphrase))
                subject_paraphrase = str(obj.get("subject", subject_paraphrase))
            except Exception:
                pass
        pair_id = pick(record, "id") or f"row-{i}"
        rows.append({
            "id": pair_id,
            "original": original,
            "paraphrase": paraphrase,
            "subject_original": subject_original,
            "subject_paraphrase": subject_paraphrase,
            "condition": pick(record, "condition") or "neutral",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
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
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v).replace("|", "\\|")[:160])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(results: pd.DataFrame, out_prefix: Path, args: argparse.Namespace) -> str:
    lines = ["# Rewrite validation summary\n"]
    lines.append(f"- input: `{args.pairs}`")
    lines.append(f"- pairs: {len(results)}")
    passed = int(results["pass"].sum())
    lines.append(f"- pass: {passed}/{len(results)} ({passed / max(len(results), 1):.1%})")
    metric_label = "/".join(
        "word" if m == "words" else "char" if m == "chars" else str(m)
        for m in sorted(set(results["metric"]))
    )
    lines.append(f"- length band: {args.min_ratio:.2f}–{args.max_ratio:.2f} ({metric_label} metric)")
    lines.append("- `pass` means automatic hard checks only; semantic warnings still need human review.\n")

    if "condition" in results.columns:
        by_cond = results.groupby("condition").agg(
            n=("pass", "size"), pass_n=("pass", "sum")
        ).reset_index()
        by_cond["pass_rate"] = by_cond["pass_n"] / by_cond["n"]
        lines.append("## By condition\n")
        lines.append(md_table(by_cond))
        lines.append("")

    fail_counts: Counter = Counter()
    for value in results["failure_codes"].fillna(""):
        for code in str(value).split(";"):
            if code.strip():
                fail_counts[code.strip()] += 1
    if fail_counts:
        lines.append("## Failure reasons\n")
        fail_df = pd.DataFrame(
            [{"reason": k, "n": v} for k, v in fail_counts.most_common()]
        )
        fail_df["share"] = fail_df["n"] / len(results)
        lines.append(md_table(fail_df))
        lines.append("")

    warn_counts: Counter = Counter()
    for value in results["warnings"].fillna(""):
        for item in str(value).split(" | "):
            if item.strip():
                match = re.match(r"[A-Za-z_]+", item)
                warn_counts[match.group(0) if match else item.strip()] += 1
    if warn_counts:
        lines.append("## Warnings\n")
        warn_df = pd.DataFrame(
            [{"warning": k, "n": v} for k, v in warn_counts.most_common()]
        )
        lines.append(md_table(warn_df))
        lines.append("")

    ratio = results["length_ratio"].dropna()
    if len(ratio):
        lines.append("## Length ratio distribution\n")
        lines.append(f"- p10={ratio.quantile(0.10):.3f}, median={ratio.median():.3f}, "
                     f"p90={ratio.quantile(0.90):.3f}, min={ratio.min():.3f}, max={ratio.max():.3f}")
        lines.append("")

    failed = results[~results["pass"]]
    if not failed.empty:
        lines.append("## First failing pairs\n")
        cols = ["id", "condition", "failure_codes", "length_ratio", "similarity", "failure_detail"]
        lines.append(md_table(failed[cols].head(args.show), max_rows=args.show))
        lines.append("")

    lines.append("## Note\n")
    lines.append("自动校验只做候选筛查，不能替代人工纳入。最终仍需判断：语义保持、实体保持、"
                 "无新增危害、流畅、确实改写；并报告双人一致性与裁决。")
    report = "\n".join(lines) + "\n"
    (out_prefix.parent / f"{out_prefix.name}_summary.md").write_text(report, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", default="validation")
    parser.add_argument("--min-ratio", type=float, default=0.80)
    parser.add_argument("--max-ratio", type=float, default=1.20)
    parser.add_argument("--check-proper-nouns", action="store_true")
    parser.add_argument("--check-numbers", action="store_true")
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.pairs)
    df = load_pairs(path)
    if args.max_rows:
        df = df.head(args.max_rows)
    df = normalize_frame(df)
    if df.empty:
        print("[error] no input rows")
        return 1

    results = []
    for _, row in df.iterrows():
        result = validate_pair(
            row["original"], row["paraphrase"],
            subject_original=row["subject_original"],
            subject_paraphrase=row["subject_paraphrase"],
            condition=row["condition"],
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            check_proper_nouns=args.check_proper_nouns,
            check_numbers=args.check_numbers,
        )
        result = {"id": row["id"], **result}
        results.append(result)

    out = pd.DataFrame(results)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report = write_summary(out, out_prefix, args)
    print(report)
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {out_prefix.parent / (out_prefix.name + '_summary.md')}")


if __name__ == "__main__":
    sys.exit(main())
