"""Shared schema for the E1 emotional-temperature annotation.

All item columns are 1-7 Likert agreement ratings. Reverse-scored items are
reversed only during analysis; raters always answer the item as written.
"""

from __future__ import annotations

SCALE_MIN = 1
SCALE_MAX = 7

DIMENSIONS = {
    "valence": {
        "label": "Valence",
        "label_zh": "效价",
        "items": ["V1_negative_tone", "V2_goodwill"],
        "reverse": ["V1_negative_tone"],
    },
    "urgency": {
        "label": "Arousal / urgency",
        "label_zh": "唤醒/紧迫",
        "items": ["A1_time_pressure", "A2_intensity", "A3_act_now"],
        "reverse": [],
    },
    "warmth": {
        "label": "Interpersonal warmth",
        "label_zh": "人际温度",
        "items": ["W1_care", "W2_friendly", "W3_person_not_ticket"],
        "reverse": [],
    },
    "distance": {
        "label": "Distance / institutionalization",
        "label_zh": "距离/机构化",
        "items": ["D1_institutional", "D2_impersonal", "D3_procedural"],
        "reverse": [],
    },
}

ITEM_TEXT = {
    "V1_negative_tone": "The overall emotional tone is negative.",
    "V2_goodwill": "The email conveys goodwill or a positive attitude toward the recipient.",
    "V3_hostility": "The email expresses hostility, resentment, or displeasure.",
    "A1_time_pressure": "The email creates a sense of time pressure.",
    "A2_intensity": "The email is emotionally intense or activated.",
    "A3_act_now": "The email pushes the reader to act immediately.",
    "W1_care": "The email shows care for the recipient's interests.",
    "W2_friendly": "The tone is friendly and approachable.",
    "W3_person_not_ticket": "The email treats the recipient as a person rather than a case or ticket.",
    "D1_institutional": "The email reads like an impersonal institutional notice.",
    "D2_impersonal": "The email avoids personal address or personal details.",
    "D3_procedural": "The language is formal and procedural.",
}

ITEM_TEXT_ZH = {
    "V1_negative_tone": "整体情感基调是消极的。",
    "V2_goodwill": "邮件对收件人表达善意或积极态度。",
    "V3_hostility": "邮件表达敌意、不满或怨恨。",
    "A1_time_pressure": "邮件制造时间压力。",
    "A2_intensity": "邮件的情绪强度高、被激活。",
    "A3_act_now": "邮件催促收件人立即行动。",
    "W1_care": "邮件关心收件人的利益。",
    "W2_friendly": "语气友好、易于接近。",
    "W3_person_not_ticket": "邮件把收件人当成人，而不是工单/编号。",
    "D1_institutional": "邮件像非个人的机构告示。",
    "D2_impersonal": "邮件避免个人称呼或个人细节。",
    "D3_procedural": "语言正式、程序化。",
}

DIMENSION_ITEMS = {name: dim["items"] for name, dim in DIMENSIONS.items()}
SCORED_ITEMS = [item for dim in DIMENSIONS.values() for item in dim["items"]]

# Collected but not part of any dimension score. V3 is retained for descriptive
# hostility prevalence; it is too rare in email corpora to support a reliable
# 1-7 scale (calibration v1.1: R1/R2 gave 1 to all 50 items).
DIAGNOSTIC_ITEMS = ["V3_hostility"]

ITEMS = ["V1_negative_tone", "V2_goodwill", "V3_hostility",
         "A1_time_pressure", "A2_intensity", "A3_act_now",
         "W1_care", "W2_friendly", "W3_person_not_ticket",
         "D1_institutional", "D2_impersonal", "D3_procedural"]
REVERSE_ITEMS = {item for dim in DIMENSIONS.values() for item in dim["reverse"]}
ITEM_DIMENSION = {item: name for name, dim in DIMENSIONS.items() for item in dim["items"]}
ITEM_DIMENSION["V3_hostility"] = "valence"
ITEM_ROLE = {item: ("diagnostic" if item in DIAGNOSTIC_ITEMS else "scored")
             for item in ITEMS}

THREAT_COLUMNS = ["threat_binary", "threat_severity"]
RATING_COLUMNS = ["rater_id", "calib_id"] + ITEMS + THREAT_COLUMNS + ["confidence", "notes"]
TEMPLATE_COLUMNS = ["rater_id", "order", "calib_id", "subject", "body"] + \
    ITEMS + THREAT_COLUMNS + ["confidence", "notes"]

SCHEMA_VERSION = "1.3"

# Version switch. The default stays v1.3 so every existing analysis remains
# reproducible; setting E1_SCHEMA=v2.0 or v2.1 replaces the exported names with
# the requested instrument. Imported last on purpose: the star import
# overwrites the v1.3 constants above.
import os as _os  # noqa: E402

_schema = _os.environ.get("E1_SCHEMA", "").strip().lower()
if _schema in {"v2", "v2.0", "2.0", "rating_schema_v2"}:
    from rating_schema_v2 import *  # noqa: F401,F403,E402
elif _schema in {"v2.1", "2.1", "rating_schema_v2_1"}:
    from rating_schema_v2_1 import *  # noqa: F401,F403,E402
