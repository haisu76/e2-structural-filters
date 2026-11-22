"""Schema for the E1 emotional-temperature annotation, instrument v2.0.

v2.0 changes relative to the frozen v1.3 schema:

- Valence gains ``V4_complaint`` so the dimension has three scored items and
  separates global negative tone from a concrete complaint/criticism.
- Distance gains ``D4_automated`` and redefines ``D1``/``D2`` so institutional
  voice, generic address, formal register, and automation are scored
  separately.
- Item wording for ``W2``/``W3`` is unchanged; only the codebook anchors are
  expanded (anchors are not part of the schema).
- ``V3_hostility`` remains a diagnostic item and is excluded from the valence
  score and from all PASS gates.

Activate this schema with ``E1_SCHEMA=v2.0``; the default in
``rating_schema.py`` remains the frozen v1.3 instrument.
"""

from __future__ import annotations

SCHEMA_VERSION = "2.0"

SCALE_MIN = 1
SCALE_MAX = 7

DIMENSIONS = {
    "valence": {
        "label": "Valence",
        "label_zh": "效价",
        "items": ["V1_negative_tone", "V2_goodwill", "V4_complaint"],
        "reverse": ["V1_negative_tone", "V4_complaint"],
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
        "items": ["D1_institutional", "D2_impersonal", "D3_procedural",
                  "D4_automated"],
        "reverse": [],
    },
}

ITEM_TEXT = {
    "V1_negative_tone": "The overall emotional tone of the email is negative.",
    "V2_goodwill": "The email conveys goodwill or a positive attitude toward the recipient.",
    "V3_hostility": "The email expresses hostility, resentment, or displeasure.",
    "V4_complaint": "The email expresses dissatisfaction, complaint, or criticism.",
    "A1_time_pressure": "The email creates a sense of time pressure.",
    "A2_intensity": "The email is emotionally intense or activated.",
    "A3_act_now": "The email pushes the reader to act immediately.",
    "W1_care": "The email shows care for the recipient's interests or well-being.",
    "W2_friendly": "The tone is friendly and approachable.",
    "W3_person_not_ticket": "The email treats the recipient as a person rather than a case, ticket, or target.",
    "D1_institutional": "The sender speaks as an organization or institution rather than as an individual.",
    "D2_impersonal": "The email is addressed to a generic recipient and assumes no personal relationship.",
    "D3_procedural": "The language is formal and procedural.",
    "D4_automated": "The email reads as an automated or template-generated message rather than something written by a person for this recipient.",
}

ITEM_TEXT_ZH = {
    "V1_negative_tone": "整体情感基调是消极的。",
    "V2_goodwill": "邮件对收件人表达善意或积极态度。",
    "V3_hostility": "邮件表达敌意、不满或怨恨。",
    "V4_complaint": "邮件表达不满、抱怨或批评。",
    "A1_time_pressure": "邮件制造时间压力。",
    "A2_intensity": "邮件的情绪强度高、被激活。",
    "A3_act_now": "邮件催促收件人立即行动。",
    "W1_care": "邮件关心收件人的利益或处境。",
    "W2_friendly": "语气友好、易于接近。",
    "W3_person_not_ticket": "邮件把收件人当成人，而不是工单/编号/目标。",
    "D1_institutional": "发件人以机构/组织的身份说话，而不是以个人身份。",
    "D2_impersonal": "邮件面向泛称的收件人，假定没有个人关系。",
    "D3_procedural": "语言正式、程序化。",
    "D4_automated": "邮件读起来像自动生成或模板生成，而不是某个人为这个收件人写的。",
}

DIMENSION_ITEMS = {name: dim["items"] for name, dim in DIMENSIONS.items()}
SCORED_ITEMS = [item for dim in DIMENSIONS.values() for item in dim["items"]]

# Collected but not part of any dimension score. V3 is retained for descriptive
# hostility prevalence; it is too rare in email corpora to support a reliable
# 1-7 scale.
DIAGNOSTIC_ITEMS = ["V3_hostility"]

ITEMS = ["V1_negative_tone", "V2_goodwill", "V3_hostility", "V4_complaint",
         "A1_time_pressure", "A2_intensity", "A3_act_now",
         "W1_care", "W2_friendly", "W3_person_not_ticket",
         "D1_institutional", "D2_impersonal", "D3_procedural", "D4_automated"]
REVERSE_ITEMS = {item for dim in DIMENSIONS.values() for item in dim["reverse"]}
ITEM_DIMENSION = {item: name for name, dim in DIMENSIONS.items() for item in dim["items"]}
ITEM_DIMENSION["V3_hostility"] = "valence"
ITEM_ROLE = {item: ("diagnostic" if item in DIAGNOSTIC_ITEMS else "scored")
             for item in ITEMS}

THREAT_COLUMNS = ["threat_binary", "threat_severity"]
RATING_COLUMNS = ["rater_id", "calib_id"] + ITEMS + THREAT_COLUMNS + ["confidence", "notes"]
TEMPLATE_COLUMNS = ["rater_id", "order", "calib_id", "subject", "body"] + \
    ITEMS + THREAT_COLUMNS + ["confidence", "notes"]
