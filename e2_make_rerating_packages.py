#!/usr/bin/env python3
"""Build targeted rerating packages after the E2 QC REVISE.

Each package contains the codebook, column guide, a rater-specific note and a
640-row sheet with ONLY the flagged item columns left blank. Previous scores
are not shown, so the rerating is blind to the first pass.

  R1: V1_negative_tone
  R2: V1_negative_tone, D1_institutional, D2_impersonal, D3_procedural
  R3: V1_negative_tone, W1_care, W2_friendly, W3_person_not_ticket

Outputs e2_ratings/packages/rerate_<rater>/
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "e2_ratings/e2_annotation_600_template.csv"
CODEBOOK = ROOT / "annotation/RATING_CODEBOOK_v1.3.md"
COLUMNS = ROOT / "annotation/COLUMN_GUIDE_v1.3.md"
OUT = ROOT / "e2_ratings/packages"

TARGETS = {
    "R1": ["V1_negative_tone"],
    "R2": ["V1_negative_tone", "D1_institutional", "D2_impersonal", "D3_procedural"],
    "R3": ["V1_negative_tone", "W1_care", "W2_friendly", "W3_person_not_ticket"],
}

V1_NOTE = """## V1_negative_tone：整体基调是否消极

**高分 = 更消极**：不满、敌意、指责、抱怨、威胁性的负面语气。
**低分 = 中性或积极**：促销、通知、礼貌请求、平静陈述。

不是：邮件是否对收件人不利、是否是垃圾/诈骗、是否在威胁（威胁另由 threat 编码）、你自己的情绪反应。

| 分数 | 表现 |
|---|---|
| 1–2 | 中性、促销、礼貌，没有负面情绪 |
| 3–4 | 轻微不满、抱怨、防御性或略带负面 |
| 5–6 | 明确的消极、敌意、指责、不满 |
| 7 | 持续的敌意、攻击、侮辱 |

常见误读：

- “This is NOT spam” 这类**否认/防御性措辞不等于 V1=5**。如果正文其余部分是平静的促销或说明，V1 应在 1–3。
- 感叹号和大写主要影响 A2_intensity，不自动提高 V1。
- 邮件是诈骗/广告，不等于语气消极；V1 只看表达出来的负面效价。
- 3 和 4 是正常可用的中间档，不要跳过。
"""

R2_NOTE = """## 本轮四个条目

### V1_negative_tone

高分 = 整体基调更消极（不满、敌意、指责、抱怨）。低分 = 中性或积极。
促销、通知、礼貌请求通常 1–3；“This is NOT spam” 的防御性否认本身不构成 5–7；只有正文整体确实消极/敌意时才给高 V1。

### D1_institutional：机构化程度

高分 = 像机构告示、自动通知、群发广告：机构口吻、程序性、没有个人关系。
低分 = 像个人写给个人的留言：有个人上下文、具体关系、自然对话。

### D2_impersonal：无人称程度

高分 = 没有具体收件人、群发口吻、对“账户/客户/用户”说话。
低分 = 对具体个人说话，有姓名、个人指代或私人上下文。

### D3_procedural：程序化语言

高分 = 步骤说明、表单、编号、条款、自动流程、清单式指令。
低分 = 自然叙述、对话式表达。

四个条目都用完整 1–7；**4 = 混合或无法判断，但不能把所有邮件都放 4/5/6**。

对照例：

| 邮件 | V1 | D1 | D2 | D3 |
|---|---|---|---|---|
| “Dear Customer, your account will be suspended unless you verify your details at…” | 2–3 | 5–6 | 6 | 4–5 |
| “Hi John, I saw your talk last week—could you send the slides?” | 1 | 1–2 | 1–2 | 1–2 |
| 群发促销 “Act now, click here for 50% off” | 1–2 | 5 | 5–6 | 3–4 |
| 个人口吻的求助/链式信 “Hello, I am a lawyer and need your help…” | 1–3 | 2–3 | 2–3 | 2–3 |

上一轮的主要问题是区分度不足：D1 约 93% 给 5、D2 约 92% 给 6、V1 只用了 1–2。本轮请真正区分“个人留言”和“机构群发”，不要默认给同一个分。
"""

R3_NOTE = """## 本轮四个条目：V1 + 三个 warmth 条目

### V1_negative_tone

高分 = 整体基调更消极（不满、敌意、指责、抱怨）；低分 = 中性或积极。
促销、通知、礼貌请求通常 1–3；防御性的 “This is NOT spam” 不等于高 V1。

### warmth 维度（W1 / W2 / W3）

| 条目 | 高分表现 | 低分表现 |
|---|---|---|
| W1_care | 明确关心收件人的利益、处境或感受 | 只关心发件人自己的目的 |
| W2_friendly | 友好、亲切、轻松的社交语气 | 冷淡、生硬、公事公办 |
| W3_person_not_ticket | 把收件人当具体的人：称呼、个人指代、私人上下文 | 把收件人当账户、记录、工单、群发对象 |

| 分数 | 表现 |
|---|---|
| 1–2 | 冷淡、交易性，没有任何关心或友好 |
| 3–4 | 常规礼貌（please / thank you 只是礼貌，本身不等于温暖） |
| 5–6 | 明确的关心、友好、个性化 |
| 7 | 强烈温暖：私人关系、情感支持、非常亲切的口吻 |

常见误读：

- 垃圾邮件也可以很礼貌；**礼貌（3–4）不等于温暖（5–7）**。
- “Dear Customer” 是模板称呼，不能仅凭称呼给高 W3；要看有没有把对方当具体的人。
- W1/W2/W3 分开判断：一封邮件可以很友好但完全不关心你的利益，也可以很关心但语气公事公办。

### 关于评分漂移

上一轮 warmth 在批次内越往后越低，三个 warmth 条目同步下降。这说明评分标准随进度漂移，而不是邮件本身变了。本轮请：

1. 分 4 次完成，每次一个区块（160 行）；每次开始前重读上面的锚点。
2. 只按锚点判断当前这一封，不要用上一封邮件作参照。
3. 完成后检查三个 warmth 条目的各区块均值是否大致稳定；如果仍然越往后越低，请休息后按锚点重做靠后的区块。
"""

NOTES = {"R1": V1_NOTE, "R2": R2_NOTE, "R3": R3_NOTE}


def readme(rater, targets):
    cols = "、".join(f"`{c}`" for c in targets)
    return f"""# E2 定向重评任务 — {rater}

首轮 QC 未通过，本包只重评被标记的条目。**不要重评其他条目，也不要参考首轮分数**（本表未提供）。

## 文件

- `e2_600_{rater}_rerate.csv`：**要填的表**，640 行。只留空了 {cols} 这几列；`rater_id` 已预填为 `{rater}`。
- `RERATE_NOTE.md`：本轮重点条目的判定锚点和常见误读，先读这一份。
- `RATING_CODEBOOK_v1.3.md`、`COLUMN_GUIDE_v1.3.md`：完整手册与逐列说明（条目定义不变）。

## 要求

1. 对 640 行的目标列逐行填 1–7 的整数；其他列不要动、不要新增列。
2. 不要修改 `rater_id`、`sheet_id`、`block`、`order`、`calib_id`、`subject`、`body`，不要增删行或排序。
3. 每一行独立评分；表中有少量重复邮件，两行都要独立评，不要对比或复制。
4. 目标列必须使用完整量表：每个条目至少 4 个不同分值，众数占比低于 .80、标准差不低于 .50；不要只给 1–2 或只给 5–6。
5. 分几次完成，每次开始前重读 `RERATE_NOTE.md` 的锚点，避免标准随进度漂移。

## 交回

另存为 `e2_600_{rater}_rerate_filled.csv`（CSV，UTF-8），不要覆盖原表。

交回前自检：

- 640 条记录，列数与原表一致；
- 目标列无空值，全部为 1–7 整数；
- `rater_id` 全部为 `{rater}`；
- 每个目标列至少 4 个不同分值。

如有疑问，先停下来询问任务发布者，不要自行改动评分规则。
"""


def main():
    template = pd.read_csv(SRC, encoding="utf-8-sig")
    assert template.shape == (640, 23)

    for rater, targets in TARGETS.items():
        folder = OUT / f"rerate_{rater}"
        folder.mkdir(parents=True, exist_ok=True)
        sheet = template[["rater_id", "sheet_id", "block", "order", "calib_id",
                          "subject", "body"] + targets].copy()
        sheet["rater_id"] = rater
        for column in targets:
            sheet[column] = ""
        sheet.to_csv(folder / f"e2_600_{rater}_rerate.csv", index=False,
                     encoding="utf-8-sig")
        (folder / "README_FIRST.md").write_text(readme(rater, targets), encoding="utf-8")
        (folder / "RERATE_NOTE.md").write_text(NOTES[rater], encoding="utf-8")
        (folder / "RATING_CODEBOOK_v1.3.md").write_text(
            CODEBOOK.read_text(encoding="utf-8"), encoding="utf-8")
        (folder / "COLUMN_GUIDE_v1.3.md").write_text(
            COLUMNS.read_text(encoding="utf-8"), encoding="utf-8")

        check = pd.read_csv(folder / f"e2_600_{rater}_rerate.csv", encoding="utf-8-sig")
        assert check.shape == (640, 7 + len(targets))
        assert (check["rater_id"] == rater).all()
        assert check[targets].isna().all().all()
        print(f"{folder.relative_to(ROOT)}: targets={targets}, 640 rows")


if __name__ == "__main__":
    main()
