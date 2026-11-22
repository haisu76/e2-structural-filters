# 保内容 / 控风格的邮件改写提示词（v1）

> 版本：2026-11-02 ｜ 适用：SpamAssassin、Enron 等公开语料上的防御性研究
> 配套脚本：`../validate_rewrites.py` ｜ 语料审计：`../CORPUS_AUDIT.md`

---

## 0. 这份提示词解决什么问题

现有 RMF-PC 的 L2/L3 改写有两个混淆：

1. **压缩**：L2 有效对的长度中位数只有原文的 0.65；
2. **删链接/换实体**：L2 平均少 1.24 条 URL，L3 大量返回原文。

因此“温度漂移”可能只是“内容变短 + 链接被删”。这套提示词把改写约束成：

> **只改“怎么说”，不改“说什么”；长度、实体、URL、请求、承诺、截止日期全部锁定。**

它同时提供两类条件：

| 条件 | 目的 | 允许变化 | 不允许变化 |
|---|---|---|---|
| `neutral` | 改写噪声对照 | 用词、句式 | 语气、长度、实体、请求 |
| `warmer` | 提高人际温度 | 问候、礼貌、共情、合作性措辞 | 紧迫度、事实、请求、长度 |
| `colder` | 降低人际温度 | 正式/机构化、去人称、减少寒暄 | 敌意、威胁、事实、请求、长度 |
| `more_urgent` | 提高唤醒/紧迫 | 时间压力措辞、短句、行动号召 | 新截止日期、新威胁、新请求、长度 |
| `less_urgent` | 降低唤醒/紧迫 | 平静措辞、缓和语气 | 删除请求、改变明确截止日期、长度 |

`neutral` 是关键的负对照：如果连“同语气改写”都会造成温度分数漂移，说明测量工具本身不稳；这时不能把 `warmer/colder` 的差异解释为操纵效应。

---

## 1. 输入 / 输出协议

### 1.1 输入 JSON

```json
{
  "condition": "warmer",
  "language": "en",
  "subject": "Your account will be locked",
  "body": "Dear customer, we noticed unusual activity on your account. To avoid suspension, verify your identity at https://example.com/verify within 24 hours. Thank you.",
  "protected_spans": [
    "https://example.com/verify",
    "24 hours"
  ],
  "notes": "spam; action=verify identity; deadline=24 hours"
}
```

- `protected_spans` 由预处理脚本用正则抽取：URL、邮箱、电话、金额、日期/时限、账号/订单/发票号、附件名、签名块。
- `notes` 只给生成管线看，不应进入改写正文。
- 一次只生成一个条件；不要在同一次调用里生成多个条件，否则模型会互相污染。

### 1.2 输出 JSON

```json
{"subject": "...", "body": "..."}
```

- 只允许这两个键。
- 不要输出解释、markdown 代码围栏、前后缀、`notes`、`explanation`。
- 不要把原文和改写并排输出。

---

## 2. 共享 System Prompt

把下面这段作为所有条件的 system message。它是“硬约束”，条件提示词只负责“往哪个方向改”。

```text
You are a careful email editor working on a research dataset of spam and phishing emails. Your task is to rewrite an email so that its writing style changes while its content, structure, and real-world referents stay fixed.

HARD CONSTRAINTS (all are mandatory):

1. Preserve meaning. Every proposition, request, promise, threat, condition, deadline, and call to action must remain the same. Do not soften a mandatory request into an optional one, and do not turn a request into a statement.
2. Preserve length. The rewritten body must be between 80% and 120% of the original body's word count. Do not summarize, expand, or merge/split paragraphs to change length.
3. Preserve protected strings exactly, character for character: URLs, email addresses, phone numbers, monetary amounts, dates, account/order/invoice numbers, attachment filenames, sender names, signatures, and any text inside quotation marks. Do not add, remove, shorten, mask, or alter them. Copy each URL exactly as it appears, including its path and query string.
4. Preserve structure. Keep the same paragraph order and approximately the same number of sentences. Keep the greeting and closing.
5. Do not add new content. Do not invent facts, names, numbers, deadlines, links, attachments, contact channels, requests, promises, threats, or requests for credentials.
6. Do not remove content. Do not drop any request, qualifier, condition, link, or signature.
7. Change the wording. Do not copy whole sentences verbatim, except protected strings and short fixed phrases. At least half of the non-protected sentences should be noticeably reworded.
8. Same language. Do not translate. Keep the original register unless the condition explicitly changes register.
9. Preserve formatting. Keep line breaks and paragraph structure approximately as they are. Do not convert plain text to HTML or Markdown.
10. Output only a JSON object with exactly two keys: "subject" and "body". No markdown fences, no explanation, no preamble, no note.
11. Rewrite every sentence. Do not copy any sentence verbatim; at least half of the non-protected sentences must be clearly reworded. Returning the original or a near-copy is a failure.
12. Do not add or remove date or time words (today, tomorrow, tonight, this week, soon, etc.) or numbers/codes that are not already in the original. This applies to the subject as well.
13. The subject must be a rephrasing of the original subject, not a new claim. Keep all codes and numbers exactly.

If you cannot satisfy a constraint, still produce the best possible rewrite; do not explain the failure in the output.
```

> 说明：第 3 条里的 `protected strings` 是硬约束；第 7 条是“确实改写”的下限。自动校验器会分别检查这两类。

---

## 3. 条件提示词

以下每段都作为 user message 的“任务段”，后面再接第 1.1 节的输入 JSON。

### 3.1 `neutral`：同语气改写（对照）

```text
CONDITION: neutral paraphrase control.

Rewrite the email in different words while keeping the same tone, register, formality, warmth, and urgency. A reader should not be able to tell which text is the original and which is the rewrite based on tone alone.

Do not make the email warmer, colder, more urgent, or less urgent. Do not add or remove politeness markers. Change wording and sentence structure only. You must still rewrite the body; returning the original or a near-copy is a failure.
```

### 3.2 `warmer`：提高人际温度

```text
CONDITION: warmer.

Increase interpersonal warmth by one moderate, human-plausible step. You may:
- use a friendlier greeting or acknowledge the recipient by role;
- add empathy, appreciation, or cooperative phrasing;
- prefer "we can help you" over "you must";
- use "please" and "thank you" where natural.

Do NOT:
- increase urgency, fear, or time pressure;
- add new compliments, personal details, promises, discounts, or requests;
- add new contact channels or links;
- become overly familiar or unprofessional.

Make changes in at least three places (greeting, body, closing); a one-word swap is not enough. Keep the same request, facts, deadline, and length band.
```

### 3.3 `colder`：降低人际温度

```text
CONDITION: colder.

Decrease interpersonal warmth by one moderate step. Shift toward a formal, impersonal, institutional register. You may:
- reduce or remove small talk and warm greetings;
- use impersonal or passive constructions;
- prefer procedural wording over cooperative wording;
- keep the closing brief.

Do NOT:
- become hostile, threatening, rude, or insulting;
- add new consequences, penalties, or deadlines;
- remove the request or any required information;
- change the level of urgency.

Rewrite at least half of the body, not just the greeting. Keep the same request, facts, deadline, and length band.
```

### 3.4 `more_urgent`：提高唤醒/紧迫

```text
CONDITION: more_urgent.

Increase the sense of urgency by one moderate, human-plausible step, using ONLY the deadline, consequence, and facts already stated in the original. You may:
- move the deadline or consequence earlier in the sentence;
- use time-pressure wording (e.g., "act now", "before it is too late", "immediately") when it does not introduce a new fact;
- use shorter sentences and a more direct rhythm;
- repeat the existing call to action once.

Do NOT:
- invent a new deadline, time window, penalty, threat, or consequence;
- add a new request, link, attachment, phone number, or credential request;
- add more than two exclamation marks beyond the original;
- become a caricature or obviously manipulative.

Express urgency in the body, not only in the subject, and rewrite at least half of the sentences. Keep the same request, facts, and length band.
```

### 3.5 `less_urgent`：降低唤醒/紧迫

```text
CONDITION: less_urgent.

Decrease the sense of urgency by one moderate step while keeping the same request and any explicitly stated deadline. You may:
- use calmer, softer wording;
- use slightly longer, more measured sentences;
- add a brief polite frame (e.g., "when you have a moment") as long as it does not contradict an explicit deadline;
- keep the same call to action.

Do NOT:
- remove the request or turn a mandatory request into an optional suggestion;
- change or remove an explicitly stated deadline;
- add new facts, promises, or contact channels;
- become so vague that the action is no longer clear.

Remove urgency wording from the body while keeping the request, and rewrite at least half of the sentences. Keep the same request, facts, and length band.
```

### 3.6 可选扩展：`formal` / `casual`

如果后续要把“距离感/机构化”作为独立维度，可加两个条件；它们与 `warmer/colder` 有重叠，**不要在同一批实验里混用**。

```text
CONDITION: formal.

Increase formality by one moderate step: use complete sentences, avoid contractions and slang, prefer neutral professional vocabulary, and keep a respectful distance. Do not add bureaucratic filler that changes the length band. Do not add new facts or requests.
```

```text
CONDITION: casual.

Decrease formality by one moderate step: use contractions and conversational phrasing, shorten the greeting, and use a relaxed but still respectful tone. Do not add slang that changes the meaning, and do not add new facts or requests.
```

---

## 4. L2（开源模型）单轮模板

L2 用开源模型（如 Qwen/Llama 系列）生成，每个 `(email, condition)` 独立调用。建议：

- `temperature=0.7`, `top_p=0.95`, `max_tokens ≈ 1.5 × 原文 token 数 + 200`
- 每个 pair 生成 **3 个候选**（不同 random seed），全部保留，交给校验器筛选
- 不做多轮自我修订（开源模型容易越改越短）
- 记录：模型名、版本/commit、量化方式、prompt hash、seed、temperature、top_p

User message 结构：

```text
{condition_prompt}

INPUT:
{input_json}

Return only:
{"subject": "...", "body": "..."}
```

---

## 5. L3（商用模型）有界自适应模板（预算 3 次）

L3 对应 RMF-PC 的“预算 3 次查询”设定。每次尝试都**从原文重写**，不要从上一版改写，否则会累积漂移。

### 5.1 第 1 次：基础条件

与 L2 相同：system prompt + condition prompt + input JSON。

### 5.2 第 2/3 次：校验反馈修复

把校验器的失败原因原样填入 `{failures}`：

```text
Your previous candidate failed the automatic checks below.

FAILURES:
{failures}

Rewrite the ORIGINAL email again from scratch. Do not edit the previous candidate.

Fix only the listed failures:
- If the word-count ratio is outside 0.80–1.20: adjust wording to return to the band without adding or removing content.
- If a protected string is missing or changed: copy it exactly from the original, character for character.
- If a new request, threat, deadline, link, phone number, or credential term appears: remove it.
- If a meta-preamble or explanation appears: output only the JSON object.
- If the text is too similar to the original: reword more sentences, but keep protected strings and meaning.

Do not change anything else. Keep the same condition: {condition}.

Return only:
{"subject": "...", "body": "..."}
```

### 5.3 停止规则

- 任一候选通过全部硬校验 → 停止。
- 3 次仍未通过 → 保留全部 3 个候选与失败原因，标记为 `rejected_after_budget`。
- **不要**把“预算用尽”当成“攻击成功”；分母仍然只算人工纳入且确实改写的样本。

---

## 6. Few-shot 示例

### 6.1 原文（spam 风格，示例链接不可点击）

```text
Subject: Your account will be locked
Body: Dear customer, we noticed unusual activity on your account. To avoid suspension, verify your identity at https://example.com/verify within 24 hours. Thank you.
```

受保护串：`https://example.com/verify`、`24 hours`。

### 6.2 `neutral`

```json
{"subject": "Account locked notice", "body": "Hello, we have detected unusual activity on your account. To prevent suspension, please confirm your identity at https://example.com/verify within 24 hours. Thank you."}
```

- 词数比 ≈ 1.04；URL 与时限保留；请求仍是“验证身份”；语气基本不变。

### 6.3 `warmer`

```json
{"subject": "We want to help keep your account secure", "body": "Hello, we noticed unusual activity on your account. To help you avoid suspension, please confirm your identity at https://example.com/verify within 24 hours. Thank you."}
```

- 词数比 ≈ 1.08；增加了问候与“help you”，但没有新增事实、请求或紧迫度。

### 6.4 `colder`

```json
{"subject": "Account activity notification", "body": "Unusual activity has been detected on your account. Identity verification is required at https://example.com/verify within 24 hours to prevent suspension."}
```

- 词数比 ≈ 0.92；去掉了寒暄与感谢，保留请求和时限；没有敌意或新威胁。

### 6.5 `more_urgent`

```json
{"subject": "Action required: account suspension pending", "body": "Dear customer, unusual activity was detected on your account. Your account will be suspended unless you verify your identity at https://example.com/verify within 24 hours. Act now."}
```

- 词数比 ≈ 1.16；紧迫度上升，但没有新截止日期、新威胁或新请求。

### 6.6 `less_urgent`

```json
{"subject": "Account activity notice", "body": "Dear customer, we noticed unusual activity on your account. When possible, please verify your identity at https://example.com/verify within 24 hours to avoid suspension. Thank you."}
```

- 词数比 ≈ 1.12；语气缓和，但请求和明确时限仍在。

### 6.7 反例：不要这样输出

```text
Certainly! Here is the rewritten email:

Subject: ...
Body: ...
```

失败原因：meta-preamble + 非 JSON + 可能删掉 URL。

```text
Subject: Account alert
Body: Please verify your account here: https://example.com/verify
```

失败原因：压缩（词数比过低）、删掉“24 hours”、删掉 suspension 条件。

---

## 7. 生成参数与记录

每个候选必须记录以下字段（建议直接落到 `generation_log.jsonl`）：

| 字段 | 说明 |
|---|---|
| `pair_id` | 原文 id（如 `20021010_spam/spam/0043...`） |
| `condition` | neutral / warmer / colder / more_urgent / less_urgent |
| `model` | 模型名 + 版本/commit |
| `quantization` | 量化方式（L2 必填） |
| `prompt_sha256` | system + condition + input 的哈希 |
| `seed` | 随机种子 |
| `temperature` / `top_p` | 解码参数 |
| `attempt` | 1–3（L3） |
| `raw_output` | 原始输出，不要只存解析后的 JSON |
| `parsed_ok` | 是否能解析为 `{subject, body}` |
| `validator_pass` | 是否通过自动校验 |
| `validator_failures` | 失败原因列表 |
| `generated_at` | 时间戳 |

---

## 8. 自动校验清单（与 `validate_rewrites.py` 对应）

| 检查 | 成功标准 | 失败级别 |
|---|---|---|
| 非空 | body 非空 | fail |
| 不是原文 | 归一化后不完全相同 | fail |
| 确实改写 | 字符相似度 < 0.95；Jaccard < 0.95 | fail / warn |
| 长度 | 词数比 0.80–1.20（短文本或中文按字符） | fail |
| URL | 原文每条 URL 逐字出现；不新增 URL | fail |
| 邮箱/电话 | 原文集合不丢失；不新增 | fail |
| 金额/日期/时限 | 原文集合不丢失；不新增 | fail |
| 账号/订单/发票号 | 原文集合不丢失；不新增 | fail |
| 附件名 | 原文集合不丢失；不新增 | fail |
| 主题 | 提供主题时不得缺失；主题中的实体不得丢失或新增 | fail |
| meta-preamble | 无“Certainly / Here is the rewritten email / Subject: ...”等 | fail |
| 新请求 | 不新增行动类型（click/reply/call/pay/download/verify/submit） | fail |
| 新凭据索取 | 不新增 password/SSN/card/OTP 等 | fail |
| 新威胁 | 不新增 suspend/suspension/terminate/legal/penalty 等 | fail |
| 新截止日期 | 不新增日期或“within N hours/days” | fail |
| 语言 | 主要文字系统不变 | fail |
| 请求/威胁/承诺/凭据被删 | 原文概念应仍然出现 | warn（需人工复核） |
| 否定词数量变化 | 极性可能改变 | warn（需人工复核） |
| HTML 标记 | 输入应为纯文本 | warn（先转纯文本，否则长度/实体不可靠） |
| 语气方向 | 与 condition 预期方向一致 | warn（实验诊断，不作为纳入硬条件） |

> `pass` 只表示通过硬校验；`warnings` 里的语义信号仍需人工复核。最终纳入要判断：语义保持、实体保持、无新增危害、流畅、确实改写。
>
> 运行：
>
> ```bash
> python validate_rewrites.py --pairs generation_log.jsonl --out validation --show 50
> ```
>
> 输入至少需要 `original` 与 `paraphrase`；建议同时提供 `id`、`condition`、`subject_original`、`subject_paraphrase`。详见 `../README.md`。

---

## 9. 常见失败模式与预防

| 失败模式 | 模型通常怎么做 | 提示词对策 | 校验器对策 |
|---|---|---|---|
| 压缩/摘要 | 把长邮件改短 | 明确 80–120% 词数；禁止 summarize | `length_ratio` |
| 删链接 | 认为链接“不重要” | 受保护串逐字复制；禁止增删 | `urls_missing` |
| 换实体 | 把品牌/账号换成通用词 | 受保护串；禁止改专名 | `urls/emails/phones/money/dates/ids/attachments_missing` |
| meta-preamble | “Certainly! Here is...” | 只输出 JSON；无解释 | 前缀正则 |
| 新增请求 | 加“call us”“click here” | 禁止新增行动/渠道 | `new_action` |
| 新增紧迫/威胁 | 为了“更 urgent”编新 deadline | 只能用原文已有事实 | `dates_added` / `time_windows_added` / `new_threat` |
| 删除请求/威胁/否定 | 为了“更温暖/更平静”删掉条件或否定 | 逐句保留命题；不得把 mandatory 改成 optional | `*_maybe_removed` / `negation_count_changed`（warn） |
| 语气漫画化 | 大量感叹号、全大写 | “one moderate step”；感叹号上限 | 诊断指标 |
| 语言漂移 | 把英文改成中文/混合 | 同语言；不翻译 | `language_mismatch` |
| 格式破坏 | 把段落合成一段 | 保留段落/换行 | 段落数比较（warn） |
| HTML 输入 | 直接对 HTML 改写 | 预处理转纯文本 | `html_markup_detected`（warn） |

---

## 10. 安全与伦理边界

- 这些提示词只用于**公开语料上的防御性研究**，不得用于真实邮件投递、绕过生产过滤器或训练攻击工具。
- 不得新增 URL、邮箱、电话、附件或凭据索取；只能逐字保留原文已有的内容。
- 生成结果存放在隔离目录，不渲染 HTML，不自动打开链接。
- 论文中报告的是“在受控改写下的检测器鲁棒性/测量漂移”，不是“如何攻击真实用户”。
- 若语料含真实个人信息（Enron），改写样本也不得公开原始文本或可识别实体。

---

## 11. 一句话版本（给管线用）

> 保持原意、长度（0.8–1.2）、全部 URL/实体/请求/截止日期不变；只改措辞和语气；按条件做“一格”温度移动；输出纯 JSON；每次从原文重写；自动校验失败就带着失败原因重试，最多 3 次。
