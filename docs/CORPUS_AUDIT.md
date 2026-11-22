# 语料审计报告（mvp_v2）

- 数据目录：`D:\paper\mail\data`
- 语料：enron, llmgen, ephishgen, phishfuzzer, nest, spamassassin, turkphish_v1
- 文档总数：98609
- 生成脚本：`audit_corpora.py`

## 0. 结论摘要

- 已加载 7 个语料，共 98609 封邮件：{'enron': 33716, 'phishfuzzer': 23100, 'ephishgen': 16616, 'spamassassin': 10751, 'turkphish_v1': 7504, 'llmgen': 6776, 'nest': 146}。
- PhishFuzzer 的 7 版本/种子结构会在长度上产生漂移（词数比 long=1.66, medium=1.77, short=1.06；变体词数中位数 long=311, medium=310, short=183）。`long` 与 `medium` 的实际长度几乎相同，说明长度控制是名义上的、并不严格。它适合做外部基准/OOD，不适合直接当作“保长度、保实体”的风格改写数据。
- PhishFuzzer 变体中 92.5% 改变了 URL 集合，42.6% 改变了邮箱集合；这进一步说明它是“实体替换”而非“风格改写”。
- 精确重复率最高的是 spamassassin：42.8%。
- NEST-Phish 仅 6 个线索特征就能达到 AUC=0.878；它适合做构念效度/人工标注对照，不适合训练大模型。
- RMF-PC 的 181 个 pair id 中，181 个能在下载的 SpamAssassin 中找到（100.0%），来源可复核。

## 1. 语料清单与基本分布

| corpus | n | labels | languages | empty_text | median_words | p10_words | p90_words | median_chars | unique_groups |
|---|---|---|---|---|---|---|---|---|---|
| enron | 33716 | spam:17171; ham:16545 | en:33716 | 51 | 114.000 | 28.000 | 504.000 | 551.000 | 0 |
| llmgen | 6776 | phishing:4497; legitimate:2279 | en:3746; zh:3030 | 0 | 213.000 | 19.000 | 249.000 | 1153.000 | 0 |
| ephishgen | 16616 | legitimate:8398; phishing:8218 | en:11502; it:2701; de:2349; es:35; fr:12; sv:5 | 0 | 87.000 | 61.000 | 135.000 | 494.000 | 0 |
| phishfuzzer | 23100 | Phishing:7882; Valid:7700; Spam:7518 | en:22358; no:203; ja:105; fr:84; pt:77; da:49 | 0 | 254.000 | 115.000 | 444.000 | 1457.000 | 3300 |
| nest | 146 | legit:77; phish:69 | en:146 | 0 | 42.000 | 29.000 | 58.000 | 246.000 | 17 |
| spamassassin | 10751 | ham:6954; spam:3797 | en:10751 | 0 | 161.000 | 48.000 | 589.000 | 863.000 | 9166 |
| turkphish_v1 | 7504 | phishing:6004; legitimate:1500 | tr:7504 | 0 | 43.000 | 41.000 | 47.000 | 181.000 | 0 |

> `empty_text` 指主题和正文都为空；`unique_groups` 仅对具有分组字段的语料有意义（PhishFuzzer/NEST）。

## 2. 长度分布（压缩混淆基线）

| corpus | label | n | median_words | p10_words | p90_words | median_chars |
|---|---|---|---|---|---|---|
| enron | ham | 16545 | 132.000 | 27.000 | 498.000 | 621.000 |
| enron | spam | 17171 | 103.000 | 29.000 | 509.000 | 507.000 |
| llmgen | phishing | 4497 | 218.000 | 23.000 | 250.000 | 1164.000 |
| llmgen | legitimate | 2279 | 206.000 | 15.000 | 245.000 | 1144.000 |
| ephishgen | legitimate | 8398 | 103.000 | 73.000 | 148.000 | 581.000 |
| ephishgen | phishing | 8218 | 75.000 | 56.000 | 104.000 | 425.000 |
| phishfuzzer | Phishing | 7882 | 223.000 | 101.000 | 373.000 | 1257.500 |
| phishfuzzer | Spam | 7518 | 276.000 | 123.000 | 477.000 | 1572.500 |
| phishfuzzer | Valid | 7700 | 272.000 | 127.000 | 461.000 | 1580.000 |
| nest | phish | 69 | 38.000 | 27.000 | 47.400 | 227.000 |
| nest | legit | 77 | 47.000 | 32.200 | 62.400 | 257.000 |
| spamassassin | ham | 6954 | 155.000 | 45.000 | 566.000 | 813.000 |
| spamassassin | spam | 3797 | 173.000 | 58.000 | 616.000 | 927.000 |
| turkphish_v1 | phishing | 6004 | 43.000 | 41.000 | 47.000 | 182.000 |
| turkphish_v1 | legitimate | 1500 | 46.000 | 44.000 | 48.000 | 180.000 |

> 词数按拉丁词计；中文语料的词数会低估，实际长度请同时看 `median_chars`。

## 3. 重复与模板

| corpus | n | empty | unique_texts | exact_dup_rows | exact_dup_rate | dup_groups | max_group | prefix_groups_gt1 | max_prefix_group |
|---|---|---|---|---|---|---|---|---|---|
| enron | 33716 | 51 | 30138 | 3578 | 0.106 | 3366 | 51 | 3368 | 107 |
| llmgen | 6776 | 0 | 6776 | 0 | 0.000 | 0 | 1 | 188 | 12 |
| ephishgen | 16616 | 0 | 16616 | 0 | 0.000 | 0 | 1 | 4 | 2 |
| phishfuzzer | 23100 | 0 | 23025 | 75 | 0.003 | 73 | 3 | 166 | 9 |
| nest | 146 | 0 | 145 | 1 | 0.007 | 1 | 2 | 1 | 2 |
| spamassassin | 10751 | 0 | 6148 | 4603 | 0.428 | 4233 | 10 | 4343 | 14 |
| turkphish_v1 | 7504 | 0 | 5452 | 2052 | 0.273 | 1650 | 4 | 40 | 432 |

### 3.llmgen 前缀模板簇

| count | examples | prefix |
|---|---|---|
| 12 | llmgen-en:116; llmgen-en:121 | subject: important update: new features and enhancements to your account from: support@techsolutions.com dear valued cus |
| 10 | llmgen-en:22; llmgen-en:29 | subject: urgent: action required to secure your account from: support@securebanking.com dear valued customer, we hope th |
| 8 | llmgen-en:380; llmgen-en:2408 | subject: immediate action required: suspicious activity detected on your github account from: github security team <secu |
| 8 | llmgen-en:583; llmgen-en:692 | subject: your monthly billing statement is ready from: billing@yourserviceprovider.com dear valued customer, we hope thi |
| 8 | llmgen-en:716; llmgen-en:922 | subject: your package is ready for pickup! from: support@deliveryservice.com dear valued customer, we hope this message  |
| 7 | llmgen-en:21; llmgen-en:47 | subject: urgent: immediate action required to secure your account from: support@securebanking.com dear valued customer,  |

### 3.turkphish_v1 前缀模板簇

| count | examples | prefix |
|---|---|---|
| 432 | turkphish:11; turkphish:23 | hesap doğrulamanızı tamamlayın sayın kullanıcı, hesabınızla ilgili ciddi bir güvenlik problemi tespit edilmiştir. lütfen |
| 430 | turkphish:15; turkphish:37 | acil hesap doğrulaması sayın kullanıcı, hesabınızla ilgili ciddi bir güvenlik problemi tespit edilmiştir. lütfen aşağıda |
| 411 | turkphish:21; turkphish:27 | şifrenizi hemen değiştirin sayın kullanıcı, hesabınızla ilgili ciddi bir güvenlik problemi tespit edilmiştir. lütfen aşa |
| 399 | turkphish:17; turkphish:39 | ödül kazandınız sayın kullanıcı, hesabınızla ilgili ciddi bir güvenlik problemi tespit edilmiştir. lütfen aşağıdaki bağl |
| 395 | turkphish:9; turkphish:19 | faturanız ödenmedi sayın kullanıcı, hesabınızla ilgili ciddi bir güvenlik problemi tespit edilmiştir. lütfen aşağıdaki b |
| 377 | turkphish:31; turkphish:53 | hesap şifreniz sıfırlandı sayın kullanıcı, hesabınızla ilgili ciddi bir güvenlik problemi tespit edilmiştir. lütfen aşağ |

### 3.nest 前缀模板簇

| count | examples | prefix |
|---|---|---|
| 2 | legit77; legit81 | draft scope doc – comments requested hi emily, please review the draft scope doc for the ridgeview controls upgrade and  |

### 3.spamassassin 前缀模板簇

| count | examples | prefix |
|---|---|---|
| 14 | 20021010_spam/spam/0487.b57549dc531f50c1ff1e3356bc38b390; 20030228_spam/spam/00476.af3a29817853a5c56bae5257c2d4b742 | the government grants you $25,000! free personal and business grants " qualify for at least $25,000 in free grants money |
| 10 | 20030228_spam_2/spam_2/00793.f081690dc64c0e3bbe8c7198e9caaffc; 20030228_spam_2/spam_2/00943.41b19a950ac03c2df9e33ab75ad595d1 | the database that bill gates doesnt want you to know about!!!!! if you are struggling with ms access to manage your data |
| 8 | 20021010_easy_ham/easy_ham/2134.6cdd3f522ea2e84ddbc75ad2d594b3a5; 20021010_easy_ham/easy_ham/2176.71c9df46ed8e77885083e1b9bb7f6276 | sourceforge sells out url: http://www.askbjoernhansen.com/archives/2002/09/24.html date: 2002-09-24t02:35:59-08:00 you a |
| 8 | 20021010_easy_ham/easy_ham/2184.a85600551a54fe5470c755feb9062027; 20021010_easy_ham/easy_ham/2140.6fbc988095074138c040158b12948483 | att wireless location tracking url: http://www.askbjoernhansen.com/archives/2002/09/12.html date: 2002-09-12t04:04:00-08 |
| 8 | 20021010_easy_ham/easy_ham/2139.9ba6145b574aaa5184fbe919d9a64fb4; 20021010_easy_ham/easy_ham/2161.044851c0caf13b43b3c1ba4acee6ea8d | ufos in the sky! url: http://www.askbjoernhansen.com/archives/2002/09/20.html date: 2002-09-20t17:10:35-08:00 yesterday  |
| 8 | 20021010_easy_ham/easy_ham/2141.1297ad634fdb16e4272387250a901f5d; 20021010_easy_ham/easy_ham/2157.f89b3083fea3eaceabba7a30ca82369a | mac os x browsers url: http://www.askbjoernhansen.com/archives/2002/09/07.html date: 2002-09-07t22:25:08-08:00 rael is p |

## 4. PhishFuzzer 的分组结构

### 4.1 每组文档数

| group_size | n_groups |
|---|---|
| 7 | 3300 |

### 4.2 变体相对种子的词数比

| length_type | n | median_var_words | median_ratio | p10_ratio | p90_ratio |
|---|---|---|---|---|---|
| long | 6600 | 311.000 | 1.659 | 0.454 | 4.788 |
| medium | 6600 | 310.000 | 1.766 | 0.477 | 4.568 |
| short | 6600 | 183.000 | 1.064 | 0.342 | 2.660 |

### 4.3 变体的 URL / 邮箱集合变化

| variants_compared | url_set_changed | url_set_changed_pct | url_missing_pct | url_added_pct | email_set_changed_pct |
|---|---|---|---|---|---|
| 19800.000 | 18324.000 | 0.925 | 0.825 | 0.881 | 0.426 |

> 结论：PhishFuzzer 的 `Length_Type` 是有意操纵的变量，且实体/URL 也被替换。若拿它做“温度漂移”，长度和实体是强混淆；只能做外部基准，或在同一 `Length_Type` 内做敏感性分析。

## 5. NEST-Phish 线索效度

| cue | phish_mean | legit_mean | diff | chi2 | p |
|---|---|---|---|---|---|
| has_link | 0.667 | 0.519 | 0.147 | 2.677 | 0.102 |
| has_attachment | 0.275 | 0.247 | 0.029 | 0.042 | 0.838 |
| has_generic_greeting | 0.609 | 0.455 | 0.154 | 2.878 | 0.090 |
| has_urgent_language | 0.623 | 0.104 | 0.519 | 40.920 | 0.000 |
| has_external_sender | 0.739 | 0.182 | 0.557 | 43.532 | 0.000 |
| has_request_personal_information | 0.319 | 0.065 | 0.254 | 13.925 | 0.000 |
| total_signals | 3.232 | 1.571 | 1.660 | 67.670 | 0.000 |

- 仅线索特征的 5 折 Logistic 回归：n=146, accuracy=0.774, macro-F1=0.773, AUC=0.878。

> 这不是检测器结果，而是检查“线索标注是否几乎等价于标签”。若线索即可高分预测标签，则文本模型的高分可能只是学到了这些显式线索，而不是更细的情感温度。

## 6. 文本可学习性探针（非正式检测结果）

| corpus | n | classes | accuracy | macro_f1 | auc |
|---|---|---|---|---|---|
| enron | 6000 | 2 | 0.979 | 0.979 | 0.997 |
| llmgen | 5279 | 2 | 0.999 | 0.999 | 1.000 |
| ephishgen | 6000 | 2 | 0.974 | 0.974 | 0.998 |
| phishfuzzer | 6000 | 3 | 0.928 | 0.928 | 0.990 |
| nest | 146 | 2 | 0.856 | 0.854 | 0.930 |
| spamassassin | 6000 | 2 | 0.988 | 0.988 | 0.998 |
| turkphish_v1 | 4500 | 2 | 1.000 | 1.000 | 1.000 |

- 跨语料指纹探针：n=2546, accuracy=0.973, macro-F1=0.975。
- PhishFuzzer 的 Human vs LLM 来源探针：n=23100, accuracy=0.978, macro-F1=0.958, AUC=0.998。

> 这些探针只用于审计：说明语料/来源在文本上是否可区分，以及标签是否有明显文本信号。它们不是论文中的正式检测器，也不能替代锁定测试集。

## 7. SpamAssassin 与 RMF-PC 来源核对

| source | n | spam | ham | spam_rate |
|---|---|---|---|---|
| 20021010_easy_ham | 2551 | 0 | 2551 | 0.000 |
| 20021010_hard_ham | 250 | 0 | 250 | 0.000 |
| 20021010_spam | 501 | 501 | 0 | 1.000 |
| 20030228_easy_ham | 2501 | 0 | 2501 | 0.000 |
| 20030228_easy_ham_2 | 1401 | 0 | 1401 | 0.000 |
| 20030228_hard_ham | 251 | 0 | 251 | 0.000 |
| 20030228_spam | 501 | 501 | 0 | 1.000 |
| 20030228_spam_2 | 1398 | 1398 | 0 | 1.000 |
| 20050311_spam_2 | 1397 | 1397 | 0 | 1.000 |

- 去重前 10751 封；按归一化正文去重后 6148 封（重复 4603 封）。
- 跨归档重复组：1557 组；主要是 `20050311_spam_2` 与 `20030228_spam_2` 为同一批邮件。建模/改写抽样前必须先去重。
- RMF-PC pair id 匹配：181/181 （100.0%）。

### 7.1 跨归档重复

| group_id | archives | examples |
|---|---|---|
| md5:004cefb51e415fe5c45a3a97091bf978 | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00381.004cefb51e415fe5c45a3a97091bf978; 20050311_spam_2/spam_2/00381.004cefb51e415fe5c45a3a97091bf978 |
| md5:0082840b49157df044b094b1a0461063 | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00730.0082840b49157df044b094b1a0461063; 20050311_spam_2/spam_2/00730.0082840b49157df044b094b1a0461063 |
| md5:008606ae6538b605f82b61913814d3dd | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00491.008606ae6538b605f82b61913814d3dd; 20050311_spam_2/spam_2/00491.008606ae6538b605f82b61913814d3dd |
| md5:00878d47d9a07c1c6f9c5a8593db7f55 | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00058.00878d47d9a07c1c6f9c5a8593db7f55; 20050311_spam_2/spam_2/00058.00878d47d9a07c1c6f9c5a8593db7f55 |
| md5:008a60368b7623f11c7bb95826eb7366 | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00929.008a60368b7623f11c7bb95826eb7366; 20050311_spam_2/spam_2/00929.008a60368b7623f11c7bb95826eb7366 |
| md5:00c19304d06d2e9fd068873434f1297e | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00321.00c19304d06d2e9fd068873434f1297e; 20050311_spam_2/spam_2/00321.00c19304d06d2e9fd068873434f1297e |
| md5:00c61b5577b6fa232434c2ae62d52394 | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00741.00c61b5577b6fa232434c2ae62d52394; 20050311_spam_2/spam_2/00741.00c61b5577b6fa232434c2ae62d52394 |
| md5:00d729b279723c9ae9d8f09e171db301 | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00761.00d729b279723c9ae9d8f09e171db301; 20050311_spam_2/spam_2/00761.00d729b279723c9ae9d8f09e171db301 |
| md5:00de24076c1599b80c13f1e028094b6c | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/00819.00de24076c1599b80c13f1e028094b6c; 20050311_spam_2/spam_2/00819.00de24076c1599b80c13f1e028094b6c |
| md5:00df3a73895dd0f6a8f22434141257f6 | 20030228_spam_2; 20050311_spam_2 | 20030228_spam_2/spam_2/01046.00df3a73895dd0f6a8f22434141257f6; 20050311_spam_2/spam_2/01046.00df3a73895dd0f6a8f22434141257f6 |

## 8. LLMGen 细节

| language | n | phishing | legitimate | median_words | median_chars | starts_subject_pct | has_from_pct | exact_dup_rate |
|---|---|---|---|---|---|---|---|---|
| en | 3746 | 2477 | 1269 | 238.000 | 1296.500 | 0.975 | 0.970 | 0.000 |
| zh | 3030 | 2020 | 1010 | 26.000 | 533.000 | 0.019 | 0.014 | 0.000 |

> 实际行数与 v1.0 方案中的 3,754 不一致；以本表为准。`starts_subject_pct` / `has_from_pct` 高说明文本里保留了邮件头模板，训练/改写时需要决定是否保留。

## 9. Enron 时间与标签

| year | n | spam | spam_rate |
|---|---|---|---|
| 1999.000 | 136.000 | 0.000 | 0.000 |
| 2000.000 | 6762.000 | 0.000 | 0.000 |
| 2001.000 | 8420.000 | 45.000 | 0.005 |
| 2002.000 | 2602.000 | 1330.000 | 0.511 |
| 2003.000 | 109.000 | 109.000 | 1.000 |
| 2004.000 | 6097.000 | 6097.000 | 1.000 |
| 2005.000 | 9590.000 | 9590.000 | 1.000 |

- 2001 年及以前的 spam rate 为 0.003；2003 年及以后为 1.000。标签与年份几乎完全共线：随机切分会把“时间”当成特征，必须按年份/发件人/线程分组切分。

## 10. 对“保风格改写”的直接含义

1. **主数据源**：从 SpamAssassin 与 Enron 的 spam 中抽样；先按归一化文本去重，再按来源/模板分层。
2. **不要用 PhishFuzzer 变体做主实验**：它的长度与实体是刻意变化的；可作为外部基准或负对照。
3. **长度约束**：0.8–1.2（按词数）；中文按字符数另算。提示词见 `prompts/rewrite_prompts.md`；生成后必须用 `validate_rewrites.py` 自动校验。
4. **实体约束**：URL、邮箱、电话、金额、日期、账号/订单号、附件名、签名块必须逐字保留。
5. **禁止新增危害**：不得新增请求、承诺、凭据索取、威胁、截止日期或新链接。
6. **人工复核不可省**：自动校验只做候选筛查；最终纳入仍需双人标注 + 裁决 + κ/PABAK。

## 11. 缺失与限制

- **TREC 2007 未下载**：目录里只有协议说明页，需要签署协议后获取 `trec07p.tgz`；本审计未覆盖 TREC。
- **TurkPhish v2 未下载**：只有论文 PDF；v1 仅作泄漏案例，不作主基准。
- **NEST-Phish 仅 146 封**：所有估计都不稳定，只用于方法学/构念效度。
- **探针使用 TF-IDF + LogisticRegression**：不是 BERT/LLM 检测器，只用于语料审计。
- **中文语料**：LLMGen CN 3,030 封，未做分词/中文情感资源校验；主语种仍为英文。
- **来源/许可**：Enron 含 PII，TREC 不可再分发；使用前请核对各语料许可。
