# 检测器实现（预注册 §6）

- 脚本：`e2_detectors.py`，seed=42，bootstrap=10000，按 seed 聚类
- 训练语料：`samples/clean_spamassassin.csv`（spam 1835 / ham 4313）
- 切分：`group_id` 分组；200 个 E2 种子所在 188 组（217 行）强制进测试集；其余组按标签分层、按行数配额 60/20/20；train=3557、val=1186、test=1405
- D4 词典：环境无 nltk/vader/textblob，使用脚本内冻结的正/负情感词表 + 项目既有 URGENCY/WARMTH/threat 词表；该实现细节在任何检测器结果产出前确定
- 超参数：验证集 AUC 选 C；阈值 = 验证集 spam 分数第 10 百分位（TPR=.90），测试集与改写共用
- 漂移：特征在训练集上 z 化；mean |Δz|、|Δz|>0.5 SD 比例、原文-改写 Spearman ρ；稀疏特征在非零并集上计算，训练文档频率 <1% 的特征因 z 分不稳定而排除；D5 用其 200 维 SVD 特征
- 漂移的显著性检验用「条件 − neutral」的 seed 内配对差；检测率/flip/分数漂移用「改写 − 原文」配对差
- 族间排序：完整配对 seed 上的 evasion 配对差，bootstrap 95% CI；BH-FDR 在每个结局族内（5 条件 × 5 检测器）
