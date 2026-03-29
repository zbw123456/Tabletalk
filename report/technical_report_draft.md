# TableTalk Technical Report (Draft)

> 说明：本版结果基于 RAVDESS 子集（200 条，8 类情绪，每类 25 条）。

## 1. 项目目标
本项目先搭了一条能稳定跑通的原型链路，覆盖四个环节：
1) 音频预处理与特征提取；
2) 叙事语气分类；
3) 自动转写；
4) 条件检索。

## 2. Task 1：音频处理与特征工程
- 采样率统一为 16kHz，单声道处理。
- 进行了峰值归一化。
- 提取特征：`MFCC`、`pitch`、`spectral centroid`、`energy`、`duration`。
- 数据划分：train=140，val=20，test=40。

这一步的目标是把原始音频整理成可直接建模的表格数据，保证后续训练与检索输入一致。

数据统计图：
- [split_distribution.png](../outputs/metrics/figures/split_distribution.png)
- [label_distribution.png](../outputs/metrics/figures/label_distribution.png)
- [duration_kde.png](../outputs/metrics/figures/duration_kde.png)

## 3. Task 2：叙事语气分类
- 候选模型：`LogisticRegression`、`RandomForest`。
- 本次最优模型：`LogisticRegression`（验证集 `macro F1=0.3589`）。

多类别情绪识别在当前轻量特征+传统模型设定下难度较高，结果可作为后续优化的基线。

测试集结果：
- Accuracy: 0.35
- Macro-F1: 0.3427
- Weighted-F1: 0.3427

参考文件：
- [test_metrics.json](../outputs/metrics/test_metrics.json)
- [model_selection.json](../outputs/metrics/model_selection.json)
- [confusion_matrix.png](../outputs/metrics/figures/confusion_matrix.png)

## 4. Task 3：AI 转写
- 模型：Whisper (`tiny`)。
- 共处理 200 条音频并输出转写文本。

ASR 评估（当前状态）：
- WER: N/A
- CER: N/A
- 原因：尚未提供与当前 200 条音频一一对应的人工参考文本。

已生成待标注模板：`data/transcripts/references_template.csv`。

参考文件：
- [transcripts.csv](../data/transcripts/transcripts.csv)
- [asr_metrics.json](../outputs/metrics/asr_metrics.json)

## 5. Task 4：音频检索原型
已支持示例查询：
- "calm narration >4 seconds"
- "high-energy speech"

当前检索采用规则过滤 + 简单打分，优点是可解释、实现快；缺点是对复杂语义的覆盖有限。

结果文件：
- [results_calm.csv](../outputs/retrieval/results_calm.csv)
- [results_high_energy.csv](../outputs/retrieval/results_high_energy.csv)
- [results.csv](../outputs/retrieval/results.csv)

## 6. 局限性与下一步
- 当前模型仍以传统手工特征为主，多类别区分能力有限。
- 下一步建议：
  1) 基于 `references_template.csv` 完成小规模人工标注并计算 WER/CER；
  2) 引入预训练音频嵌入（如 wav2vec2/HuBERT）提升分类效果；
  3) 做类别不平衡与错误样本分析，优化特征与模型；
  4) 检索模块加入可解释打分与多条件加权排序。
