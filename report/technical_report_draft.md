# TableTalk Technical Report (Draft)

> 本草稿基于当前联调数据（synthetic calm/urgency, 24 条）自动填写，后续可替换为真实数据实验结果。

## 1. 项目目标
本项目实现了一个完整的叙事语音原型流程：
1) 音频预处理与特征提取；
2) 叙事语气分类；
3) AI 自动转写；
4) 条件检索与结果返回。

## 2. Task 1：音频处理与特征工程
- 采样率统一为 16kHz，单声道处理。
- 进行了峰值归一化。
- 提取特征：`MFCC`、`pitch`、`spectral centroid`、`energy`、`duration`。

数据统计图：
- [split_distribution.png](../outputs/metrics/figures/split_distribution.png)
- [label_distribution.png](../outputs/metrics/figures/label_distribution.png)
- [duration_kde.png](../outputs/metrics/figures/duration_kde.png)

## 3. Task 2：叙事语气分类
- 候选模型：`LogisticRegression`、`RandomForest`。
- 本次最优模型：`LogisticRegression`（验证集 `macro F1=1.0`）。

测试集结果：
- Accuracy: 1.0
- Macro-F1: 1.0
- Weighted-F1: 1.0

参考文件：
- [test_metrics.json](../outputs/metrics/test_metrics.json)
- [model_selection.json](../outputs/metrics/model_selection.json)
- [confusion_matrix.png](../outputs/metrics/figures/confusion_matrix.png)

## 4. Task 3：AI 转写
- 模型：Whisper (`tiny`)。
- 共处理 24 条音频并输出转写文本。

ASR 评估（当前联调参考文本）：
- WER: 0.0
- CER: 0.0

参考文件：
- [transcripts.csv](../data/transcripts/transcripts.csv)
- [asr_metrics.json](../outputs/metrics/asr_metrics.json)

## 5. Task 4：音频检索原型
已支持示例查询：
- "calm narration >4 seconds"
- "high-energy speech"

结果文件：
- [results_calm.csv](../outputs/retrieval/results_calm.csv)
- [results_high_energy.csv](../outputs/retrieval/results_high_energy.csv)
- [results.csv](../outputs/retrieval/results.csv)

## 6. 局限性与下一步
- 当前结果基于 synthetic 数据，和真实叙事语音存在域差异。
- 下一步建议：
  1) 接入 RAVDESS/CREMA-D/Common Voice 真实子集；
  2) 使用人工标注参考文本评估真实 WER/CER；
  3) 引入预训练音频嵌入提升泛化；
  4) 检索模块加入可解释打分与多条件加权排序。
