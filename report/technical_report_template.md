# TableTalk Technical Report (Template)

## 1. 项目目标
- 任务目标：构建叙事语音处理、分类、转写与检索原型。
- 数据来源：说明数据集名称与抽样策略（例如 50–200 条）。

## 2. Task 1：音频处理与特征工程
### 2.1 预处理
- 采样率统一：
- 声道处理：
- 归一化策略：
- 静音处理策略：

### 2.2 特征
- 使用特征：`MFCC`、`pitch`、`spectral centroid`、`energy`、`duration`
- 输出结构：每条音频一行，字段说明

### 2.3 结果
- 数据规模与切分统计（train/val/test）
- 图表引用：
  - [split_distribution.png](../outputs/metrics/figures/split_distribution.png)
  - [label_distribution.png](../outputs/metrics/figures/label_distribution.png)

## 3. Task 2：叙事语气分类
### 3.1 模型方法
- 候选模型：
- 选择依据：

### 3.2 评估结果
- Accuracy：
- Macro-F1：
- Weighted-F1：
- 图表引用：
  - [confusion_matrix.png](../outputs/metrics/figures/confusion_matrix.png)

### 3.3 误差分析
- 易混淆类别：
- 原因讨论：

## 4. Task 3：AI 转写
### 4.1 模型与配置
- 模型：Whisper / 其他
- 推理参数：

### 4.2 质量评估
- 评价指标：`WER`、`CER`
- 示例输出分析：

## 5. Task 4：音频检索原型
### 5.1 检索规则
- 查询字段：类别、时长、能量、文本
- 过滤与排序策略：

### 5.2 示例查询
- Query 1：
- Query 2：
- Query 3：

### 5.3 输出示例
- 结果文件：
  - [results.csv](../outputs/retrieval/results.csv)

## 6. Bonus：Storytelling vs Conversational
- 对比维度：语速、停顿、音高变化、能量动态
- 可自动化识别的候选特征：

## 7. 结论与改进
- 当前方案优点：
- 局限性：
- 下一步优化：
  - 更强的音频嵌入（如 wav2vec2 / HuBERT）
  - 更完善的检索打分函数
  - 人工标注集扩充与交叉验证
