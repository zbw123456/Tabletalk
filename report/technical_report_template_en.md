# TableTalk Technical Report (Template - English)

## 1. Project Goal
This test aims to build a runnable pipeline for narrative audio:
- audio preprocessing,
- tone/emotion classification,
- ASR transcription,
- simple audio retrieval.

Please include:
- dataset name and source link,
- sampling strategy (Applicants may work with a subset of the dataset, for example 50–200 audio recordings),
- why this sampling is reasonable (label balance, duration coverage, etc.).

## 2. Task 1: Audio Processing & Feature Engineering
### 2.1 Preprocessing
- Sampling rate normalization:
- Channel handling (mono/stereo):
- Loudness/peak normalization:
- Silence trimming strategy:
- Invalid file handling (broken/silent/too short):

### 2.2 Features
- Feature set: `MFCC`, `pitch`, `spectral centroid`, `energy`, `duration`
- Output format: one row per clip (path, label, split, feature columns)

### 2.3 Results
- Dataset size and split stats (train/val/test)
- Figures:
  - [split_distribution.png](../outputs/metrics/figures/split_distribution.png)
  - [label_distribution.png](../outputs/metrics/figures/label_distribution.png)

## 3. Task 2: Narrative Tone Classification
### 3.1 Modeling
- Candidate models:
- Selection rationale (quality, robustness, cost):

### 3.2 Evaluation
- Accuracy:
- Macro-F1:
- Weighted-F1:
- Figure:
  - [confusion_matrix.png](../outputs/metrics/figures/confusion_matrix.png)

### 3.3 Error Analysis
- Most confused classes:
- Typical failure samples:
- Possible reasons and fixes:

## 4. Task 3: AI-based Transcription
### 4.1 Model & Setup
- Model: Whisper / other
- Inference settings (model size, language, fp16 on/off):

### 4.2 Quality Assessment
- Metrics: `WER`, `CER`
- Example comparison (reference vs prediction):

## 5. Task 4: Audio Retrieval Prototype
### 5.1 Retrieval Rules
- Query fields: label, duration, energy, transcript text
- Filtering/ranking method (threshold or scoring):

### 5.2 Example Queries
- Query 1:
- Query 2:
- Query 3:

### 5.3 Output Samples
- Result files:
  - [results.csv](../outputs/retrieval/results.csv)

## 6. Bonus: Storytelling vs Conversational Speech
- Comparison dimensions: pacing, pauses, pitch dynamics, energy dynamics
- Candidate features for automatic detection:

## 7. Conclusion & Next Steps
- Strengths of current solution:
- Current limitations:
- Suggested improvements:
  - stronger audio embeddings (wav2vec2 / HuBERT)
  - better retrieval scoring
  - larger human-labeled evaluation set with cross-validation
