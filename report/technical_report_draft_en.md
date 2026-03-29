# TableTalk Technical Report (Draft - English)

> Note: this draft uses a synthetic debugging dataset (24 clips, calm/urgency). Final metrics should be replaced by results from real data.

## 1. Project Goal
A complete prototype pipeline was implemented with four stages:
1) audio preprocessing + feature extraction,
2) narrative tone classification,
3) automatic transcription,
4) query-based retrieval.

## 2. Task 1: Audio Processing & Feature Engineering
- Audio was normalized to 16 kHz mono.
- Peak normalization was applied.
- Extracted features: `MFCC`, `pitch`, `spectral centroid`, `energy`, `duration`.

The purpose of this stage is to convert raw audio into a consistent tabular format for both modeling and retrieval.

Figures:
- [split_distribution.png](../outputs/metrics/figures/split_distribution.png)
- [label_distribution.png](../outputs/metrics/figures/label_distribution.png)
- [duration_kde.png](../outputs/metrics/figures/duration_kde.png)

## 3. Task 2: Narrative Tone Classification
- Candidate models: `LogisticRegression`, `RandomForest`.
- Best model in this run: `LogisticRegression` (validation `macro F1=1.0`).

Test metrics:
- Accuracy: 1.0
- Macro-F1: 1.0
- Weighted-F1: 1.0

Because this is a simple synthetic dataset, separability is high. Real-world data is expected to be more challenging.

References:
- [test_metrics.json](../outputs/metrics/test_metrics.json)
- [model_selection.json](../outputs/metrics/model_selection.json)
- [confusion_matrix.png](../outputs/metrics/figures/confusion_matrix.png)

## 4. Task 3: ASR Transcription
- Model: Whisper (`tiny`).
- 24 clips were transcribed and saved.

ASR metrics (for pipeline verification):
- WER: 0.0
- CER: 0.0

This result validates the evaluation path, but it should not be treated as final quality on real data.

References:
- [transcripts.csv](../data/transcripts/transcripts.csv)
- [asr_metrics.json](../outputs/metrics/asr_metrics.json)

## 5. Task 4: Retrieval Prototype
Supported example queries:
- "calm narration >4 seconds"
- "high-energy speech"

Result files:
- [results_calm.csv](../outputs/retrieval/results_calm.csv)
- [results_high_energy.csv](../outputs/retrieval/results_high_energy.csv)
- [results.csv](../outputs/retrieval/results.csv)

## 6. Limitations & Next Steps
- Current results are from synthetic data and may not transfer to real narrative speech.
- Next steps:
  1) run on real subsets from RAVDESS / CREMA-D / Common Voice,
  2) evaluate ASR with manually prepared references,
  3) add pretrained audio embeddings for better generalization,
  4) improve retrieval with interpretable weighted scoring.
