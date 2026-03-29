# TableTalk Technical Report (Draft - English)

> Note: this version is based on a RAVDESS subset (200 clips, 8 emotion classes, 25 clips each).

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
- Data split: train=140, val=20, test=40.

The purpose of this stage is to convert raw audio into a consistent tabular format for both modeling and retrieval.

Figures:
- [split_distribution.png](../outputs/metrics/figures/split_distribution.png)
- [label_distribution.png](../outputs/metrics/figures/label_distribution.png)
- [duration_kde.png](../outputs/metrics/figures/duration_kde.png)

## 3. Task 2: Narrative Tone Classification
- Candidate models: `LogisticRegression`, `RandomForest`.
- Best model in this run: `LogisticRegression` (validation `macro F1=0.3589`).

Test metrics:
- Accuracy: 0.35
- Macro-F1: 0.3427
- Weighted-F1: 0.3427

With simple handcrafted features + classical models, 8-class emotion classification remains challenging. These numbers are used as a baseline for next iterations.

References:
- [test_metrics.json](../outputs/metrics/test_metrics.json)
- [model_selection.json](../outputs/metrics/model_selection.json)
- [confusion_matrix.png](../outputs/metrics/figures/confusion_matrix.png)

## 4. Task 3: ASR Transcription
- Model: Whisper (`tiny`).
- 200 clips were transcribed and saved.

ASR metrics (current status):
- WER: N/A
- CER: N/A
- Reason: no manual references matched to the current 200 processed clips yet.

A labeling template has been prepared at `data/transcripts/references_template.csv`.

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
- Current classifier still relies on handcrafted features and has limited multi-class discrimination.
- Next steps:
  1) fill `references_template.csv` for a subset and report WER/CER,
  2) add pretrained audio embeddings (wav2vec2 / HuBERT) for better generalization,
  3) run per-class error analysis and improve feature/model choices,
  4) improve retrieval with interpretable weighted scoring.
