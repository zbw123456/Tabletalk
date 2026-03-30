# TableTalk Technical Test - Starter Pipeline

The English version is provided below.

本仓库按测试任务拆成可逐步执行的脚本。

## 1. 安装依赖

```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

## 2. 准备数据

把数据放到 `data/raw/`，建议结构：

- `data/raw/<label_name>/*.wav`（推荐）
- 或直接放 RAVDESS 文件名格式音频（脚本会自动解析情绪标签）

建议按测试要求先使用子集进行实验：`50–200` 条音频即可。

如需先联调流程（无真实数据时），可先生成示例数据：

```bash
/usr/bin/python3 src/00_generate_synthetic_data.py
```

## 3. 第一步：生成清单与切分

```bash
python src/01_prepare_dataset.py \
  --raw_dir data/raw \
  --output_dir data/processed \
  --min_duration 0.8 \
  --max_duration 30 \
  --test_size 0.2 \
  --val_size 0.1
```

输出：`data/processed/manifest.csv`

## 4. 第二步：音频预处理

```bash
python src/02_preprocess_audio.py \
  --manifest data/processed/manifest.csv \
  --out_dir data/processed/audio \
  --target_sr 16000 \
  --trim_silence
```

输出：`data/processed/audio/processed_manifest.csv`

## 5. 第三步：特征提取

```bash
python src/03_extract_features.py \
  --processed_manifest data/processed/audio/processed_manifest.csv \
  --output data/features/features.csv
```

输出：`data/features/features.csv`

---

当前版本已包含：
- 分类训练与评估（Task 2）
- ASR 转写与评估（Task 3）
- 检索原型（Task 4）

## 6. 第四步：训练分类模型

```bash
/usr/bin/python3 src/04_train_classifier.py \
  --features data/features/features.csv \
  --model_dir outputs/models \
  --metrics_dir outputs/metrics
```

## 7. 第五步：评估分类模型

```bash
/usr/bin/python3 src/05_evaluate_classifier.py \
  --features data/features/features.csv \
  --model outputs/models/tone_classifier.joblib \
  --meta outputs/models/tone_classifier_meta.json \
  --out_dir outputs/metrics
```

## 8. 第六步：ASR 转写（Whisper）

先确保系统已安装 `ffmpeg`（macOS 可用 `brew install ffmpeg`）。

```bash
/usr/bin/python3 src/06_transcribe_asr.py \
  --processed_manifest data/processed/audio/processed_manifest.csv \
  --output_csv data/transcripts/transcripts.csv \
  --model_size base \
  --language en
```

## 9. 第七步：ASR 评估

先准备 `data/transcripts/references.csv`，至少包含：
- `processed_path`
- `reference_text`

```bash
/usr/bin/python3 src/07_eval_asr.py \
  --transcripts data/transcripts/transcripts.csv \
  --references data/transcripts/references.csv \
  --out_json outputs/metrics/asr_metrics.json
```

## 10. 第八步：检索原型

```bash
/usr/bin/python3 src/08_retrieval_demo.py \
  --features data/features/features.csv \
  --transcripts data/transcripts/transcripts.csv \
  --query "calm narration >4 seconds" \
  --top_k 10 \
  --out_csv outputs/retrieval/results.csv
```

## 11. 导出报告图表素材

```bash
/usr/bin/python3 src/10_export_report_assets.py
```

输出目录：`outputs/metrics/figures/`

## 12. 一键运行（可选）

```bash
/usr/bin/python3 src/run_all.py --python /usr/bin/python3 --with_asr
```

如果你希望先自动生成示例数据再跑全流程：

```bash
/usr/bin/python3 src/run_all.py --python /usr/bin/python3 --with_synthetic --with_asr
```

也可以使用脚本：

```bash
./scripts/run_pipeline.sh /usr/bin/python3
```

## 13. 提交材料模板

- 中文模板：`report/technical_report_template.md`
- 英文模板：`report/technical_report_template_en.md`
- 中文草稿：`report/technical_report_draft.md`
- 英文草稿：`report/technical_report_draft_en.md`

---

# English Version

This repository is organized into runnable scripts mapped to each task in the technical test.

## 1. Install dependencies

```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

## 2. Prepare data

Put audio files under `data/raw/`.

Recommended structure:

- `data/raw/<label_name>/*.wav` (recommended)
- or raw RAVDESS-style filenames (label can be inferred automatically)

As suggested in the test, using a subset of `50–200` recordings is enough for this assignment.

If you want to run a quick local dry-run without real data, generate synthetic samples first:

```bash
/usr/bin/python3 src/00_generate_synthetic_data.py
```

## 3. Step 1: Build manifest and split

```bash
python src/01_prepare_dataset.py \
  --raw_dir data/raw \
  --output_dir data/processed \
  --min_duration 0.8 \
  --max_duration 30 \
  --test_size 0.2 \
  --val_size 0.1
```

Output: `data/processed/manifest.csv`

## 4. Step 2: Audio preprocessing

```bash
python src/02_preprocess_audio.py \
  --manifest data/processed/manifest.csv \
  --out_dir data/processed/audio \
  --target_sr 16000 \
  --trim_silence
```

Output: `data/processed/audio/processed_manifest.csv`

## 5. Step 3: Feature extraction

```bash
python src/03_extract_features.py \
  --processed_manifest data/processed/audio/processed_manifest.csv \
  --output data/features/features.csv
```

Output: `data/features/features.csv`

---

Current version already includes:
- Classification training and evaluation (Task 2)
- ASR transcription and evaluation (Task 3)
- Retrieval prototype (Task 4)

## 6. Step 4: Train classifier

```bash
/usr/bin/python3 src/04_train_classifier.py \
  --features data/features/features.csv \
  --model_dir outputs/models \
  --metrics_dir outputs/metrics
```

## 7. Step 5: Evaluate classifier

```bash
/usr/bin/python3 src/05_evaluate_classifier.py \
  --features data/features/features.csv \
  --model outputs/models/tone_classifier.joblib \
  --meta outputs/models/tone_classifier_meta.json \
  --out_dir outputs/metrics
```

## 8. Step 6: ASR transcription (Whisper)

Make sure `ffmpeg` is installed first (on macOS: `brew install ffmpeg`).

```bash
/usr/bin/python3 src/06_transcribe_asr.py \
  --processed_manifest data/processed/audio/processed_manifest.csv \
  --output_csv data/transcripts/transcripts.csv \
  --model_size base \
  --language en
```

## 9. Step 7: ASR evaluation

Prepare `data/transcripts/references.csv` with at least:
- `processed_path`
- `reference_text`

```bash
/usr/bin/python3 src/07_eval_asr.py \
  --transcripts data/transcripts/transcripts.csv \
  --references data/transcripts/references.csv \
  --out_json outputs/metrics/asr_metrics.json
```

## 10. Step 8: Retrieval prototype

```bash
/usr/bin/python3 src/08_retrieval_demo.py \
  --features data/features/features.csv \
  --transcripts data/transcripts/transcripts.csv \
  --query "calm narration >4 seconds" \
  --top_k 10 \
  --out_csv outputs/retrieval/results.csv
```

## 11. Export report figures

```bash
/usr/bin/python3 src/10_export_report_assets.py
```

Output folder: `outputs/metrics/figures/`

## 12. One-command run (optional)

```bash
/usr/bin/python3 src/run_all.py --python /usr/bin/python3 --with_asr
```

If you want synthetic data generation + full run:

```bash
/usr/bin/python3 src/run_all.py --python /usr/bin/python3 --with_synthetic --with_asr
```

Or use the shell script:

```bash
./scripts/run_pipeline.sh /usr/bin/python3
```

## 13. Submission templates

- Chinese template: `report/technical_report_template.md`
- English template: `report/technical_report_template_en.md`
- Chinese draft: `report/technical_report_draft.md`
- English draft: `report/technical_report_draft_en.md`
