from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TableTalk pipeline end-to-end.")
    parser.add_argument("--python", type=str, default="/usr/bin/python3")
    parser.add_argument("--with_synthetic", action="store_true")
    parser.add_argument("--with_asr", action="store_true")
    parser.add_argument("--query", type=str, default="calm narration >4 seconds")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    if args.with_synthetic:
        run([args.python, "src/00_generate_synthetic_data.py"], cwd=root)

    run([args.python, "src/01_prepare_dataset.py", "--raw_dir", "data/raw", "--output_dir", "data/processed"], cwd=root)
    run([args.python, "src/02_preprocess_audio.py", "--manifest", "data/processed/manifest.csv", "--out_dir", "data/processed/audio", "--target_sr", "16000"], cwd=root)
    run([args.python, "src/03_extract_features.py", "--processed_manifest", "data/processed/audio/processed_manifest.csv", "--output", "data/features/features.csv"], cwd=root)
    run([args.python, "src/04_train_classifier.py", "--features", "data/features/features.csv", "--model_dir", "outputs/models", "--metrics_dir", "outputs/metrics"], cwd=root)
    run([args.python, "src/05_evaluate_classifier.py", "--features", "data/features/features.csv", "--model", "outputs/models/tone_classifier.joblib", "--meta", "outputs/models/tone_classifier_meta.json", "--out_dir", "outputs/metrics"], cwd=root)

    if args.with_asr:
        run([args.python, "src/06_transcribe_asr.py", "--processed_manifest", "data/processed/audio/processed_manifest.csv", "--output_csv", "data/transcripts/transcripts.csv", "--model_size", "tiny", "--language", "en"], cwd=root)

    run([args.python, "src/08_retrieval_demo.py", "--features", "data/features/features.csv", "--transcripts", "data/transcripts/transcripts.csv", "--query", args.query, "--top_k", "10", "--out_csv", "outputs/retrieval/results.csv"], cwd=root)
    run([args.python, "src/10_export_report_assets.py"], cwd=root)

    print("[OK] Pipeline finished.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
