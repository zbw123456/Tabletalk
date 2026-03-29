from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch transcribe audio with Whisper.")
    parser.add_argument("--processed_manifest", type=Path, default=Path("data/processed/audio/processed_manifest.csv"))
    parser.add_argument("--output_csv", type=Path, default=Path("data/transcripts/transcripts.csv"))
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    if not args.processed_manifest.exists():
        raise FileNotFoundError(f"Processed manifest not found: {args.processed_manifest}")

    df = pd.read_csv(args.processed_manifest)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        pd.DataFrame(columns=["processed_path", "label", "split", "transcript"]).to_csv(args.output_csv, index=False)
        print("[WARN] Processed manifest is empty, wrote empty transcripts.csv.")
        print(f"[OK] Output: {args.output_csv.resolve()}")
        return

    try:
        import whisper
    except Exception as e:
        raise RuntimeError("Please install 'openai-whisper' first.") from e

    model = whisper.load_model(args.model_size)

    rows = []
    failed = 0
    error_samples: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        audio_path = Path(row["processed_path"])
        if not audio_path.exists():
            failed += 1
            continue
        try:
            result = model.transcribe(str(audio_path), language=args.language, fp16=False)
            rows.append(
                {
                    "processed_path": str(audio_path),
                    "label": row.get("label", "unknown"),
                    "split": row.get("split", "unknown"),
                    "transcript": result.get("text", "").strip(),
                }
            )
        except Exception as e:
            failed += 1
            if len(error_samples) < 5:
                error_samples.append(f"{audio_path.name}: {type(e).__name__}: {e}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)

    print(f"[OK] Transcribed: {len(out_df)}")
    print(f"[OK] Failed: {failed}")
    if error_samples:
        print("[INFO] Example errors:")
        for msg in error_samples:
            print(f"  - {msg}")
    print(f"[OK] Output: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
