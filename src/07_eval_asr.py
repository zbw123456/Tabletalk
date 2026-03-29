from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR quality using reference transcripts.")
    parser.add_argument("--transcripts", type=Path, default=Path("data/transcripts/transcripts.csv"))
    parser.add_argument("--references", type=Path, default=Path("data/transcripts/references.csv"))
    parser.add_argument("--out_json", type=Path, default=Path("outputs/metrics/asr_metrics.json"))
    args = parser.parse_args()

    if not args.transcripts.exists():
        raise FileNotFoundError(f"Transcripts not found: {args.transcripts}")
    if not args.references.exists():
        raise FileNotFoundError(
            f"References not found: {args.references}. Expected columns: processed_path, reference_text"
        )

    pred_df = pd.read_csv(args.transcripts)
    ref_df = pd.read_csv(args.references)

    if pred_df.empty or ref_df.empty:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"wer": None, "cer": None, "pairs": 0}, indent=2), encoding="utf-8")
        print("[WARN] Empty transcript/reference input.")
        return

    merged = pred_df.merge(ref_df, on="processed_path", how="inner")
    if merged.empty:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"wer": None, "cer": None, "pairs": 0}, indent=2), encoding="utf-8")
        print("[WARN] No matched rows between predictions and references.")
        return

    try:
        from jiwer import cer, wer
    except Exception as e:
        raise RuntimeError("Please install 'jiwer' first.") from e

    truths = merged["reference_text"].fillna("").astype(str).tolist()
    preds = merged["transcript"].fillna("").astype(str).tolist()

    metrics = {
        "pairs": len(merged),
        "wer": float(wer(truths, preds)),
        "cer": float(cer(truths, preds)),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] ASR metrics: {metrics}")
    print(f"[OK] Saved: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
