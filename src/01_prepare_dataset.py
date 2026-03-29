from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.audio_io import get_audio_metadata

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def infer_label(path: Path, raw_dir: Path) -> str:
    """Infer label from folder name, then try RAVDESS filename pattern."""
    try:
        rel = path.relative_to(raw_dir)
        if len(rel.parts) >= 2:
            parent_label = rel.parts[0].strip()
            if parent_label and parent_label.lower() not in {"audio", "wav", "clips"}:
                return parent_label.lower()
    except Exception:
        pass

    match = re.match(r"^\d{2}-\d{2}-(\d{2})-\d{2}-\d{2}-\d{2}-\d{2}\.[a-zA-Z0-9]+$", path.name)
    if match:
        return RAVDESS_EMOTION_MAP.get(match.group(1), "unknown")

    return "unknown"


def safe_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    label_counts = df["label"].value_counts()
    can_stratify = (label_counts.min() >= 2) and (df["label"].nunique() > 1)
    stratify = df["label"] if can_stratify else None

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    # val_size is relative to full set; convert to train_val fraction
    adjusted_val = val_size / (1 - test_size)

    label_counts_tv = train_val["label"].value_counts() if not train_val.empty else pd.Series(dtype=int)
    can_stratify_tv = (not label_counts_tv.empty) and (label_counts_tv.min() >= 2) and (train_val["label"].nunique() > 1)
    stratify_tv = train_val["label"] if can_stratify_tv else None

    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        random_state=seed,
        stratify=stratify_tv,
    )
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset manifest and train/val/test split.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min_duration", type=float, default=0.8)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio_files: list[Path] = []
    for ext in args.extensions:
        audio_files.extend(args.raw_dir.rglob(f"*{ext}"))

    if not audio_files:
        print(f"[WARN] No audio files found under: {args.raw_dir.resolve()}")
        empty_cols = [
            "path",
            "label",
            "samplerate",
            "channels",
            "frames",
            "duration_sec",
            "format",
            "subtype",
            "split",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(args.output_dir / "manifest.csv", index=False)
        print("[OK] Wrote empty manifest.csv")
        return

    rows: list[dict] = []
    unreadable = 0

    for p in tqdm(audio_files, desc="Reading metadata"):
        meta = get_audio_metadata(p)
        if meta is None:
            unreadable += 1
            continue
        meta["label"] = infer_label(p, args.raw_dir)
        rows.append(meta)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] All audio files were unreadable.")
        return

    duration_mask = df["duration_sec"].between(args.min_duration, args.max_duration)
    kept = df[duration_mask].copy().reset_index(drop=True)

    if kept.empty:
        print("[WARN] No samples remain after duration filtering.")
        kept["split"] = []
    else:
        train, val, test = safe_split(
            kept,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )
        split_map = {
            **{p: "train" for p in train["path"]},
            **{p: "val" for p in val["path"]},
            **{p: "test" for p in test["path"]},
        }
        kept["split"] = kept["path"].map(split_map)

    output_path = args.output_dir / "manifest.csv"
    kept = kept[
        [
            "path",
            "label",
            "samplerate",
            "channels",
            "frames",
            "duration_sec",
            "format",
            "subtype",
            "split",
        ]
    ].sort_values(by=["split", "label", "path"], na_position="last")

    kept.to_csv(output_path, index=False)

    print(f"[OK] Total files found: {len(audio_files)}")
    print(f"[OK] Unreadable files: {unreadable}")
    print(f"[OK] Kept after duration filter: {len(kept)}")

    if len(kept):
        split_counts = kept["split"].value_counts(dropna=False).to_dict()
        label_counts = kept["label"].value_counts().to_dict()
        print(f"[OK] Split counts: {split_counts}")
        print(f"[OK] Label counts: {label_counts}")

    print(f"[OK] Manifest saved: {output_path.resolve()}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
