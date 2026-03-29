from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_features(audio_path: Path, n_mfcc: int = 13) -> dict:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    duration_sec = librosa.get_duration(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    feats = {
        "duration_sec": float(duration_sec),
        "pitch_mean": float(np.mean(f0_valid)) if len(f0_valid) else 0.0,
        "pitch_std": float(np.std(f0_valid)) if len(f0_valid) else 0.0,
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
    }

    for i, (m, s) in enumerate(zip(mfcc_mean, mfcc_std), start=1):
        feats[f"mfcc_{i}_mean"] = float(m)
        feats[f"mfcc_{i}_std"] = float(s)

    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features from processed manifest.")
    parser.add_argument("--processed_manifest", type=Path, default=Path("data/processed/audio/processed_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/features/features.csv"))
    parser.add_argument("--n_mfcc", type=int, default=13)
    args = parser.parse_args()

    if not args.processed_manifest.exists():
        raise FileNotFoundError(f"Processed manifest not found: {args.processed_manifest}")

    df = pd.read_csv(args.processed_manifest)
    if df.empty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["processed_path", "label", "split", "duration_sec"]
        ).to_csv(args.output, index=False)
        print("[WARN] Processed manifest is empty, wrote empty features.csv.")
        print(f"[OK] Feature file: {args.output.resolve()}")
        return

    rows = []
    failures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        p = Path(row["processed_path"])
        if not p.exists():
            failures += 1
            continue

        try:
            feats = extract_features(p, n_mfcc=args.n_mfcc)
            feats.update(
                {
                    "processed_path": str(p),
                    "label": row.get("label", "unknown"),
                    "split": row.get("split", "unknown"),
                }
            )
            rows.append(feats)
        except Exception:
            failures += 1

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"[OK] Features extracted: {len(out_df)}")
    print(f"[OK] Failed files: {failures}")
    print(f"[OK] Feature file: {args.output.resolve()}")


if __name__ == "__main__":
    main()
