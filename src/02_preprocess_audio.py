from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def normalize_peak(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_val = np.max(np.abs(y))
    if max_val <= 0:
        return y
    return y * (peak / max_val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio from manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed/audio"))
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--mono", action="store_true", default=True)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--top_db", type=float, default=25.0)
    parser.add_argument("--normalize_peak", type=float, default=0.95)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    out_manifest = args.out_dir / "processed_manifest.csv"
    if df.empty:
        pd.DataFrame(
            columns=[
                "src_path",
                "processed_path",
                "label",
                "split",
                "samplerate",
                "duration_sec",
            ]
        ).to_csv(out_manifest, index=False)
        print("[WARN] Manifest is empty, wrote empty processed_manifest.csv.")
        print(f"[OK] Output manifest: {out_manifest.resolve()}")
        return

    out_rows = []
    failures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        src = Path(row["path"])
        if not src.exists():
            failures += 1
            continue

        try:
            y, sr = librosa.load(str(src), sr=args.target_sr, mono=args.mono)
            if args.trim_silence:
                y, _ = librosa.effects.trim(y, top_db=args.top_db)

            y = normalize_peak(y, peak=args.normalize_peak)

            split = row.get("split", "unknown")
            label = row.get("label", "unknown")
            dst_dir = args.out_dir / str(split) / str(label)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{src.stem}.wav"

            sf.write(str(dst), y, args.target_sr)

            out_rows.append(
                {
                    "src_path": str(src),
                    "processed_path": str(dst),
                    "label": label,
                    "split": split,
                    "samplerate": args.target_sr,
                    "duration_sec": len(y) / float(args.target_sr) if args.target_sr else 0.0,
                }
            )
        except Exception:
            failures += 1

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_manifest, index=False)

    print(f"[OK] Processed files: {len(out_df)}")
    print(f"[OK] Failed files: {failures}")
    print(f"[OK] Output manifest: {out_manifest.resolve()}")


if __name__ == "__main__":
    main()
