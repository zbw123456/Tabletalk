from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def retrieve(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.lower().strip()
    out = df.copy()

    if "calm" in q:
        out = out[out["label"].astype(str).str.contains("calm", case=False, na=False)]
    if "dramatic" in q:
        out = out[out["label"].astype(str).str.contains("dramatic|angry|surprised", case=False, na=False)]
    if "dialogue" in q:
        out = out[out["transcript"].astype(str).str.len() > 0]
    if "high-energy" in q or "high energy" in q:
        label_hit = out["label"].astype(str).str.contains(
            "urgency|angry|fear|fearful|surpris|dramatic|excited|high",
            case=False,
            na=False,
        )
        if label_hit.any():
            out = out[label_hit]

        score_cols = [c for c in ["energy_mean", "pitch_mean", "spectral_centroid_mean"] if c in out.columns]
        if score_cols:
            score = np.zeros(len(out), dtype=float)
            for c in score_cols:
                s = out[c].astype(float)
                std = float(s.std())
                if std > 1e-8:
                    score += ((s - float(s.mean())) / std).to_numpy()
            out = out.assign(_energy_score=score)
            threshold = out["_energy_score"].quantile(0.75)
            out = out[out["_energy_score"] >= threshold]
            out = out.drop(columns=["_energy_score"])

    # parse pattern like ">4 seconds" or "> 4 sec"
    import re

    m = re.search(r">\s*(\d+(?:\.\d+)?)\s*(seconds|sec|s)", q)
    if m and "duration_sec" in out.columns:
        v = float(m.group(1))
        out = out[out["duration_sec"] > v]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple narrative audio retrieval prototype.")
    parser.add_argument("--features", type=Path, default=Path("data/features/features.csv"))
    parser.add_argument("--transcripts", type=Path, default=Path("data/transcripts/transcripts.csv"))
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--out_csv", type=Path, default=Path("outputs/retrieval/results.csv"))
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Features not found: {args.features}")

    feat_df = pd.read_csv(args.features)
    tr_df = pd.read_csv(args.transcripts) if args.transcripts.exists() else pd.DataFrame(columns=["processed_path", "transcript"])

    if feat_df.empty:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.out_csv, index=False)
        print("[WARN] Feature file is empty.")
        return

    merged = feat_df.merge(tr_df[["processed_path", "transcript"]], on="processed_path", how="left")
    merged["transcript"] = merged["transcript"].fillna("")

    hits = retrieve(merged, args.query).sort_values(by="duration_sec", ascending=False).head(args.top_k)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(args.out_csv, index=False)

    print(f"[OK] Query: {args.query}")
    print(f"[OK] Hits: {len(hits)}")
    print(f"[OK] Saved: {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
