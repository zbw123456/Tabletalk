from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train narrative tone classifiers.")
    parser.add_argument("--features", type=Path, default=Path("data/features/features.csv"))
    parser.add_argument("--model_dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--metrics_dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Features file not found: {args.features}")

    df = pd.read_csv(args.features)
    if df.empty:
        print("[WARN] Features file is empty, skip training.")
        return

    required = {"label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Features file must include columns: {required}")

    feature_cols = [
        c
        for c in df.columns
        if c not in {"processed_path", "label", "split"} and np.issubdtype(df[c].dtype, np.number)
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        print("[WARN] Missing train/val split rows, skip training.")
        return

    X_train = train_df[feature_cols].fillna(0.0).to_numpy()
    y_train = train_df["label"].astype(str).to_numpy()
    X_val = val_df[feature_cols].fillna(0.0).to_numpy()
    y_val = val_df["label"].astype(str).to_numpy()

    models = build_models(seed=args.seed)

    results = []
    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        results.append({"model": name, "val_macro_f1": float(f1)})
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    model_path = args.model_dir / "tone_classifier.joblib"
    meta_path = args.model_dir / "tone_classifier_meta.json"
    metrics_path = args.metrics_dir / "model_selection.json"

    joblib.dump(best_model, model_path)

    meta = {
        "best_model": best_name,
        "val_macro_f1": best_f1,
        "feature_cols": feature_cols,
        "labels": sorted(df["label"].astype(str).unique().tolist()),
        "seed": args.seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Candidate metrics saved: {metrics_path.resolve()}")
    print(f"[OK] Best model: {best_name} (macro F1={best_f1:.4f})")
    print(f"[OK] Model saved: {model_path.resolve()}")
    print(f"[OK] Meta saved: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
