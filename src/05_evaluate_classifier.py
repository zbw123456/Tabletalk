from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained classifier on test split.")
    parser.add_argument("--features", type=Path, default=Path("data/features/features.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/tone_classifier.joblib"))
    parser.add_argument("--meta", type=Path, default=Path("outputs/models/tone_classifier_meta.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Features file not found: {args.features}")
    if not args.model.exists() or not args.meta.exists():
        raise FileNotFoundError("Model or metadata file not found. Run training first.")

    df = pd.read_csv(args.features)
    if df.empty:
        print("[WARN] Features file is empty, skip evaluation.")
        return

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        print("[WARN] Test split is empty, skip evaluation.")
        return

    X_test = test_df[feature_cols].fillna(0.0).to_numpy()
    y_test = test_df["label"].astype(str).to_numpy()

    model = joblib.load(args.model)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
    }

    labels = sorted(np.unique(np.concatenate([y_test, pred])))
    cm = confusion_matrix(y_test, pred, labels=labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "test_metrics.json"
    report_path = args.out_dir / "test_classification_report.json"
    cm_path = args.out_dir / "test_confusion_matrix.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(cm_path, index=True)

    print(f"[OK] Test metrics: {metrics}")
    print(f"[OK] Saved: {metrics_path.resolve()}")
    print(f"[OK] Saved: {report_path.resolve()}")
    print(f"[OK] Saved: {cm_path.resolve()}")


if __name__ == "__main__":
    main()
