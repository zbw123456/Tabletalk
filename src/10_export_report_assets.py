from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_bar(series: pd.Series, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel("count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm_csv: Path, out_png: Path) -> None:
    if not cm_csv.exists():
        return
    cm_df = pd.read_csv(cm_csv, index_col=0)
    if cm_df.empty:
        return

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_df.values, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm_df.index)), cm_df.index)
    plt.title("Confusion Matrix")

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, str(cm_df.iloc[i, j]), ha="center", va="center")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export charts for the technical report.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/features.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics/test_metrics.json"))
    parser.add_argument("--cm_csv", type=Path, default=Path("outputs/metrics/test_confusion_matrix.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/metrics/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest.exists():
        manifest = pd.read_csv(args.manifest)
        if not manifest.empty:
            if "split" in manifest.columns:
                plot_bar(manifest["split"].value_counts(), "Split Distribution", args.out_dir / "split_distribution.png")
            if "label" in manifest.columns:
                plot_bar(manifest["label"].value_counts(), "Label Distribution", args.out_dir / "label_distribution.png")

    if args.features.exists():
        features = pd.read_csv(args.features)
        if not features.empty and "label" in features.columns and "duration_sec" in features.columns:
            plt.figure(figsize=(6, 4))
            for label, sub in features.groupby("label"):
                sub["duration_sec"].plot(kind="kde", label=str(label))
            plt.legend()
            plt.title("Duration KDE by Label")
            plt.xlabel("duration_sec")
            plt.tight_layout()
            plt.savefig(args.out_dir / "duration_kde.png", dpi=160)
            plt.close()

    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        metrics_md = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        (args.out_dir / "metrics_summary.md").write_text(metrics_md + "\n", encoding="utf-8")

    plot_confusion_matrix(args.cm_csv, args.out_dir / "confusion_matrix.png")
    print(f"[OK] Report assets exported to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
