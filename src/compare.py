"""Compare test-set results of different training runs (e.g. with vs without data augmentation).

    python src/compare.py
Writes outputs/comparison.md and outputs/comparison.png.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

RUNS = {
    "ResNet50 + augmentation (proposed)": C.OUTPUT_DIR,
    "ResNet50, no augmentation": C.OUTPUT_DIR / "no_augmentation",
}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
LABELS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]


def main():
    results = {}
    for name, d in RUNS.items():
        f = d / "test_metrics.json"
        if f.exists():
            results[name] = json.loads(f.read_text())
        else:
            print(f"skipping {name}: {f} not found")
    if not results:
        raise SystemExit("No runs with test_metrics.json found.")

    lines = ["| Model | " + " | ".join(LABELS) + " | Missed malignant | False alarms |",
             "|---|" + "---|" * (len(LABELS) + 2)]
    for name, r in results.items():
        cm = r["confusion_matrix"]
        lines.append(f"| {name} | " + " | ".join(f"{r[m]:.3f}" for m in METRICS)
                     + f" | {cm[1][0]} / {sum(cm[1])} | {cm[0][1]} / {sum(cm[0])} |")
    table = "\n".join(lines)
    note = ("\n\nPrecision, recall and F1 are for the malignant class. All runs use the same "
            f"stratified test set ({next(iter(results.values()))['n_test']} images).\n")
    (C.OUTPUT_DIR / "comparison.md").write_text("# Model comparison (test set)\n\n" + table + note)
    print(table)

    x = np.arange(len(METRICS))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, r) in enumerate(results.items()):
        vals = [r[m] for m in METRICS]
        bars = ax.bar(x + (i - (len(results) - 1) / 2) * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Test-set performance: effect of data augmentation")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.OUTPUT_DIR / "comparison.png", dpi=150)
    print(f"\nSaved {C.OUTPUT_DIR / 'comparison.md'} and comparison.png")


if __name__ == "__main__":
    main()
