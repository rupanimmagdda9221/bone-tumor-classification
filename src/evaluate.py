"""Evaluate a saved model on the held-out test split: metrics, confusion matrix, ROC curve and
an error-analysis gallery of misclassified X-rays with Grad-CAM.

    python src/evaluate.py
    python src/evaluate.py --output-dir outputs/no_augmentation
"""
import argparse
import json
from pathlib import Path

import config as C
from dataset import get_loaders, make_splits
from metrics import (compute_metrics, confidence_bands, plot_confusion_matrix, plot_error_gallery,
                     plot_roc, predict_loader, report)
from model import load_checkpoint


def evaluate_checkpoint(checkpoint, test_loader, device, out_dir=C.OUTPUT_DIR):
    out_dir = Path(out_dir)
    model = load_checkpoint(checkpoint, device)
    y_true, y_prob = predict_loader(model, test_loader, device)
    metrics, y_pred = compute_metrics(y_true, y_prob)
    cm = plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_roc(y_true, y_prob, out_dir / "roc_curve.png")

    paths = [p for p, _ in test_loader.dataset.items]
    errors = [(paths[i], int(y_true[i]), float(y_prob[i])) for i in range(len(paths)) if y_pred[i] != y_true[i]]
    if errors:
        plot_error_gallery(errors, checkpoint, out_dir / "misclassified.png")

    print("Test set results")
    for k, v in metrics.items():
        print(f"  {k:<9} {v:.4f}")
    print("\n" + report(y_true, y_pred))
    print("Confusion matrix (rows = actual, cols = predicted):\n", cm)

    out = {**metrics, "confusion_matrix": cm.tolist(), "classes": C.CLASSES, "n_test": int(len(y_true)),
           "confidence_bands": confidence_bands(y_true, y_prob),
           "misclassified": [{"image": Path(p).name, "actual": C.CLASSES[t], "p_malignant": round(pm, 4)}
                             for p, t, pm in errors]}
    (out_dir / "test_metrics.json").write_text(json.dumps(out, indent=1))
    print(f"\nSaved metrics and plots to {out_dir}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR,
                    help="run folder containing best_model.pt; results are written there too")
    args = ap.parse_args()
    if not C.SPLIT_FILE.exists():
        raise SystemExit("No splits.json found - run src/train.py first.")
    splits = make_splits()
    evaluate_checkpoint(args.output_dir / "best_model.pt", get_loaders(splits)["test"], C.get_device(),
                        args.output_dir)


if __name__ == "__main__":
    main()
