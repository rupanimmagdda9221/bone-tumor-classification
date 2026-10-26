"""Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix and plots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from config import CLASSES


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        y_prob.extend(probs.cpu().tolist())
        y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_prob)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    m["roc_auc"] = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    return m, y_pred


CONFIDENCE_BANDS = [(0.5, 0.7, "low"), (0.7, 0.9, "moderate"), (0.9, 1.01, "high")]


def confidence_bands(y_true, y_prob):
    """How often the model was right on the test set, grouped by the confidence of its own prediction."""
    conf = np.maximum(y_prob, 1 - y_prob)
    correct = (y_prob >= 0.5).astype(int) == y_true
    bands = []
    for lo, hi, name in CONFIDENCE_BANDS:
        m = (conf >= lo) & (conf < hi)
        bands.append({"band": name, "min": lo, "max": min(hi, 1.0), "n": int(m.sum()),
                      "accuracy": float(correct[m].mean()) if m.any() else None})
    return bands


def report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"ResNet50 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC Curve - malignant vs benign (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_gallery(errors, checkpoint, path, max_items=12):
    """Misclassified test X-rays next to their Grad-CAM heatmaps, for error analysis."""
    from PIL import Image

    from predict import Predictor

    predictor = Predictor(checkpoint)
    errors = sorted(errors, key=lambda e: e[1], reverse=True)[:max_items]   # missed malignant first
    fig, axes = plt.subplots(len(errors), 2, figsize=(6, 3 * len(errors)), squeeze=False)
    for row, (img_path, actual, p_mal) in zip(axes, errors):
        img = Image.open(img_path).convert("RGB")
        _, pred, heatmap = predictor(img)
        row[0].imshow(img.resize((224, 224)))
        row[0].set_title(f"{Path(img_path).name}\nactual: {CLASSES[actual]}", fontsize=9)
        row[1].imshow(heatmap)
        row[1].set_title(f"predicted: {pred}\nP(malignant) = {p_mal:.2f}", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.suptitle("Misclassified test images (left) and Grad-CAM (right)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)
