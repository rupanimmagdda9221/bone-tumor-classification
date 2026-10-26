"""Two-stage ResNet50 transfer-learning training.

Stage 1 trains only the new classification head on top of frozen ImageNet features.
Stage 2 unfreezes layer3/layer4 and fine-tunes with a lower learning rate.
The best checkpoint (by validation F1) is saved and then evaluated on the held-out test set.

    python src/train.py
    python src/train.py --head-epochs 3 --finetune-epochs 10 --batch-size 16
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C
from dataset import get_loaders, make_splits
from evaluate import evaluate_checkpoint
from metrics import compute_metrics, plot_history, predict_loader
from model import build_model, freeze_backbone, unfreeze_top


def class_weights(train_items, device):
    labels = np.array([y for _, y in train_items])
    counts = np.bincount(labels, minlength=len(C.CLASSES)).astype(float)
    weights = counts.sum() / (len(C.CLASSES) * np.maximum(counts, 1))
    print("Train class counts:", dict(zip(C.CLASSES, counts.astype(int).tolist())),
          "-> loss weights:", np.round(weights, 3))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_stage(name, model, loaders, criterion, device, epochs, lr, history, best, patience, checkpoint):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    stale = 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        y_true, y_prob = predict_loader(model, loaders["val"], device)
        va_f1 = compute_metrics(y_true, y_prob)[0]["f1"]
        scheduler.step(va_f1)

        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc", "val_f1"),
                        (tr_loss, tr_acc, va_loss, va_acc, va_f1)):
            history[k].append(v)
        print(f"[{name} {epoch:02d}/{epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")

        if va_f1 > best["f1"] or (va_f1 == best["f1"] and va_loss < best["loss"]):
            best.update(f1=va_f1, loss=va_loss)
            torch.save({"model": model.state_dict(), "classes": C.CLASSES,
                        "img_size": C.IMG_SIZE, "val_f1": va_f1}, checkpoint)
            print(f"  saved best model (val F1 {va_f1:.4f})")
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stopping {name}: no val F1 improvement for {patience} epochs")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--head-epochs", type=int, default=C.HEAD_EPOCHS)
    ap.add_argument("--finetune-epochs", type=int, default=C.FINETUNE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--refresh-splits", action="store_true", help="re-split after changing the dataset")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests only)")
    ap.add_argument("--no-augment", action="store_true", help="disable data augmentation (ablation study)")
    ap.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR,
                    help="where to save the checkpoint, curves and metrics for this run")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    device = C.get_device()
    print("Device:", device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "best_model.pt"

    splits = make_splits(args.data_dir, refresh=args.refresh_splits)
    print({k: len(v) for k, v in splits.items()})
    loaders = get_loaders(splits, args.batch_size, augment=not args.no_augment)
    print("Augmentation:", "off" if args.no_augment else "on")

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"], device))
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}
    best = {"f1": -1.0, "loss": float("inf")}

    freeze_backbone(model)
    train_stage("head", model, loaders, criterion, device, args.head_epochs, C.HEAD_LR,
                history, best, C.PATIENCE, checkpoint)
    unfreeze_top(model)
    train_stage("finetune", model, loaders, criterion, device, args.finetune_epochs, C.FINETUNE_LR,
                history, best, C.PATIENCE, checkpoint)

    (out / "history.json").write_text(json.dumps(history, indent=1))
    plot_history(history, out / "training_curves.png")
    print(f"\nBest validation F1: {best['f1']:.4f}. Evaluating on the test set...\n")
    evaluate_checkpoint(checkpoint, loaders["test"], device, out)


if __name__ == "__main__":
    main()
