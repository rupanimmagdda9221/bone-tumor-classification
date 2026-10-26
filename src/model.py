"""ResNet50 transfer-learning classifier."""
import os

import certifi
import torch
import torch.nn as nn
from torchvision import models

from config import CLASSES


def build_model(pretrained=True, num_classes=len(CLASSES)) -> nn.Module:
    # python.org builds on macOS ship without CA certs, which breaks the weight download
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Stage 1: only the new classification head is trainable."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")


def unfreeze_top(model: nn.Module, blocks=("layer3", "layer4", "fc")):
    """Stage 2: fine-tune the deepest residual blocks together with the head."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(blocks)


def load_checkpoint(path, device) -> nn.Module:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trained model not found: {path}. Copy outputs/best_model.pt into the "
                                "project or train one with src/train.py.")
    # mmap + meta-device model: weights are read straight from the file instead of being held twice in RAM
    state = torch.load(path, map_location="cpu", mmap=True)
    if state.get("classes") != CLASSES:
        raise ValueError(f"{path} was trained for classes {state.get('classes')}, expected {CLASSES}.")
    with torch.device("meta"):
        model = build_model(pretrained=False)
    model.load_state_dict(state["model"], assign=True)   # strict: fails if the file is not this ResNet50 architecture
    return model.to(device).eval()
