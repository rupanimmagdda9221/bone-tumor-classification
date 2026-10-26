"""Central configuration for the bone tumor classification project."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"          # data/raw/benign/*.jpg, data/raw/malignant/*.jpg
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT = OUTPUT_DIR / "best_model.pt"
SPLIT_FILE = OUTPUT_DIR / "splits.json"

CLASSES = ["benign", "malignant"]
IMG_SIZE = 224
MAX_SIDE = 512           # training images were stored as 8-bit grayscale, longest side <= 512 px
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Train / validation / test split (stratified)
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

# Training
BATCH_SIZE = 32
NUM_WORKERS = 2
HEAD_EPOCHS = 5          # stage 1: train only the new classifier head
FINETUNE_EPOCHS = 20     # stage 2: unfreeze layer3 + layer4 and fine-tune
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6             # early stopping on validation F1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
