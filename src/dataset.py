"""Dataset loading, stratified splitting, preprocessing and augmentation."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (BATCH_SIZE, CLASSES, DATA_DIR, IMAGENET_MEAN, IMAGENET_STD,
                    IMG_SIZE, MAX_SIDE, NUM_WORKERS, ROOT, SEED, SPLIT_FILE, TEST_FRAC, VAL_FRAC)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def train_transform():
    # Augmentation: geometric + intensity changes that remain plausible for radiographs
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def standardize_xray(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Bring any X-ray into the format the training images were stored in (see prepare_btxrd.py):
    8-bit grayscale with the longest side <= max_side. Used for the dataset AND for uploads in the app,
    so a full-resolution or colour JPEG gets exactly the same treatment as the training data."""
    img = ImageOps.exif_transpose(img)    # phone photos; no BTXRD image carries a rotation tag
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        # 16-bit / float images: rescale to 0-255 instead of letting PIL clip them to white
        a = np.asarray(img, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        img = Image.fromarray(((a - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8))
    img = img.convert("L")
    img.thumbnail((max_side, max_side))
    return img


def load_image(path) -> Image.Image:
    # X-rays are often grayscale; replicate to 3 channels for the ImageNet-pretrained ResNet50
    return Image.open(path).convert("RGB")


def scan_dataset(data_dir: Path = DATA_DIR):
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        for p in sorted((data_dir / cls).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(label)
    if not paths:
        raise SystemExit(f"No images found under {data_dir}/{{{','.join(CLASSES)}}}. "
                         "See README.md for how to prepare the dataset.")
    return paths, labels


def _to_rel(path):
    # Store paths relative to the project so the folder can be moved or copied to Colab
    p = Path(path).resolve()
    # as_posix: forward slashes, so a split made on Windows also loads on Mac/Linux/Colab
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def _to_abs(path):
    return path if Path(path).is_absolute() else str(ROOT / path)


def make_splits(data_dir: Path = DATA_DIR, split_file: Path = SPLIT_FILE, refresh=False):
    """Stratified train/val/test split, saved to disk so evaluation uses the same test set."""
    if split_file.exists() and not refresh:
        saved = json.loads(split_file.read_text())
        return {k: [(_to_abs(p), y) for p, y in v] for k, v in saved.items()}
    paths, labels = scan_dataset(data_dir)
    from sklearn.model_selection import train_test_split  # imported here: inference does not need sklearn
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=TEST_FRAC, stratify=labels, random_state=SEED)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=VAL_FRAC / (1 - TEST_FRAC), stratify=y_tmp, random_state=SEED)
    splits = {
        "train": list(zip(x_train, y_train)),
        "val": list(zip(x_val, y_val)),
        "test": list(zip(x_test, y_test)),
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps({k: [(_to_rel(p), y) for p, y in v] for k, v in splits.items()}, indent=1))
    return splits


class XRayDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return self.transform(load_image(path)), label


def get_loaders(splits, batch_size=BATCH_SIZE, augment=True):
    datasets = {
        "train": XRayDataset(splits["train"], train_transform() if augment else eval_transform()),
        "val": XRayDataset(splits["val"], eval_transform()),
        "test": XRayDataset(splits["test"], eval_transform()),
    }
    return {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                         num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
        for name, ds in datasets.items()
    }
