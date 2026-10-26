"""Organise the public BTXRD dataset (Bone Tumor X-ray Radiograph Dataset, Yao et al., Scientific Data 2025)
into data/raw/benign and data/raw/malignant.

Download BTXRD from figshare (search "BTXRD bone tumor X-ray"), unzip it, then run:

    python src/prepare_btxrd.py --src /path/to/BTXRD

The source folder must contain an `images/` folder and `dataset.xlsx` whose rows have an image id
column and 0/1 `benign` / `malignant` columns. Normal (no-tumor) images are skipped because this project
classifies benign vs malignant tumors.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from config import DATA_DIR, MAX_SIDE
from dataset import standardize_xray


def find_col(df, *names):
    cols = {c.strip().lower(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    raise SystemExit(f"Could not find any of {names} in columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, default=DATA_DIR)
    ap.add_argument("--max-side", type=int, default=MAX_SIDE,
                    help="downscale so the longest side is at most this (speeds up training); 0 = copy as-is")
    args = ap.parse_args()

    sheet = next(args.src.rglob("dataset.xlsx"), None) or next(args.src.rglob("*.xlsx"), None)
    if sheet is None:
        raise SystemExit(f"No .xlsx metadata found under {args.src}")
    images_dir = next((p for p in args.src.rglob("images") if p.is_dir()), args.src)
    index = {p.name: p for p in images_dir.rglob("*") if p.is_file()}
    by_stem = {p.stem: p for p in index.values()}

    df = pd.read_excel(sheet)
    id_col = find_col(df, "image_id", "image", "filename", "image_name")
    ben_col = find_col(df, "benign", "benign_tumor")
    mal_col = find_col(df, "malignant", "malignant_tumor")

    counts = {"benign": 0, "malignant": 0, "skipped": 0, "missing": 0}
    for cls in ("benign", "malignant"):
        (args.dst / cls).mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        name = str(row[id_col]).strip()
        cls = "malignant" if row[mal_col] == 1 else "benign" if row[ben_col] == 1 else None
        if cls is None:
            counts["skipped"] += 1
            continue
        src = index.get(name) or by_stem.get(Path(name).stem)
        if src is None:
            counts["missing"] += 1
            continue
        if args.max_side:
            standardize_xray(Image.open(src), args.max_side).save(args.dst / cls / f"{src.stem}.png")
        else:
            shutil.copy2(src, args.dst / cls / src.name)
        counts[cls] += 1
    print(f"Metadata: {sheet}\nImages:   {images_dir}\nResult:   {counts}")


if __name__ == "__main__":
    main()
