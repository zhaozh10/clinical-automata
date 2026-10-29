"""
data.py — Chest drain binary classification on NIH ChestX-ray14.
Loads preprocessed 512x512 PNGs, applies augmentation, returns dataloaders.
"""

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as T

from prepare import DATA_DIR, EXT_DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLIT_CSV = os.path.join(
    os.path.dirname(DATA_DIR.rstrip("/")),  # NIH_ChestX-ray14/
    "ChestDrain_split.csv"
)
IMAGE_ROOT = os.path.join(DATA_DIR, "preprocessed")

# Build a lookup: filename -> full path (images are spread across images_001..012)
_IMAGE_PATH_CACHE = None

def _build_image_path_cache():
    global _IMAGE_PATH_CACHE
    if _IMAGE_PATH_CACHE is not None:
        return _IMAGE_PATH_CACHE
    _IMAGE_PATH_CACHE = {}
    for subdir in sorted(glob.glob(os.path.join(IMAGE_ROOT, "images_*", "images"))):
        for fname in os.listdir(subdir):
            _IMAGE_PATH_CACHE[fname] = os.path.join(subdir, fname)
    return _IMAGE_PATH_CACHE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ChestDrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_cache = _build_image_path_cache()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["Image Index"]
        label = int(row["Drain"])

        img_path = self.path_cache[fname]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, fname


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMG_SIZE = 512

def get_train_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size=32, num_workers=4):
    df = pd.read_csv(SPLIT_CSV)

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    train_ds = ChestDrainDataset(df_train, transform=get_train_transform())
    val_ds = ChestDrainDataset(df_val, transform=get_val_transform())
    test_ds = ChestDrainDataset(df_test, transform=get_val_transform())

    # Weighted sampler to oversample positive class for higher recall
    labels = df_train["Drain"].values
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
