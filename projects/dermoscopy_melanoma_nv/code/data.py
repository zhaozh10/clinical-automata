"""
data.py — Binary MEL vs NV classification from ISIC 2019.
Patient-level (lesion_id) stratified split.
All images preloaded into RAM for fast training (~3GB for 21K images at 224px).
"""

import os
import csv
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from prepare import DATA_DIR, EXT_DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 384
NUM_WORKERS = 0  # images in RAM, single GPU — no workers needed
CLASSES = ["NV", "MEL"]  # 0 = NV (negative), 1 = MEL (positive)
VAL_FRACTION = 0.20
SEED = 42

# ---------------------------------------------------------------------------
# Parse labels — only keep MEL and NV
# ---------------------------------------------------------------------------

def load_labels(csv_path):
    """Returns dict: image_id -> label (0=NV, 1=MEL)."""
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["MEL"]) == 1.0:
                labels[row["image"]] = 1
            elif float(row["NV"]) == 1.0:
                labels[row["image"]] = 0
    return labels


def load_metadata(csv_path):
    """Returns dict: image_id -> lesion_id (empty string if missing)."""
    meta = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["image"]] = row.get("lesion_id", "").strip()
    return meta


# ---------------------------------------------------------------------------
# Patient-level stratified split
# ---------------------------------------------------------------------------

def patient_stratified_split(labels, metadata, val_frac=VAL_FRACTION, seed=SEED):
    rng = random.Random(seed)
    lesion_to_images = defaultdict(list)
    unique_counter = 0
    for img_id, label in labels.items():
        lid = metadata.get(img_id, "")
        if not lid:
            lid = f"__unique_{unique_counter}"
            unique_counter += 1
        lesion_to_images[lid].append((img_id, label))

    lesion_by_class = defaultdict(list)
    for lid, items in lesion_to_images.items():
        majority_label = max(set(l for _, l in items), key=lambda l: sum(1 for _, ll in items if ll == l))
        lesion_by_class[majority_label].append(lid)

    train_ids, val_ids = [], []
    for label in sorted(lesion_by_class.keys()):
        lids = lesion_by_class[label]
        rng.shuffle(lids)
        n_val = max(1, int(len(lids) * val_frac))
        val_lids = set(lids[:n_val])
        for lid in lids:
            for img_id, img_label in lesion_to_images[lid]:
                if lid in val_lids:
                    val_ids.append((img_id, img_label))
                else:
                    train_ids.append((img_id, img_label))
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Preload all images into RAM
# ---------------------------------------------------------------------------

def _load_one(args):
    img_id, img_dir, size = args
    path = os.path.join(img_dir, f"{img_id}.jpg")
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return img_id, np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return img_id, np.zeros((size, size, 3), dtype=np.uint8)


def preload_images(image_ids, img_dir, size=IMG_SIZE, max_workers=16):
    """Load all images into a dict {image_id: np.array(H,W,3)}."""
    tasks = [(img_id, img_dir, size) for img_id in image_ids]
    images = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for img_id, arr in pool.map(_load_one, tasks):
            images[img_id] = arr
    return images


# ---------------------------------------------------------------------------
# Dataset (images in RAM)
# ---------------------------------------------------------------------------

class ISIC_MEL_NV_Dataset(Dataset):
    def __init__(self, samples, image_cache, transform=None):
        self.samples = samples
        self.image_cache = image_cache
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        arr = self.image_cache[img_id]
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, label, img_id


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Main data builder
# ---------------------------------------------------------------------------

def build_dataloaders(batch_size=64, num_workers=NUM_WORKERS):
    train_img_dir = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")
    test_img_dir = os.path.join(DATA_DIR, "ISIC_2019_Test_Input")

    # Load labels
    train_labels = load_labels(os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv"))
    test_labels = load_labels(os.path.join(DATA_DIR, "ISIC_2019_Test_GroundTruth.csv"))
    metadata = load_metadata(os.path.join(DATA_DIR, "ISIC_2019_Training_Metadata.csv"))

    # Patient-level stratified split
    train_samples, val_samples = patient_stratified_split(train_labels, metadata)
    test_samples = [(img_id, label) for img_id, label in test_labels.items()]

    # Print split stats
    train_mel = sum(1 for _, l in train_samples if l == 1)
    val_mel = sum(1 for _, l in val_samples if l == 1)
    test_mel = sum(1 for _, l in test_samples if l == 1)
    print(f"Train: {len(train_samples)} (MEL={train_mel}, NV={len(train_samples)-train_mel})")
    print(f"Val:   {len(val_samples)} (MEL={val_mel}, NV={len(val_samples)-val_mel})")
    print(f"Test:  {len(test_samples)} (MEL={test_mel}, NV={len(test_samples)-test_mel})")

    # Preload ALL images into RAM
    all_train_ids = list(set(s[0] for s in train_samples + val_samples))
    all_test_ids = list(set(s[0] for s in test_samples))
    print(f"Preloading {len(all_train_ids)} train+val images into RAM...")
    train_cache = preload_images(all_train_ids, train_img_dir)
    print(f"Preloading {len(all_test_ids)} test images into RAM...")
    test_cache = preload_images(all_test_ids, test_img_dir)
    image_cache = {**train_cache, **test_cache}
    ram_mb = sum(v.nbytes for v in image_cache.values()) / (1024**2)
    print(f"Total images in RAM: {len(image_cache)}, {ram_mb:.0f} MB")

    # Build datasets
    train_ds = ISIC_MEL_NV_Dataset(train_samples, image_cache, get_train_transform())
    val_ds = ISIC_MEL_NV_Dataset(val_samples, image_cache, get_val_transform())
    test_ds = ISIC_MEL_NV_Dataset(test_samples, image_cache, get_val_transform())

    # Weighted sampler for class imbalance (oversample MEL)
    train_labels_list = [l for _, l in train_samples]
    class_counts = [train_labels_list.count(c) for c in range(2)]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[l] for l in train_labels_list]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    info = {
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "class_counts_train": class_counts,
        "classes": CLASSES,
    }
    return train_loader, val_loader, test_loader, info
