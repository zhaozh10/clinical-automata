"""
ISIC 2019 Skin Lesion Dataset — data loading, splitting, augmentation.
Classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC (8 classes, UNK only in test)
Patient-level (lesion_id) stratified split for train/val.
"""

import os
import io
import csv
import random
import time
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from prepare import DATA_DIR, EXT_DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 384  # good balance for dermoscopy: preserves detail, fits in memory

SEED = 42
VAL_FRACTION = 0.20  # 20% of training data for validation


# ---------------------------------------------------------------------------
# Local NVMe cache — pre-resize images from NFS and save to local disk
# ---------------------------------------------------------------------------
LOCAL_CACHE_DIR = "/tmp/isic2019_cache"
CACHE_SIZE = int(IMG_SIZE * 1.15)  # slightly larger for RandomResizedCrop headroom


def _ensure_local_cache(nfs_img_dir, split_name):
    """Pre-resize images from NFS to local NVMe cache. Returns local dir path."""
    local_dir = os.path.join(LOCAL_CACHE_DIR, f"{split_name}_{CACHE_SIZE}")
    marker = os.path.join(local_dir, ".cache_complete")
    if os.path.exists(marker):
        print(f"Local cache hit: {local_dir}")
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    t0 = time.time()
    src_files = [f for f in os.listdir(nfs_img_dir) if f.endswith(".jpg")]
    print(f"Caching {len(src_files)} pre-resized ({CACHE_SIZE}px) images to {local_dir}...")

    def _resize_one(fname):
        src = os.path.join(nfs_img_dir, fname)
        dst = os.path.join(local_dir, fname)
        if not os.path.exists(dst):
            img = Image.open(src).convert("RGB")
            # Resize so shortest side = CACHE_SIZE, maintain aspect ratio
            w, h = img.size
            scale = CACHE_SIZE / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(dst, "JPEG", quality=95)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_resize_one, src_files))

    with open(marker, "w") as f:
        f.write("done")
    print(f"  cached {len(src_files)} images in {time.time()-t0:.1f}s")
    return local_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ground_truth(csv_path):
    """Return list of (image_id, class_index) from a ground-truth CSV."""
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image"]
            for cls in CLASSES:
                if float(row[cls]) == 1.0:
                    samples.append((img_id, CLASS_TO_IDX[cls]))
                    break
            else:
                # UNK or no label among our 8 classes — skip for training
                # but keep for test with label -1
                samples.append((img_id, -1))
    return samples


def _load_lesion_ids(metadata_path):
    """Return dict: image_id -> lesion_id (or None)."""
    mapping = {}
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = row.get("lesion_id", "").strip()
            mapping[row["image"]] = lid if lid else None
    return mapping


def _patient_stratified_split(samples, lesion_ids, val_frac, seed):
    """
    Split samples into train/val at lesion (patient) level.
    Samples without lesion_id are split at sample level.
    Stratified by class to preserve class balance.
    """
    rng = random.Random(seed)

    # Group samples by class
    class_samples = defaultdict(list)
    for img_id, label in samples:
        class_samples[label].append((img_id, label))

    train_samples, val_samples = [], []

    for label, items in class_samples.items():
        # Separate items with and without lesion_id
        lesion_groups = defaultdict(list)  # lesion_id -> [items]
        no_lesion = []
        for item in items:
            lid = lesion_ids.get(item[0])
            if lid:
                lesion_groups[lid].append(item)
            else:
                no_lesion.append(item)

        # Shuffle and split lesion groups
        group_keys = list(lesion_groups.keys())
        rng.shuffle(group_keys)
        n_val_groups = max(1, int(len(group_keys) * val_frac))

        val_keys = set(group_keys[:n_val_groups])
        for key in group_keys:
            dest = val_samples if key in val_keys else train_samples
            dest.extend(lesion_groups[key])

        # Split no-lesion samples at sample level
        rng.shuffle(no_lesion)
        n_val_no_lesion = max(1, int(len(no_lesion) * val_frac)) if no_lesion else 0
        val_samples.extend(no_lesion[:n_val_no_lesion])
        train_samples.extend(no_lesion[n_val_no_lesion:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ISICDataset(Dataset):
    def __init__(self, samples, img_dir, transform=None):
        """
        samples: list of (image_id, class_index)
        img_dir: path to image directory
        """
        self.samples = samples
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, img_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_class_weights(samples):
    """Compute inverse-frequency class weights for loss function."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, label in samples:
        if 0 <= label < NUM_CLASSES:
            counts[label] += 1
    # Inverse frequency, normalized
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def get_sampler_weights(samples):
    """Compute per-sample weights for WeightedRandomSampler (oversample minorities)."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, label in samples:
        if 0 <= label < NUM_CLASSES:
            counts[label] += 1
    class_weight = 1.0 / (counts + 1e-6)
    sample_weights = [class_weight[label] for _, label in samples]
    return torch.tensor(sample_weights, dtype=torch.float64)


def build_datasets():
    """
    Build train/val/test datasets and return them along with metadata.
    Returns: train_dataset, val_dataset, test_dataset, train_samples, val_samples, test_samples
    """
    train_img_dir = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")
    test_img_dir = os.path.join(DATA_DIR, "ISIC_2019_Test_Input")
    train_gt = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
    test_gt = os.path.join(DATA_DIR, "ISIC_2019_Test_GroundTruth.csv")
    train_meta = os.path.join(DATA_DIR, "ISIC_2019_Training_Metadata.csv")

    # Load all training samples and lesion IDs
    all_train = _load_ground_truth(train_gt)
    lesion_ids = _load_lesion_ids(train_meta)

    # Patient-level stratified split
    train_samples, val_samples = _patient_stratified_split(
        all_train, lesion_ids, VAL_FRACTION, SEED
    )

    # Test samples (includes UNK with label=-1)
    test_samples = _load_ground_truth(test_gt)

    # Cache images to local NVMe for fast I/O
    local_train_dir = _ensure_local_cache(train_img_dir, "train")
    local_test_dir = _ensure_local_cache(test_img_dir, "test")

    train_dataset = ISICDataset(train_samples, local_train_dir, get_train_transform())
    val_dataset = ISICDataset(val_samples, local_train_dir, get_val_transform())
    test_dataset = ISICDataset(test_samples, local_test_dir, get_val_transform())

    return train_dataset, val_dataset, test_dataset, train_samples, val_samples, test_samples


def build_dataloaders(batch_size=32, num_workers=4):
    """Build train/val/test DataLoaders with oversampling for train."""
    train_ds, val_ds, test_ds, train_samples, val_samples, test_samples = build_datasets()

    # Weighted random sampler for training (oversample minority classes)
    sampler_weights = get_sampler_weights(train_samples)
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_samples, val_samples, test_samples
