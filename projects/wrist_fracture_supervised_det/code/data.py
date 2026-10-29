"""Fracture detection data pipeline for GRAZPEDWRI-DX."""

import os
import json
import csv
import random
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from prepare import DATA_DIR

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOT_CSV = os.path.join(DATA_DIR, "annotations_voc.csv")
SPLIT_JSON = os.path.join(DATA_DIR, "train_test_split.json")

TARGET_CLASS = "fracture"
VAL_RATIO = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_fracture_annotations():
    """Return {filestem: [[x1,y1,x2,y2], ...]} for fracture boxes only."""
    annots = defaultdict(list)
    with open(ANNOT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["class_name"] == TARGET_CLASS:
                x1, y1 = int(row["xmin"]), int(row["ymin"])
                x2, y2 = int(row["xmax"]), int(row["ymax"])
                if x2 > x1 and y2 > y1:
                    annots[row["filestem"]].append([x1, y1, x2, y2])
    return dict(annots)


def load_splits():
    """Patient-level train/val/test split. Val = 20 % of original train patients."""
    with open(SPLIT_JSON) as f:
        sp = json.load(f)
    train_all, test = sp["train"], sp["test"]

    pid = lambda fs: fs.split("_")[0]
    patients = sorted(set(pid(fs) for fs in train_all))
    rng = random.Random(SEED)
    rng.shuffle(patients)
    val_pids = set(patients[: int(len(patients) * VAL_RATIO)])

    train = [fs for fs in train_all if pid(fs) not in val_pids]
    val = [fs for fs in train_all if pid(fs) in val_pids]
    return train, val, test


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FractureDataset(Dataset):
    def __init__(self, filestems, annotations, transforms=None):
        self.filestems = filestems
        self.annots = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.filestems)

    def __getitem__(self, idx):
        fs = self.filestems[idx]
        # 16-bit grayscale → float32 [0, 1] → 3-channel (for pretrained backbone)
        arr = np.array(Image.open(os.path.join(IMG_DIR, f"{fs}.png")), dtype=np.float32)
        mx = arr.max()
        if mx > 0:
            arr /= mx
        h, w = arr.shape
        img = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1).contiguous()

        bboxes = self.annots.get(fs, [])
        if bboxes:
            boxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(h, w))
            labels = torch.ones(len(bboxes), dtype=torch.int64)
        else:
            boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32), format="XYXY", canvas_size=(h, w)
            )
            labels = torch.zeros(0, dtype=torch.int64)

        img = tv_tensors.Image(img)
        target = {"boxes": boxes, "labels": labels, "image_id": idx, "filestem": fs}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img.float(), {
            "boxes": target["boxes"].float(),
            "labels": target["labels"],
            "image_id": target["image_id"],
            "filestem": target["filestem"],
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ClampBoundingBoxes(),
        T.SanitizeBoundingBoxes(min_size=2),
    ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(batch_size=4, num_workers=4, distributed=False,
                    world_size=1, rank=0):
    annots = load_fracture_annotations()
    train_fs, val_fs, test_fs = load_splits()

    train_ds = FractureDataset(train_fs, annots, get_train_transforms())
    val_ds = FractureDataset(val_fs, annots)
    test_ds = FractureDataset(test_fs, annots)

    sampler = (DistributedSampler(train_ds, world_size, rank, shuffle=True)
               if distributed else None)
    kw = dict(num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
              persistent_workers=num_workers > 0)

    train_ld = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=(sampler is None), sampler=sampler,
                          drop_last=True, **kw)
    # Val / test are non-distributed — evaluated only on rank 0
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    return train_ld, val_ld, test_ld, sampler
