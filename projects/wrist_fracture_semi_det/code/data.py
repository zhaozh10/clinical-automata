"""
data.py — Prepare YOLO-format dataset for fracture detection on GRAZPEDWRI-DX.

- Single class: fracture
- Real bbox labels (540 training images) + pseudo-labels from weak supervision
- Patient-level train/val split (80/20 of training set)
- 16-bit grayscale → 8-bit conversion (p1-p99 windowing)
- Test: bbox-annotated test images + confirmed negatives
"""

import os
import sys
import json
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import DATA_DIR, EXT_DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations_voc_5pct.csv")
IMAGE_LABELS_FILE = os.path.join(DATA_DIR, "image_labels.csv")
DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")
SPLIT_FILE = os.path.join(DATA_DIR, "train_test_split.json")
YOLO_DIR = os.path.join(EXT_DATA_DIR, "grazpedwri_yolo_fracture")
YAML_PATH = os.path.join(YOLO_DIR, "dataset.yaml")
PSEUDO_IMG_DIR = os.path.join(EXT_DATA_DIR, "grazpedwri_pseudo_images")

VAL_RATIO = 0.2
SEED = 42
NUM_WORKERS = 8
NEG_RATIO = 2.0  # max negatives per real positive (patient-level subsampling)
PSEUDO_CONF = 0.5  # confidence threshold for pseudo-labels


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_split():
    with open(SPLIT_FILE) as f:
        d = json.load(f)
    return set(d["train"]), set(d["test"])


def load_fracture_bboxes():
    """filestem -> list of (xmin, ymin, xmax, ymax)"""
    bboxes = defaultdict(list)
    with open(ANNOTATIONS_FILE) as f:
        for row in csv.DictReader(f):
            if row["class_name"] == "fracture":
                bboxes[row["filestem"]].append(
                    (int(row["xmin"]), int(row["ymin"]),
                     int(row["xmax"]), int(row["ymax"])))
    return dict(bboxes)


def load_image_labels():
    """Training image labels: filestem -> fracture_visible (0 or 1)."""
    labels = {}
    with open(IMAGE_LABELS_FILE) as f:
        for row in csv.DictReader(f):
            labels[row["filestem"]] = int(row["fracture_visible"])
    return labels


def load_all_fracture_labels():
    """All images (dataset.csv): filestem -> fracture_visible (0 or 1)."""
    labels = {}
    with open(DATASET_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["filestem"]] = 1 if row.get("fracture_visible") == "1" else 0
    return labels


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def get_patient_id(filestem):
    return filestem.split("_")[0]


def patient_level_split(filestems, val_ratio=VAL_RATIO, seed=SEED):
    """Patient-level train/val split."""
    patient_files = defaultdict(list)
    for fs in filestems:
        patient_files[get_patient_id(fs)].append(fs)

    patients = sorted(patient_files.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_ratio))
    val_pats = set(patients[:n_val])

    train = [fs for p in patients if p not in val_pats for fs in patient_files[p]]
    val = [fs for p in val_pats for fs in patient_files[p]]
    return train, val


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------
def _convert_one(args):
    """Convert a single 16-bit grayscale PNG to 8-bit (p1–p99 windowing)."""
    src, dst = args
    if os.path.exists(dst):
        return
    img = Image.open(src)
    arr = np.array(img, dtype=np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 > p1:
        arr = (np.clip(arr, p1, p99) - p1) / (p99 - p1) * 255.0
    else:
        arr = np.zeros_like(arr)
    Image.fromarray(arr.astype(np.uint8)).save(dst)


def voc_to_yolo(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------
def generate_pseudo_labels(model_path, conf_threshold=PSEUDO_CONF):
    """Run detector on fracture_visible=1 images without bbox annotations.
    Returns dict: filestem -> list of (cx, cy, w, h) normalized."""
    from ultralytics import YOLO

    train_set, _ = load_split()
    bboxes = load_fracture_bboxes()
    img_labels = load_image_labels()

    # Fracture-positive training images without bbox annotations
    unlabeled = [fs for fs, lbl in img_labels.items()
                 if lbl == 1 and fs in train_set and fs not in bboxes]
    print(f"Pseudo-labeling {len(unlabeled)} images (conf>{conf_threshold})...")

    # Convert to 8-bit
    os.makedirs(PSEUDO_IMG_DIR, exist_ok=True)
    tasks = []
    for fs in unlabeled:
        src = os.path.join(IMG_DIR, f"{fs}.png")
        dst = os.path.join(PSEUDO_IMG_DIR, f"{fs}.png")
        if os.path.exists(src):
            tasks.append((src, dst))
    print(f"  Converting {len(tasks)} images to 8-bit...")
    with Pool(NUM_WORKERS) as pool:
        pool.map(_convert_one, tasks)

    # Batch inference
    model = YOLO(model_path, task="detect")
    pseudo = {}
    paths = [os.path.join(PSEUDO_IMG_DIR, f"{fs}.png") for fs in unlabeled
             if os.path.exists(os.path.join(PSEUDO_IMG_DIR, f"{fs}.png"))]

    print(f"  Running inference on {len(paths)} images...")
    for i in range(0, len(paths), 64):
        batch = paths[i:i + 64]
        results = model(batch, conf=conf_threshold, verbose=False, imgsz=640)
        for p, r in zip(batch, results):
            fs = Path(p).stem
            if len(r.boxes) > 0:
                pseudo[fs] = [tuple(b.xywhn[0].cpu().tolist()) for b in r.boxes]

    print(f"  Result: {len(pseudo)} images with detections, "
          f"{len(unlabeled) - len(pseudo)} without")
    return pseudo


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------
def prepare_yolo_dataset(force=False, pseudo_labels=None):
    """Build YOLO-format dataset. Returns (yaml_path, n_train, n_val, n_test)."""
    if os.path.exists(YAML_PATH) and not force:
        n_tr = len(list(Path(YOLO_DIR, "images", "train").glob("*.png")))
        n_va = len(list(Path(YOLO_DIR, "images", "val").glob("*.png")))
        n_te = len(list(Path(YOLO_DIR, "images", "test").glob("*.png")))
        print(f"YOLO dataset already exists: train={n_tr}, val={n_va}, test={n_te}")
        return YAML_PATH, n_tr, n_va, n_te

    # Clean old train/val to prevent stale data
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            d = os.path.join(YOLO_DIR, sub, split)
            if os.path.exists(d):
                shutil.rmtree(d)

    print("Preparing YOLO-format dataset...")
    train_set, test_set = load_split()
    bboxes = load_fracture_bboxes()
    img_labels = load_image_labels()
    all_labels = load_all_fracture_labels()

    # --- Real training candidates: bbox images + subsampled negatives ---
    bbox_train = [fs for fs in bboxes if fs in train_set]
    neg_all = [fs for fs, lbl in img_labels.items()
               if lbl == 0 and fs in train_set]
    max_neg = int(NEG_RATIO * len(bbox_train))
    if len(neg_all) > max_neg:
        neg_by_pat = defaultdict(list)
        for fs in neg_all:
            neg_by_pat[get_patient_id(fs)].append(fs)
        neg_pats = sorted(neg_by_pat.keys())
        random.Random(SEED).shuffle(neg_pats)
        neg_train = []
        for p in neg_pats:
            neg_train.extend(neg_by_pat[p])
            if len(neg_train) >= max_neg:
                break
        neg_train = neg_train[:max_neg]
    else:
        neg_train = neg_all
    real_train = bbox_train + neg_train

    # --- Stratified patient-level train/val split on ALL training patients ---
    # Ensures bbox patients are distributed 80/20 (not all in one split)
    all_train_pids = sorted(set(get_patient_id(fs) for fs in train_set))
    bbox_pids = sorted(set(get_patient_id(fs) for fs in bbox_train))
    other_pids = [p for p in all_train_pids if p not in set(bbox_pids)]

    rng = random.Random(SEED)
    rng.shuffle(bbox_pids)
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(other_pids)

    n_val_bbox = max(1, int(len(bbox_pids) * VAL_RATIO))
    n_val_other = max(1, int(len(other_pids) * VAL_RATIO))
    val_pats = set(bbox_pids[:n_val_bbox]) | set(other_pids[:n_val_other])
    train_pats = set(bbox_pids[n_val_bbox:]) | set(other_pids[n_val_other:])

    train_files = [fs for fs in real_train if get_patient_id(fs) in train_pats]
    val_files = [fs for fs in real_train if get_patient_id(fs) in val_pats]

    # --- Add pseudo-labeled images to training only ---
    n_pseudo = 0
    if pseudo_labels:
        pseudo_train = [fs for fs in pseudo_labels
                        if get_patient_id(fs) in train_pats]
        train_files.extend(pseudo_train)
        n_pseudo = len(pseudo_train)

    # --- Test: bbox images + confirmed negatives ---
    bbox_test = [fs for fs in bboxes if fs in test_set]
    neg_test = [fs for fs in test_set
                if all_labels.get(fs, 0) == 0 and fs not in bboxes]
    all_test = bbox_test + neg_test

    tr_real = sum(1 for f in train_files if f in bboxes)
    va_pos = sum(1 for f in val_files if f in bboxes)
    print(f"Train: {len(train_files)} ({tr_real} real bbox, {n_pseudo} pseudo, "
          f"{len(train_files)-tr_real-n_pseudo} neg)")
    print(f"Val:   {len(val_files)} ({va_pos} pos, {len(val_files)-va_pos} neg)")
    print(f"Test:  {len(all_test)} ({len(bbox_test)} pos, {len(neg_test)} neg)")

    # --- Create directories ---
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

    splits = {"train": train_files, "val": val_files, "test": all_test}

    for split_name, filestems in splits.items():
        print(f"Converting {split_name} ({len(filestems)} images)...")
        tasks = []
        for fs in filestems:
            # Check both original and pseudo-converted directories
            dst = os.path.join(YOLO_DIR, "images", split_name, f"{fs}.png")
            src = os.path.join(IMG_DIR, f"{fs}.png")
            pseudo_src = os.path.join(PSEUDO_IMG_DIR, f"{fs}.png")
            if os.path.exists(pseudo_src) and not os.path.exists(dst):
                # Already 8-bit converted — just copy
                shutil.copy2(pseudo_src, dst)
            elif os.path.exists(src):
                tasks.append((src, dst))
        if tasks:
            with Pool(NUM_WORKERS) as pool:
                pool.map(_convert_one, tasks)

        # Write YOLO labels
        for fs in filestems:
            dst_lbl = os.path.join(YOLO_DIR, "labels", split_name, f"{fs}.txt")
            if fs in bboxes:
                # Real bbox annotation
                src = os.path.join(IMG_DIR, f"{fs}.png")
                if not os.path.exists(src):
                    continue
                img = Image.open(src)
                w, h = img.size
                with open(dst_lbl, "w") as f:
                    for bbox in bboxes[fs]:
                        cx, cy, bw, bh = voc_to_yolo(bbox, w, h)
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            elif pseudo_labels and fs in pseudo_labels:
                # Pseudo-label (already normalized YOLO format)
                with open(dst_lbl, "w") as f:
                    for (cx, cy, bw, bh) in pseudo_labels[fs]:
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                # Negative — empty label
                open(dst_lbl, "w").close()

    # --- Write dataset YAML ---
    with open(YAML_PATH, "w") as f:
        f.write(f"path: {YOLO_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  - fracture\n")

    n_tr, n_va, n_te = len(train_files), len(val_files), len(all_test)
    print(f"\nDone. train={n_tr}, val={n_va}, test={n_te}")
    print(f"YAML: {YAML_PATH}")
    return YAML_PATH, n_tr, n_va, n_te


if __name__ == "__main__":
    prepare_yolo_dataset(force="--force" in sys.argv)
