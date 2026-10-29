"""Data pipeline for pneumothorax classification on NIH ChestX-ray14 (512x512).

Key design decision for this project: the *chest drain / chest tube* is a strong
confounder.  In NIH-CXR14, a large fraction of pneumothorax-positive images were
acquired *after* a chest drain was inserted, so a naive model can reach decent
AUROC just by detecting drains ("drain present -> pneumothorax").  This module
therefore exposes, for every image, both the pneumothorax label and a chest-drain
label (human consensus where available, model pseudo-label otherwise), and a
joint ``group`` id (PTX x drain).  Downstream training / evaluation code uses
these to (a) evaluate *drain-stratified* performance and (b) build debiasing
samplers that make the drain label non-predictive of pneumothorax in a batch.

Splits: the master list ships a fixed *patient-level* train/valid/test split
(verified disjoint at patient level); we use it as-is.

Performance: this node has very few CPU cores, so per-batch PNG decode is the
bottleneck.  On first use we build a uint8 memmap of all images at 512x512 under
EXT_DATA_DIR (the original dataset is never modified); subsequent reads are
near-instant (and fully RAM-cached after the first epoch).
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as T

from prepare import DATA_DIR, EXT_DATA_DIR

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
LABELS_DIR = os.path.dirname(DATA_DIR.rstrip("/"))                     # .../NIH_ChestX-ray14
MASTER_CSV = os.path.join(LABELS_DIR, "final_cxr14_master_list.csv")   # image-level labels + split
DRAIN_CSV = os.path.join(LABELS_DIR, "NIH-CX14_ChestDrain_full_pseudo.csv")  # drain label (gt|pseudo) for every image
IMG_ROOT = os.path.join(DATA_DIR, "preprocessed")                      # .../CXR14/preprocessed/<img_rel_path>

CACHE_DIR = os.path.join(EXT_DATA_DIR, "cxr14_cache")
CACHE_DAT = os.path.join(CACHE_DIR, "images_u8_512.dat")               # memmap (N, 512, 512) uint8
CACHE_IDX = os.path.join(CACHE_DIR, "cache_index.csv")                 # image_id -> row in master order

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMG_SIZE = 512                       # native preprocessed size; pneumothorax features (thin pleural line) are subtle
NATIVE_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# group ids: PTX x drain  ->  0: PTX-/drain-,  1: PTX-/drain+,  2: PTX+/drain-,  3: PTX+/drain+
GROUP_NAMES = {0: "PTXneg_drainneg", 1: "PTXneg_drainpos", 2: "PTXpos_drainneg", 3: "PTXpos_drainpos"}


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def load_metadata() -> pd.DataFrame:
    """Return one row per image with: image_id, patient_id, split, path,
    ptx (0/1), drain (0/1, gt-where-available else pseudo), drain_prob,
    drain_is_gt (0/1), group (0..3).  Order matches `final_cxr14_master_list.csv`."""
    m = pd.read_csv(MASTER_CSV)
    d = pd.read_csv(DRAIN_CSV)  # columns: image_id, drain_label, drain_prob, source
    df = m.merge(d, on="image_id", how="left")
    assert df["drain_label"].isna().sum() == 0, "every image must have a drain label"

    df["ptx"] = df["pneumothorax"].astype(int)
    df["drain"] = df["drain_label"].astype(int)
    df["drain_is_gt"] = (df["source"].astype(str) == "gt").astype(int)
    df["group"] = df["ptx"].values * 2 + df["drain"].values
    df["path"] = df["img_rel_path"].apply(lambda p: os.path.join(IMG_ROOT, p))
    df["cache_idx"] = np.arange(len(df), dtype=np.int64)   # row in the memmap
    keep = [
        "image_id", "patient_id", "split", "path", "cache_idx", "ptx", "drain",
        "drain_prob", "drain_is_gt", "group", "view_position",
    ]
    return df[keep].reset_index(drop=True)


def describe_splits(df: pd.DataFrame) -> str:
    lines = []
    for sp in ["train", "valid", "test"]:
        sub = df[df.split == sp]
        lines.append(f"[{sp}] n={len(sub)} patients={sub.patient_id.nunique()} "
                     f"ptx+={int(sub.ptx.sum())} drain+={int(sub.drain.sum())}")
        for g in range(4):
            n = int((sub.group == g).sum())
            lines.append(f"    {GROUP_NAMES[g]:>18s}: {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Image cache (uint8 memmap at 512x512), built once under EXT_DATA_DIR
# --------------------------------------------------------------------------- #
def _decode_one(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != (NATIVE_SIZE, NATIVE_SIZE):
        im = im.resize((NATIVE_SIZE, NATIVE_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def build_image_cache(df: pd.DataFrame | None = None, workers: int = 6, verbose: bool = True):
    """Build the uint8 memmap of all images (idempotent). The original dataset
    is never modified — this only writes new files under EXT_DATA_DIR."""
    if df is None:
        df = load_metadata()
    n = len(df)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_DAT) and os.path.exists(CACHE_IDX):
        idx = pd.read_csv(CACHE_IDX)
        expected_bytes = n * NATIVE_SIZE * NATIVE_SIZE
        if len(idx) == n and os.path.getsize(CACHE_DAT) == expected_bytes:
            if verbose:
                print(f"[cache] reuse existing cache: {CACHE_DAT} ({expected_bytes/1e9:.1f} GB, {n} imgs)")
            return
        if verbose:
            print("[cache] existing cache incomplete/mismatched — rebuilding")

    tmp = CACHE_DAT + ".tmp"
    mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(n, NATIVE_SIZE, NATIVE_SIZE))
    paths = df["path"].tolist()
    if verbose:
        print(f"[cache] building {tmp}  ({n} imgs, {(n*NATIVE_SIZE*NATIVE_SIZE)/1e9:.1f} GB) with {workers} workers ...")
    import time as _time
    t0 = _time.time()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, arr in enumerate(pool.imap(_decode_one, paths, chunksize=64)):
                mm[i] = arr
                if verbose and (i + 1) % 10000 == 0:
                    print(f"[cache]   {i+1}/{n}  ({_time.time()-t0:.0f}s)", flush=True)
    else:
        for i, p in enumerate(paths):
            mm[i] = _decode_one(p)
            if verbose and (i + 1) % 10000 == 0:
                print(f"[cache]   {i+1}/{n}  ({_time.time()-t0:.0f}s)", flush=True)
    mm.flush()
    del mm
    os.replace(tmp, CACHE_DAT)
    df[["image_id", "cache_idx"]].to_csv(CACHE_IDX, index=False)
    if verbose:
        print(f"[cache] done in {_time.time()-t0:.0f}s -> {CACHE_DAT}")


_CACHE_MM = None  # per-process lazily-opened read-only memmap


def _get_cache(n: int):
    global _CACHE_MM
    if _CACHE_MM is None:
        if not os.path.exists(CACHE_DAT):
            raise FileNotFoundError(f"image cache missing: {CACHE_DAT} — run data.build_image_cache() first")
        _CACHE_MM = np.memmap(CACHE_DAT, dtype=np.uint8, mode="r", shape=(n, NATIVE_SIZE, NATIVE_SIZE))
    return _CACHE_MM


def prewarm_cache(n_total: int, verbose: bool = True):
    """Sequentially read the whole image memmap so the OS page-cache holds it
    (the node has ~1 TB RAM; the cache is ~29 GB).  After this, all random
    image reads during training/eval are RAM-speed.  Counts as startup."""
    import time as _time
    mm = _get_cache(n_total)
    t0 = _time.time()
    chunk = 2048
    acc = 0
    for i in range(0, n_total, chunk):
        acc += int(mm[i:i + chunk].sum())               # sequential read of the whole file
    if verbose:
        gb = n_total * NATIVE_SIZE * NATIVE_SIZE / 1e9
        print(f"[cache] prewarmed {gb:.0f} GB ({n_total} imgs) into page cache in {_time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# Transforms  (medically restrained augmentation; input is single-channel uint8 [H,W])
# --------------------------------------------------------------------------- #
class GrayToRGB(T.Transform):
    """Replicate a 1-channel image to 3 channels (so ImageNet-pretrained backbones apply)."""
    def transform(self, inpt, params):
        return inpt.repeat(3, 1, 1) if inpt.shape[-3] == 1 else inpt


def build_transforms(train: bool, img_size: int = IMG_SIZE):
    if train:
        return T.Compose([
            T.ToImage(),                                       # uint8 [1,H,W]
            T.RandomResizedCrop(img_size, scale=(0.88, 1.0), ratio=(0.92, 1.08), antialias=True),
            T.RandomHorizontalFlip(0.5),                       # PTX appearance is side-agnostic
            T.RandomRotation(7, interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15),     # mild acquisition variation
            GrayToRGB(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.ToImage(),
        T.Resize(img_size, antialias=True) if img_size != NATIVE_SIZE else T.Identity(),
        GrayToRGB(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, n_total: int, use_cache: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.n_total = n_total
        self.use_cache = use_cache
        self.cidx = self.df["cache_idx"].values.astype(np.int64)
        self.paths = self.df["path"].values
        self.ptx = self.df["ptx"].values.astype(np.float32)
        self.drain = self.df["drain"].values.astype(np.float32)
        self.group = self.df["group"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        if self.use_cache:
            arr = np.array(_get_cache(self.n_total)[self.cidx[i]])     # [H,W] uint8
            img = arr[None, :, :]                                      # [1,H,W]
        else:
            img = np.asarray(Image.open(self.paths[i]).convert("L"), dtype=np.uint8)[None, :, :]
        x = self.transform(torch.from_numpy(img))
        return {
            "image": x,
            "ptx": torch.tensor(self.ptx[i], dtype=torch.float32),
            "drain": torch.tensor(self.drain[i], dtype=torch.float32),
            "group": torch.tensor(int(self.group[i]), dtype=torch.long),
            "row": i,  # index into self.df (for writing per-sample predictions)
        }


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
def label_balanced_weights(df: pd.DataFrame, pos_frac: float = 0.25) -> torch.Tensor:
    """Sample weights so that, in expectation, a fraction ``pos_frac`` of each
    batch is pneumothorax-positive.  This is the *naive* (confounder-unaware)
    imbalance handling used by the baseline."""
    y = df["ptx"].values
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    w = np.where(y == 1, pos_frac / n_pos, (1.0 - pos_frac) / n_neg).astype(np.float64)
    return torch.as_tensor(w, dtype=torch.double)


def group_balanced_weights(df: pd.DataFrame, drain_pos_frac: float = 0.5,
                           ptx_pos_frac: float = 0.5) -> torch.Tensor:
    """Sample weights making the drain label *non-predictive* of pneumothorax in
    a batch: P(drain+) = drain_pos_frac and, within each drain stratum,
    P(PTX+) = ptx_pos_frac.  (Used by debiasing runs, not the baseline.)"""
    g = df["group"].values  # 0:PTX-/drain-  1:PTX-/drain+  2:PTX+/drain-  3:PTX+/drain+
    target = {
        0: (1 - ptx_pos_frac) * (1 - drain_pos_frac),
        1: (1 - ptx_pos_frac) * drain_pos_frac,
        2: ptx_pos_frac * (1 - drain_pos_frac),
        3: ptx_pos_frac * drain_pos_frac,
    }
    w = np.zeros(len(g), dtype=np.float64)
    for gid, t in target.items():
        m = (g == gid)
        w[m] = t / max(int(m.sum()), 1)
    return torch.as_tensor(w, dtype=torch.double)


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def build_dataloaders(df: pd.DataFrame, batch_size: int = 96, num_workers: int = 5,
                      img_size: int = IMG_SIZE, sampler_kind: str = "label_balanced",
                      pos_frac: float = 0.30, gb_drain_pos_frac: float = 0.5,
                      gb_ptx_pos_frac: float = 0.5, eval_batch_size: int | None = None,
                      use_cache: bool = True):
    """Returns (train_loader, val_loader, test_loader, train_df, val_df, test_df).

    The returned per-split DataFrames are aligned with the ``row`` index returned
    by the datasets, so callers can attach predictions back to image ids.
    """
    eval_batch_size = eval_batch_size or batch_size
    n_total = len(df)
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "valid"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)

    train_ds = CXRDataset(tr, build_transforms(True, img_size), n_total, use_cache)
    val_ds = CXRDataset(va, build_transforms(False, img_size), n_total, use_cache)
    test_ds = CXRDataset(te, build_transforms(False, img_size), n_total, use_cache)

    sampler, shuffle = None, True
    if sampler_kind == "label_balanced":
        sampler = WeightedRandomSampler(label_balanced_weights(tr, pos_frac), num_samples=len(tr), replacement=True)
        shuffle = False
    elif sampler_kind == "group_balanced":
        sampler = WeightedRandomSampler(group_balanced_weights(tr, gb_drain_pos_frac, gb_ptx_pos_frac),
                                        num_samples=len(tr), replacement=True)
        shuffle = False
    elif sampler_kind == "none":
        sampler, shuffle = None, True
    else:
        raise ValueError(f"unknown sampler_kind={sampler_kind}")

    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0,
                  prefetch_factor=2 if num_workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader, tr, va, te


if __name__ == "__main__":
    df = load_metadata()
    print(f"total images: {len(df)}  patients: {df.patient_id.nunique()}")
    print(describe_splits(df))
    if "--build-cache" in sys.argv:
        build_image_cache(df, workers=6)
    # quick sanity load (falls back to PNG decode if cache absent)
    use_cache = os.path.exists(CACHE_DAT)
    _, val_loader, _, _, va, _ = build_dataloaders(df, batch_size=4, num_workers=2, use_cache=use_cache)
    b = next(iter(val_loader))
    print("batch image:", b["image"].shape, b["image"].dtype, b["image"].min().item(), b["image"].max().item(),
          "ptx:", b["ptx"].tolist(), "group:", b["group"].tolist())
