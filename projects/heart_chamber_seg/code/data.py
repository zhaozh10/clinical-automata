"""Data pipeline for PAXRay++ cardiac segmentation (frontal + lateral).

The 7 cardiac channels are multi-label and overlap heavily (a pixel inside the
cardiac silhouette belongs to `heart` and to whichever chambers project onto it),
so targets stay as independent binary planes -- never argmax.

Only channel PRIMARY_CH ("heart", the cardiac silhouette) drives model selection.
The chamber channels are trained and reported but are atlas priors, not
radiographically observable structures: a plain film carries no evidence that
separates LA from LV. See USAGE_MANUAL.md.
"""
import os
import re
import json
import collections
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from prepare import DATA_DIR, EXT_DATA_DIR

PAX_ROOT = DATA_DIR["chest x-ray (anatomy segmentation)"]
IMG_ROOT = os.path.join(PAX_ROOT, "paxray_images_unfiltered", "images_patlas")
CACHE_DIR = os.path.join(EXT_DATA_DIR, "jul17_heart", "cache")

# Category ids into PAXRay's 159-class dictionary.
HEART_CATS = [115, 121, 122, 123, 124, 125, 126]
HEART_NAMES = ["cardiomediastinum", "heart", "atrium_left", "atrium_right",
               "myocardium", "ventricle_left", "ventricle_right"]
N_CH = len(HEART_CATS)
PRIMARY_CH = 1                      # "heart" -- the only channel used for model selection
CHAMBER_CH = [2, 3, 4, 5, 6]        # reported separately, never used for selection

IMG_SIZE = 512                      # native PAXRay resolution; also xraydino's native grid
VAL_FRAC = 0.2
SPLIT_SEED = 17


def patient_key(stem: str) -> str:
    """Group images by patient, not by study.

    patlas ids repeat the same patient across scans (1012a..1012e), so the
    trailing letter must be stripped or a val split leaks patients into train.
    RSNAPE stems are <study>_<series>; the study uid identifies the patient.
    """
    m = re.match(r"^(\d+)[a-z]$", stem)
    if m:
        return "patlas_" + m.group(1)
    if stem.startswith("RSNAPE_"):
        return "_".join(stem.split("_")[:2])
    return stem


def _stem_of(file_name: str) -> str:
    return file_name.rsplit("_", 1)[0]


def _view_of(file_name: str) -> str:
    return "lateral" if "lateral" in file_name else "frontal"


# --------------------------------------------------------------------------
# Cache construction (one-time).  The dataset storage is NFS: a cold single-image
# read costs ~83 ms of which only ~6 ms is PNG decode.  Per-file reads would
# spend ~15 min/epoch in network latency, so the whole split is packed into
# two flat arrays that get paged into RAM once (~6 GB) and then cost nothing.
# --------------------------------------------------------------------------

def _read_one(args):
    idx, file_name = args
    img = cv2.imread(os.path.join(IMG_ROOT, file_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"unreadable image: {file_name}")
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return idx, img


def _decode_masks(ann_by_cat):
    from pycocotools import mask as maskutil
    out = np.zeros((N_CH, IMG_SIZE, IMG_SIZE), dtype=bool)
    for k, cat in enumerate(HEART_CATS):
        seg = ann_by_cat.get(cat)
        if seg is None:
            continue
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode()
        out[k] = maskutil.decode(rle).astype(bool)
    return out


def _build_split(split_name, json_name, workers=12):
    """Decode one PAXRay json into flat image/mask arrays + an index."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    img_path = os.path.join(CACHE_DIR, f"{split_name}_imgs.npy")
    msk_path = os.path.join(CACHE_DIR, f"{split_name}_masks.npy")
    idx_path = os.path.join(CACHE_DIR, f"{split_name}_index.json")
    if all(os.path.exists(p) for p in (img_path, msk_path, idx_path)):
        return

    print(f"[cache] building {split_name} from {json_name} ...", flush=True)
    with open(os.path.join(PAX_ROOT, json_name)) as f:
        j = json.load(f)

    # The json RLE is the *filtered* GT -- the version the official metrics use.
    # (Cross-checked against labels.zip: IoU 0.97-0.99, difference is edge cleanup.)
    id2fn = {i["id"]: i["file_name"] for i in j["images"]}
    by_img = collections.defaultdict(dict)
    for a in j["annotations"]:
        if a["category_id"] in HEART_CATS:
            by_img[a["image_id"]][a["category_id"]] = a["segmentation"]

    img_ids = sorted(id2fn.keys())
    files = [id2fn[i] for i in img_ids]
    n = len(files)

    imgs = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                     shape=(n, IMG_SIZE, IMG_SIZE))
    # 7 binary planes bit-packed along the last axis: 7*512*64 B = 229 KB/image.
    masks = np.lib.format.open_memmap(msk_path, mode="w+", dtype=np.uint8,
                                      shape=(n, N_CH, IMG_SIZE, IMG_SIZE // 8))

    with Pool(workers) as pool:
        for done, (i, img) in enumerate(
                pool.imap_unordered(_read_one, list(enumerate(files)), chunksize=16)):
            imgs[i] = img
            if done % 2000 == 0:
                print(f"[cache]   images {done}/{n}", flush=True)
    for i, img_id in enumerate(img_ids):
        masks[i] = np.packbits(_decode_masks(by_img[img_id]), axis=-1)
        if i % 2000 == 0:
            print(f"[cache]   masks {i}/{n}", flush=True)
    imgs.flush()
    masks.flush()

    meta = {
        "files": files,
        "stems": [_stem_of(f) for f in files],
        "patients": [patient_key(_stem_of(f)) for f in files],
        "views": [_view_of(f) for f in files],
    }
    with open(idx_path, "w") as f:
        json.dump(meta, f)
    print(f"[cache] {split_name}: {n} images, {len(set(meta['patients']))} patients", flush=True)


def build_cache():
    _build_split("trainval", "paxray_train_val.json")
    _build_split("test", "paxray_test.json")


def _load_split(split_name):
    img_path = os.path.join(CACHE_DIR, f"{split_name}_imgs.npy")
    msk_path = os.path.join(CACHE_DIR, f"{split_name}_masks.npy")
    idx_path = os.path.join(CACHE_DIR, f"{split_name}_index.json")
    with open(idx_path) as f:
        meta = json.load(f)
    return np.load(img_path, mmap_mode="r"), np.load(msk_path, mmap_mode="r"), meta


def make_splits():
    """train / val from paxray_train_val.json, split at PATIENT level.

    The official test json is held out untouched; it is already disjoint from
    train_val at patient level (verified: 0 shared patlas ids, 0 shared RSNAPE
    studies).  Both views of a patient always land on the same side -- they are
    two projections of one CT volume, so splitting them apart is leakage.
    """
    build_cache()
    imgs, masks, meta = _load_split("trainval")
    patients = sorted(set(meta["patients"]))
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(patients)
    n_val = int(round(len(patients) * VAL_FRAC))
    val_pat = set(patients[:n_val])
    train_idx = [i for i, p in enumerate(meta["patients"]) if p not in val_pat]
    val_idx = [i for i, p in enumerate(meta["patients"]) if p in val_pat]
    assert not (set(meta["patients"][i] for i in train_idx)
                & set(meta["patients"][i] for i in val_idx)), "patient leak"
    return (imgs, masks, meta), train_idx, val_idx


class HeartDataset(Dataset):
    def __init__(self, arrays, indices, train=False):
        self.imgs, self.masks, self.meta = arrays
        self.indices = list(indices)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = torch.from_numpy(np.ascontiguousarray(self.imgs[j]))          # uint8 (512,512)
        msk = torch.from_numpy(np.ascontiguousarray(self.masks[j]))         # uint8 packed
        return img, msk, j


def unpack_masks(packed: torch.Tensor) -> torch.Tensor:
    """(B,7,512,64) uint8 -> (B,7,512,512) float, on device."""
    bits = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    out = (packed.unsqueeze(-1) >> bits) & 1
    return out.flatten(-2).float()


# --------------------------------------------------------------------------
# Augmentation (GPU, batched).
#
# No horizontal flip: the heart is left-sided, so a mirrored chest X-ray is
# dextrocardia -- an anatomically implausible sample that teaches the model the
# wrong laterality prior on exactly the cases where laterality matters.
#
# The photometric block is domain randomisation for the DRR -> real-film gap:
# these images are CT projections, so they lack the scatter, quantum noise and
# exposure spread of a real radiograph.  Jittering those axes stops the encoder
# from keying on DRR-specific intensity statistics.
# --------------------------------------------------------------------------

AUG = os.environ.get("AUG", "strong")   # "base" | "strong" (heavier DRR->real randomisation); strong is the winning recipe


def augment(img, msk, gen=None):
    """img (B,1,H,W) float in [0,1]; msk (B,7,H,W) float. Returns same shapes."""
    B = img.shape[0]
    dev = img.device
    strong = AUG == "strong"

    def rand(lo, hi, shape=(B,)):
        return torch.rand(shape, device=dev, generator=gen) * (hi - lo) + lo

    # --- geometry: small rigid jitter, mirrors patient positioning only
    ang = rand(-10, 10) * np.pi / 180
    scale = rand(0.90, 1.12)
    tx, ty = rand(-0.05, 0.05), rand(-0.05, 0.05)
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    msk = F.grid_sample(msk, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    # --- photometric / acquisition simulation
    glo, ghi = (0.5, 1.9) if strong else (0.65, 1.5)
    img = img.clamp(1e-4, 1) ** rand(glo, ghi).view(B, 1, 1, 1)             # exposure / gamma
    clo, chi = (0.7, 1.4) if strong else (0.8, 1.25)
    img = ((img - 0.5) * rand(clo, chi).view(B, 1, 1, 1) + 0.5
           + rand(-0.08, 0.08).view(B, 1, 1, 1))                            # window level/width

    if strong:
        # local contrast swing: real films carry grid/processing texture and a
        # spatially varying response that a globally-normalised DRR never shows.
        low = F.avg_pool2d(img, 65, stride=1, padding=32)
        img = img + (img - low) * rand(-0.3, 0.7).view(B, 1, 1, 1)          # unsharp-ish local contrast
        # correlated (blurred) noise ~ quantum + scatter grain, not white noise
        nz = torch.randn(img.shape, device=dev, generator=gen)
        nz = F.avg_pool2d(nz, 3, stride=1, padding=1)
        img = img + nz * rand(0.0, 0.09).view(B, 1, 1, 1)
    else:
        img = img + torch.randn(img.shape, device=dev, generator=gen) * rand(0.0, 0.04).view(B, 1, 1, 1)

    # scatter: a real film's low-frequency veiling glare, absent from DRRs
    p_scatter = 0.8 if strong else 0.5
    if gen is None or torch.rand(1, generator=gen).item() < p_scatter:
        blur = F.avg_pool2d(img, 33, stride=1, padding=16)
        smax = 0.45 if strong else 0.25
        img = img + blur * rand(0.0, smax).view(B, 1, 1, 1)

    return img.clamp(0, 1), msk
