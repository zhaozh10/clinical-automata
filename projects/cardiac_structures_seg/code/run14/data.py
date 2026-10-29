"""Data pipeline for cardiac chamber segmentation on projection radiographs (PAXRay++).

Teaching-oriented task: the model segments the cardiac chambers, the great vessels
and the derived valve-plane bands on frontal AND lateral chest radiographs.

Key modelling point: PAXRay labels are HIERARCHICAL / OVERLAPPING because a
radiograph is a projection - e.g. the left ventricle and the right ventricle
share a large part of their frontal footprint. Therefore the target is a stack of
independent binary channels (multi-label, sigmoid), never an argmax label map.
"""

import io
import json
import os
import pickle
import re

import cv2
import numpy as np
import torch
from pycocotools import mask as maskutil
from torch.utils.data import Dataset

from prepare import DATA_DIR, EXT_DATA_DIR

cv2.setNumThreads(0)

ROOT = DATA_DIR["chest x-ray (anatomy segmentation)"]
IMAGES_ROOT = os.path.join(ROOT, "paxray_images_unfiltered", "images_patlas")
CACHE_DIR = os.path.join(EXT_DATA_DIR, "cache_cardiac")

IMG_SIZE = 512

# ---------------------------------------------------------------------------
# Channel definition
# ---------------------------------------------------------------------------
# (name, paxray_category_id).  Order is fixed and referenced by index everywhere.
ANNOTATED_CHANNELS = [
    ("heart", 121),
    ("atrium_left", 122),
    ("atrium_right", 123),
    ("myocardium", 124),
    ("ventricle_left", 125),
    ("ventricle_right", 126),
    ("aorta_ascending", 128),
    ("aorta_arch", 130),
    ("aorta_descending", 129),
    ("pulmonary_artery", 131),
    ("vena_cava_inferior", 132),
    ("cardiomediastinum", 115),
    ("trachea", 154),
    ("carina", 155),
    ("lung_right", 135),
    ("lung_left", 136),
]
# PAXRay's "pulmonary artery" label covers the whole arterial tree down into the
# lungs.  The border-forming and valve-relevant part is the central portion inside
# the mediastinal shadow (pulmonary trunk + main branches) - derived here.
DERIVED_STRUCTURES = ["pulmonary_trunk"]

# Valve planes are not annotated in PAXRay.  They are derived as the contact band
# between the two structures the valve separates (see _valve_band).
VALVE_CHANNELS = [
    ("valve_mitral", "atrium_left", "ventricle_left"),
    ("valve_tricuspid", "atrium_right", "ventricle_right"),
    ("valve_aortic", "ventricle_left", "aorta_ascending"),
    ("valve_pulmonary", "ventricle_right", "pulmonary_trunk"),
]

CHANNEL_NAMES = ([n for n, _ in ANNOTATED_CHANNELS] + DERIVED_STRUCTURES
                 + [n for n, _, _ in VALVE_CHANNELS])
NUM_CHANNELS = len(CHANNEL_NAMES)
IDX = {n: i for i, n in enumerate(CHANNEL_NAMES)}

# lungs are only annotated on frontal views -> loss/metrics must ignore them on lateral
LUNG_IDX = [IDX["lung_right"], IDX["lung_left"]]
CHAMBER_IDX = [IDX["atrium_left"], IDX["atrium_right"], IDX["ventricle_left"], IDX["ventricle_right"]]
VESSEL_IDX = [IDX["aorta_ascending"], IDX["aorta_arch"], IDX["aorta_descending"],
              IDX["pulmonary_artery"], IDX["pulmonary_trunk"], IDX["vena_cava_inferior"]]
VALVE_IDX = [IDX[n] for n, _, _ in VALVE_CHANNELS]

VALVE_BAND_RADIUS = 5  # px at 512 -> ~4 mm on each side of the interface


# ---------------------------------------------------------------------------
# Annotation cache
# ---------------------------------------------------------------------------
def _patient_id(file_name):
    stem = file_name.rsplit(".", 1)[0].replace("_frontal", "").replace("_lateral", "")
    if stem.startswith("RSNAPE_"):
        return "RSNAPE_" + stem.split("_")[1]
    m = re.fullmatch(r"(\d+)[a-z]", stem)
    if m:
        return "PAT_" + m.group(1)
    return stem


def _build_cache(split):
    """Extract only the cardiac categories from the (large) COCO json once."""
    src = os.path.join(ROOT, "paxray_train_val.json" if split == "train_val" else "paxray_test.json")
    wanted = {cid: name for name, cid in ANNOTATED_CHANNELS}
    with open(src) as f:
        coco = json.load(f)
    recs = {}
    for im in coco["images"]:
        recs[im["id"]] = {
            "file_name": im["file_name"],
            "view": "lateral" if "lateral" in im["file_name"] else "frontal",
            "patient": _patient_id(im["file_name"]),
            "rles": {},
        }
    for a in coco["annotations"]:
        cid = a["category_id"]
        if cid in wanted:
            seg = a["segmentation"]
            counts = seg["counts"]
            if isinstance(counts, str):
                counts = counts.encode()
            recs[a["image_id"]]["rles"][wanted[cid]] = {"size": seg["size"], "counts": counts}
    out = sorted(recs.values(), key=lambda r: r["file_name"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{split}.pkl"), "wb") as f:
        pickle.dump(out, f, protocol=4)
    return out


def load_records(split):
    path = os.path.join(CACHE_DIR, f"{split}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return _build_cache(split)


def split_train_val(records, val_frac=0.2, seed=1234):
    """Patient-level split: frontal and lateral of a patient always stay together."""
    patients = sorted({r["patient"] for r in records})
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(patients))
    n_val = int(round(val_frac * len(patients)))
    val_pat = {patients[i] for i in perm[:n_val]}
    train = [r for r in records if r["patient"] not in val_pat]
    val = [r for r in records if r["patient"] in val_pat]
    return train, val


def fixed_val_subset(val_records, n_per_view=300, seed=7):
    """Deterministic, view-balanced validation subset - kept constant across all runs."""
    rng = np.random.RandomState(seed)
    out = []
    for view in ("frontal", "lateral"):
        pool = [r for r in val_records if r["view"] == view]
        pool = sorted(pool, key=lambda r: r["file_name"])
        idx = rng.permutation(len(pool))[:n_per_view]
        out += [pool[i] for i in sorted(idx)]
    return out


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * VALVE_BAND_RADIUS + 1, 2 * VALVE_BAND_RADIUS + 1))


def _valve_band(a, b):
    """Contact band between two projected structures = dilate(a) AND dilate(b).

    On a projection the two structures adjoining a valve overlap or touch; the
    intersection of their dilated masks marks the plane of the valve orifice.
    """
    if a.max() == 0 or b.max() == 0:
        return np.zeros_like(a)
    da = cv2.dilate(a, _DILATE_KERNEL)
    db = cv2.dilate(b, _DILATE_KERNEL)
    band = (da & db).astype(np.uint8)
    if band.max() == 0:
        return band
    # keep the dominant component: valves are single, connected orifices
    n, lab, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    if n > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        band = (lab == keep).astype(np.uint8)
    return band


_NEAR_HEART_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))


def _pulmonary_trunk(pa, cardiomediastinum, heart):
    """Central pulmonary arteries: inside the mediastinal shadow and next to the heart."""
    if pa.max() == 0:
        return np.zeros_like(pa)
    near = cv2.dilate(heart, _NEAR_HEART_KERNEL) if heart.max() else np.ones_like(pa)
    t = (pa & cardiomediastinum & near).astype(np.uint8)
    if t.max() == 0:
        return t
    n, lab, stats, _ = cv2.connectedComponentsWithStats(t, 8)
    if n > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        t = (lab == keep).astype(np.uint8)
    return t


def build_mask(record):
    """Return (NUM_CHANNELS, H, W) uint8 mask stack for one radiograph."""
    m = np.zeros((NUM_CHANNELS, IMG_SIZE, IMG_SIZE), np.uint8)
    for name, _ in ANNOTATED_CHANNELS:
        rle = record["rles"].get(name)
        if rle is not None:
            m[IDX[name]] = maskutil.decode(rle)
    m[IDX["pulmonary_trunk"]] = _pulmonary_trunk(
        m[IDX["pulmonary_artery"]], m[IDX["cardiomediastinum"]], m[IDX["heart"]])
    for name, a, b in VALVE_CHANNELS:
        m[IDX[name]] = _valve_band(m[IDX[a]], m[IDX[b]])
    return m


MASK_CACHE_DIR = os.path.join(CACHE_DIR, "masks")


def cached_mask_path(file_name):
    return os.path.join(MASK_CACHE_DIR, file_name.replace(".png", ".npy"))


def load_mask(record):
    """Mask stack from the bit-packed cache if present, else built on the fly."""
    p = cached_mask_path(record["file_name"])
    if os.path.exists(p):
        packed = np.load(p)
        return np.unpackbits(packed, axis=-1, count=IMG_SIZE)
    return build_mask(record)


def _cache_worker(record):
    p = cached_mask_path(record["file_name"])
    if os.path.exists(p):
        return 0
    m = build_mask(record)
    tmp = p + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, np.packbits(m, axis=-1))
    os.replace(tmp, p)
    return 1


def build_mask_cache(splits=("train_val", "test"), procs=6):
    """One-off: materialise the mask stacks so the loader only reads bits."""
    from multiprocessing import Pool
    os.makedirs(MASK_CACHE_DIR, exist_ok=True)
    for split in splits:
        recs = load_records(split)
        with Pool(procs) as pool:
            n = sum(pool.imap_unordered(_cache_worker, recs, chunksize=16))
        print(f"{split}: {n} masks written ({len(recs)} total)", flush=True)


def channel_valid(view):
    """Per-channel supervision mask: lungs are unlabelled on lateral views."""
    valid = np.ones(NUM_CHANNELS, np.float32)
    if view == "lateral":
        for i in LUNG_IDX:
            valid[i] = 0.0
    return valid


def read_image(file_name):
    img = cv2.imread(os.path.join(IMAGES_ROOT, file_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(file_name)
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return img


# ---------------------------------------------------------------------------
# Augmentation (radiography-appropriate: no horizontal flip - situs matters)
# ---------------------------------------------------------------------------
def augment(img, mask, rng):
    ang = rng.uniform(-7, 7)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.04, 0.04) * IMG_SIZE
    ty = rng.uniform(-0.04, 0.04) * IMG_SIZE
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), ang, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.stack([cv2.warpAffine(c, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in mask])
    # acquisition variation: exposure / contrast / gamma
    img = img.astype(np.float32) / 255.0
    img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.06, 0.06)
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.rand() < 0.2:
        img = img + rng.randn(*img.shape).astype(np.float32) * rng.uniform(0.005, 0.02)
    return np.clip(img, 0, 1), mask


class PaxrayCardiacDataset(Dataset):
    """gpu_aug=True: hand raw uint8 tensors to the GPU and warp there (the box's
    CPUs are shared, the GPUs are not)."""

    def __init__(self, records, train=True, gpu_aug=False):
        self.records = records
        self.train = train
        self.gpu_aug = gpu_aug

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = read_image(r["file_name"])
        mask = load_mask(r)
        view = 1.0 if r["view"] == "lateral" else 0.0
        if self.gpu_aug:
            return (torch.from_numpy(img),                       # uint8 (H, W)
                    torch.from_numpy(mask),                      # uint8 (C, H, W)
                    torch.tensor(view),
                    torch.from_numpy(channel_valid(r["view"])),
                    i)
        if self.train:
            img, mask = augment(img, mask, np.random)
        else:
            img = img.astype(np.float32) / 255.0
        # channel 0 = radiograph, channel 1 = view indicator (frontal / lateral)
        x = np.stack([img, np.full_like(img, view)])
        return (torch.from_numpy(x),
                torch.from_numpy(mask.astype(np.float32)),
                torch.from_numpy(channel_valid(r["view"])),
                i)


def gpu_batch(img, mask, view, train, generator=None):
    """Move a raw uint8 batch to model input + target, augmenting on the GPU."""
    B, H, W = img.shape
    x = img.float().unsqueeze(1) / 255.0
    y = mask.float()
    if train:
        dev = x.device
        ang = (torch.rand(B, device=dev, generator=generator) * 14 - 7) * np.pi / 180
        scale = 1.0 / (0.92 + torch.rand(B, device=dev, generator=generator) * 0.16)
        tx = (torch.rand(B, device=dev, generator=generator) * 0.08 - 0.04) * 2
        ty = (torch.rand(B, device=dev, generator=generator) * 0.08 - 0.04) * 2
        cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
        grid = F_affine_grid(theta, (B, 1, H, W))
        x = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                                            align_corners=False)
        y = torch.nn.functional.grid_sample(y, grid, mode="nearest", padding_mode="zeros",
                                            align_corners=False)
        g = lambda lo, hi: (torch.rand(B, 1, 1, 1, device=dev, generator=generator) * (hi - lo) + lo)
        x = torch.clamp(x * g(0.9, 1.1) + g(-0.06, 0.06), 0, 1) ** g(0.8, 1.25)
        noise = (torch.rand(B, 1, 1, 1, device=dev, generator=generator) < 0.2).float()
        x = torch.clamp(x + noise * torch.randn_like(x) * 0.012, 0, 1)
    v = view.to(x.device).view(-1, 1, 1, 1).expand(-1, 1, H, W)
    return torch.cat([x, v], 1), y


def F_affine_grid(theta, size):
    return torch.nn.functional.affine_grid(theta, size, align_corners=False)


def build_loaders(batch_size=8, workers=8, val_per_view=300, seed=1234, gpu_aug=False,
                  rank=0, world=1):
    recs = load_records("train_val")
    train_recs, val_recs = split_train_val(recs, 0.2, seed)
    val_recs = fixed_val_subset(val_recs, val_per_view)
    train_ds = PaxrayCardiacDataset(train_recs, train=True, gpu_aug=gpu_aug)
    val_ds = PaxrayCardiacDataset(val_recs, train=False)
    sampler = None
    if world > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=workers, pin_memory=True, drop_last=True,
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
    val_workers = max(2, workers // 2) if rank == 0 else 0     # only rank 0 evaluates
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=val_workers,
        pin_memory=True, persistent_workers=val_workers > 0)
    return train_loader, val_loader, train_recs, val_recs


if __name__ == "__main__":
    for split in ("train_val", "test"):
        recs = load_records(split)
        print(split, len(recs), "images,", len({r['patient'] for r in recs}), "patients")
    tr, va = split_train_val(load_records("train_val"))
    print("train", len(tr), "val", len(va),
          "patient overlap", len({r['patient'] for r in tr} & {r['patient'] for r in va}))
    m = build_mask(tr[0])
    print(tr[0]["file_name"], m.shape, "areas:",
          {CHANNEL_NAMES[i]: int(m[i].sum()) for i in range(NUM_CHANNELS)})
