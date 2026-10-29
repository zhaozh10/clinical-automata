"""PENGWIN 2026 Task 1 — peripelvic fracture fragment instance segmentation.

Data layer: one-time preprocessing cache + patch dataset + full-volume loaders.

Cache build (run once, outside the training time budget):
    uv run data.py --build-cache [--workers 8]

Key dataset facts established by inspection (see also the notes in the repo history):
  * 340 train subjects: 170 "pelvic" (sacrum + left hip + right hip labelled)
    and 170 "femur" (one single fractured femur labelled).
  * Label values encode instance AND bone:  1-50 sacrum, 51-100 left hip,
    101-150 right hip, 151-200 femur.  0 = background.
  * 71 / 340 volumes are stored with a flipped (-1,-1,1) direction matrix, so
    raw array axes are NOT anatomically consistent -> everything is reoriented
    to LPS before caching (x index increases towards patient LEFT).
  * The two subsets carry CONTRADICTORY conventions for the same anatomy (the
    femoral head is background in pelvic cases, foreground in femur cases;
    the pelvis is background in femur cases).  The network therefore receives a
    `region` conditioning channel, and at inference the region is predicted by a
    small classifier on a coarse whole-volume view.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import EXT_DATA_DIR

# ---------------------------------------------------------------------------
# paths / constants
# ---------------------------------------------------------------------------
PENGWIN_ROOT = "/path/to/your/datasets/pengwin2026"
T12_ROOT = os.path.join(PENGWIN_ROOT, "task1_2")
SPLIT_JSON = os.path.join(PENGWIN_ROOT, "splits", "split_v1_task12.json")
SPLIT_CSV = os.path.join(PENGWIN_ROOT, "splits", "split_v1_task12.csv")

CACHE_DIR = os.path.join(EXT_DATA_DIR, "pengwin_t12_cache_1mm")
CACHE_SPACING = 1.0                      # mm, isotropic
CROP_HU = 200.0                          # bone threshold used for the (label-free) ROI crop
CROP_PAD = 16                            # voxels of margin around the ROI

# semantic classes
CLS_NAMES = ["background", "sacrum", "left_hip", "right_hip", "femur"]
N_CLS = 5
# instance value range -> semantic class id
VAL_RANGES = [(1, 50, 1), (51, 100, 2), (101, 150, 3), (151, 200, 4)]
# anatomy id -> first output instance value (challenge output encoding)
CLS_BASE = {1: 1, 2: 51, 3: 101, 4: 151}
MIN_FRAG_MM3 = 500.0                     # challenge rule: smaller fragments are omitted

# CT intensity normalisation (bone-oriented window, kept wide enough for soft tissue context)
HU_LO, HU_HI = -500.0, 1800.0
HU_MEAN, HU_STD = 300.0, 500.0

N_POINTS_CACHED = 4000                   # foreground / contact-surface sample points per case


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def get_splits():
    """-> dict(split -> list[sid]) and dict(sid -> dict(region, n_fragments))."""
    with open(SPLIT_JSON) as f:
        sp = json.load(f)
    info = {}
    with open(SPLIT_CSV) as f:
        hdr = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(hdr, line.strip().split(",")))
            info[r["subject_id"]] = {
                "region": r["region"],
                "n_fragments": int(r["n_fragments"]),
                "split": r["split"],
            }
    return {k: sp[k] for k in ("train", "val", "test")}, info


def subject_dir(sid):
    for part in (1, 2, 3, 4):
        d = os.path.join(T12_ROOT, f"PENGWIN26_task1_2_train_part{part}", sid)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(sid)


def sem_from_inst(lab):
    """instance label volume (uint8, values 0..200) -> semantic class map (uint8 0..4)."""
    sem = np.zeros_like(lab)
    for lo, hi, c in VAL_RANGES:
        sem[(lab >= lo) & (lab <= hi)] = c
    return sem


# ---------------------------------------------------------------------------
# cache build
# ---------------------------------------------------------------------------
def _contact_surface(lab):
    """Voxels adjacent to a different fragment: >=2 distinct non-zero ids in the
    3x3x3 neighbourhood.  Computed as (max != min) over non-zero neighbours."""
    import scipy.ndimage as ndi

    nz = lab > 0
    big = np.where(nz, lab.astype(np.int16), np.int16(0))
    mx = ndi.maximum_filter(big, size=3, mode="nearest")
    small = np.where(nz, lab.astype(np.int16), np.int16(255))
    mn = ndi.minimum_filter(small, size=3, mode="nearest")
    cs = (mx != mn) & (mx > 0) & (mn < 255)
    # thin skin of surrounding background is useful supervision too -> keep as is,
    # then dilate by one voxel so the target is ~2mm thick and learnable.
    cs = ndi.binary_dilation(cs, iterations=1)
    return cs.astype(np.uint8)


def _build_one(sid):
    import SimpleITK as sitk

    t0 = time.time()
    d = subject_dir(sid)
    img = sitk.ReadImage(os.path.join(d, "image.mha"))
    lab = sitk.ReadImage(os.path.join(d, "label.mha"))

    orig_meta = {
        "size": list(img.GetSize()),
        "spacing": list(img.GetSpacing()),
        "origin": list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
    }
    if list(lab.GetSize()) != list(img.GetSize()):
        raise RuntimeError(f"{sid}: image/label size mismatch")

    # 1) canonical anatomical orientation
    img = sitk.DICOMOrient(img, "LPS")
    lab = sitk.DICOMOrient(lab, "LPS")

    # 2) isotropic resample onto a shared reference grid
    osz = np.array(img.GetSize(), dtype=np.float64)
    osp = np.array(img.GetSpacing(), dtype=np.float64)
    nsz = np.maximum(np.ceil(osz * osp / CACHE_SPACING).astype(int), 1)
    ref = sitk.Image([int(v) for v in nsz], sitk.sitkFloat32)
    ref.SetSpacing((CACHE_SPACING,) * 3)
    ref.SetOrigin(img.GetOrigin())
    ref.SetDirection(img.GetDirection())
    img_r = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    lab_r = sitk.Resample(lab, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)

    a = sitk.GetArrayFromImage(img_r)              # (Z, Y, X)
    L = sitk.GetArrayFromImage(lab_r)

    # 3) label-free ROI crop (identical rule is used at inference time)
    bone = a > CROP_HU
    if not bone.any():
        bone = a > -200.0
    idx = np.nonzero(bone)
    lo = np.array([idx[i].min() for i in range(3)])
    hi = np.array([idx[i].max() + 1 for i in range(3)])
    if (L > 0).any():                               # safety: never crop away GT
        lidx = np.nonzero(L)
        lo = np.minimum(lo, [lidx[i].min() for i in range(3)])
        hi = np.maximum(hi, [lidx[i].max() + 1 for i in range(3)])
    lo = np.maximum(lo - CROP_PAD, 0)
    hi = np.minimum(hi + CROP_PAD, a.shape)
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    a, L = a[sl], L[sl]

    # 4) store
    np.clip(a, -1024.0, 3071.0, out=a)
    ai = a.astype(np.int16)
    cs = _contact_surface(L)

    fg = np.nonzero(L > 0)
    csi = np.nonzero(cs > 0)
    rng = np.random.default_rng(int(sid))

    def sub(pts, n):
        m = len(pts[0])
        if m == 0:
            return np.zeros((0, 3), np.int16)
        sel = rng.choice(m, size=min(n, m), replace=False)
        return np.stack([pts[i][sel] for i in range(3)], 1).astype(np.int16)

    vals, cnts = np.unique(L[L > 0], return_counts=True)
    vol_mm3 = cnts * (CACHE_SPACING ** 3)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f"{sid}_img.npy"), ai)
    np.save(os.path.join(CACHE_DIR, f"{sid}_lab.npy"), L)
    np.save(os.path.join(CACHE_DIR, f"{sid}_cs.npy"), cs)
    np.savez(os.path.join(CACHE_DIR, f"{sid}_pts.npz"),
             fg=sub(fg, N_POINTS_CACHED), cs=sub(csi, N_POINTS_CACHED))
    meta = {
        "sid": sid,
        "shape": list(ai.shape),
        "crop_lo": [int(v) for v in lo],
        "grid_origin": list(img_r.GetOrigin()),
        "grid_spacing": list(img_r.GetSpacing()),
        "grid_direction": list(img_r.GetDirection()),
        "grid_size_full": [int(v) for v in nsz],
        "orig": orig_meta,
        "frag_values": [int(v) for v in vals],
        "frag_vol_mm3": [float(v) for v in vol_mm3],
        "n_frag": int(len(vals)),
        "n_cs_vox": int(cs.sum()),
    }
    with open(os.path.join(CACHE_DIR, f"{sid}_meta.json"), "w") as f:
        json.dump(meta, f)
    return sid, ai.shape, len(vals), round(time.time() - t0, 1)


def build_cache(workers=8, force=False):
    splits, info = get_splits()
    sids = sorted(set(splits["train"] + splits["val"] + splits["test"]))
    todo = [s for s in sids
            if force or not os.path.exists(os.path.join(CACHE_DIR, f"{s}_meta.json"))]
    print(f"cache dir {CACHE_DIR}\n{len(sids)} subjects, {len(todo)} to build", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    if todo:
        with Pool(workers) as p:
            for i, r in enumerate(p.imap_unordered(_build_one, todo)):
                if i % 10 == 0 or i == len(todo) - 1:
                    print(f"[{i+1}/{len(todo)}] {r} elapsed {time.time()-t0:.0f}s", flush=True)
    # summary
    shapes, nf = [], []
    for s in sids:
        m = json.load(open(os.path.join(CACHE_DIR, f"{s}_meta.json")))
        shapes.append(m["shape"]); nf.append(m["n_frag"])
    shapes = np.array(shapes)
    print(f"done in {time.time()-t0:.0f}s")
    print("cached shape  med", np.median(shapes, 0), " max", shapes.max(0))
    print("Mvox          med %.0f  max %.0f" % (np.median(shapes.prod(1)) / 1e6,
                                                shapes.prod(1).max() / 1e6))
    print("fragments     total", sum(nf))
    sz = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR))
    print("cache size    %.1f GB" % (sz / 1e9))


# ---------------------------------------------------------------------------
# runtime accessors
# ---------------------------------------------------------------------------
class Case:
    """Memory-mapped access to one cached subject."""

    __slots__ = ("sid", "meta", "region", "shape", "_img", "_lab", "_cs", "pts")

    def __init__(self, sid, region):
        self.sid = sid
        self.region = region
        self.meta = json.load(open(os.path.join(CACHE_DIR, f"{sid}_meta.json")))
        self.shape = tuple(self.meta["shape"])
        self._img = self._lab = self._cs = None
        p = np.load(os.path.join(CACHE_DIR, f"{sid}_pts.npz"))
        self.pts = {"fg": p["fg"], "cs": p["cs"]}

    def _mm(self, name):
        return np.load(os.path.join(CACHE_DIR, f"{self.sid}_{name}.npy"), mmap_mode="r")

    @property
    def img(self):
        if self._img is None:
            self._img = self._mm("img")
        return self._img

    @property
    def lab(self):
        if self._lab is None:
            self._lab = self._mm("lab")
        return self._lab

    @property
    def cs(self):
        if self._cs is None:
            self._cs = self._mm("cs")
        return self._cs


def load_cases(split):
    splits, info = get_splits()
    return [Case(s, info[s]["region"]) for s in splits[split]]


def normalize(x):
    """int16 HU array -> float32, bone-windowed and standardised."""
    x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_MEAN) / HU_STD


def coarse_view(case, out=48):
    """Fixed-size low-resolution whole-volume view, for the region classifier.

    Cached on disk: reading it back costs ~200 kB instead of streaming the whole
    volume through the page cache on every run.
    """
    import torch
    import torch.nn.functional as F

    fp = os.path.join(CACHE_DIR, f"{case.sid}_coarse{out}.npy")
    if os.path.exists(fp):
        return torch.from_numpy(np.load(fp).astype(np.float32))[None]
    a = np.asarray(case.img[::4, ::4, ::4])
    t = torch.from_numpy(normalize(a))[None, None]
    t = F.interpolate(t, size=(out, out, out), mode="trilinear", align_corners=False)
    np.save(fp, t[0, 0].numpy().astype(np.float16))
    return t[0]


# ---------------------------------------------------------------------------
# patch dataset
# ---------------------------------------------------------------------------
class PatchDataset:
    """Random patch sampler.  Returns raw int16 / uint8 crops; normalisation and
    augmentation happen on the GPU inside train.py."""

    def __init__(self, cases, patch=(128, 128, 128), steps=10 ** 9,
                 p_fg=0.34, p_cs=0.33, seed=0):
        self.cases = cases
        self.patch = np.array(patch)
        self.steps = steps
        self.p_fg, self.p_cs = p_fg, p_cs
        self.seed = seed

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1_000_003 + i) & 0xFFFFFFFF)
        c = self.cases[rng.integers(len(self.cases))]
        shp = np.array(c.shape)
        P = np.minimum(self.patch, shp)
        u = rng.random()
        if u < self.p_cs and len(c.pts["cs"]):
            ctr = c.pts["cs"][rng.integers(len(c.pts["cs"]))].astype(np.int64)
        elif u < self.p_cs + self.p_fg and len(c.pts["fg"]):
            ctr = c.pts["fg"][rng.integers(len(c.pts["fg"]))].astype(np.int64)
        else:
            ctr = np.array([rng.integers(shp[k]) for k in range(3)])
        lo = np.clip(ctr - P // 2, 0, shp - P)
        sl = tuple(slice(int(lo[k]), int(lo[k] + P[k])) for k in range(3))

        img = np.asarray(c.img[sl])
        lab = np.asarray(c.lab[sl])
        cs = np.asarray(c.cs[sl])
        # pad if the case is smaller than the patch
        if tuple(P) != tuple(self.patch):
            pw = [(0, int(self.patch[k] - P[k])) for k in range(3)]
            img = np.pad(img, pw, constant_values=-1024)
            lab = np.pad(lab, pw)
            cs = np.pad(cs, pw)
        return (img[None].astype(np.int16), lab[None], cs[None],
                np.int64(1 if c.region == "femur" else 0))


def collate(batch):
    import torch

    img = torch.from_numpy(np.stack([b[0] for b in batch]))
    lab = torch.from_numpy(np.stack([b[1] for b in batch]))
    cs = torch.from_numpy(np.stack([b[2] for b in batch]))
    reg = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return img, lab, cs, reg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--build-coarse", action="store_true")
    a = ap.parse_args()
    if a.build_cache:
        build_cache(a.workers, a.force)
    elif a.build_coarse:
        t0 = time.time()
        n = 0
        for sp in ("train", "val", "test"):
            for c in load_cases(sp):
                coarse_view(c)
                n += 1
        print(f"coarse views cached: {n} in {time.time()-t0:.0f}s")
    else:
        splits, info = get_splits()
        for k, v in splits.items():
            reg = [info[s]["region"] for s in v]
            print(k, len(v), "pelvic", reg.count("pelvic"), "femur", reg.count("femur"))
