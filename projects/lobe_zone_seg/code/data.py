"""
data.py — PAXRay++ frontal chest X-ray: lung LOBE & ZONE segmentation (teaching tool).

Task: multi-label (overlapping) segmentation on frontal CXR of
  - 5 anatomical lobes: RUL, RML, RLL, LUL, LLL
  - 6 radiographic zones (3 per side): upper (apical+upper merged), mid, base
  - 3 lung-context channels: whole lung, right lung, left lung  (auxiliary)

Masks are the PAXRay++ (159, 512, 512) bool stack, read on demand from labels.zip.
Multi-label => sigmoid + BCE/Dice, NOT argmax (structures overlap / are hierarchical).

Only train.py and data.py are edited during experiments. prepare.py is fixed.
"""
import io
import os
import json
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from prepare import DATA_DIR, EXT_DATA_DIR

# ---------------------------------------------------------------------------
# Paths (from fixed prepare.py)
# ---------------------------------------------------------------------------
ROOT = DATA_DIR["chest x-ray (anatomy segmentation)"]           # .../pax_ray++
IMAGES_ROOT = os.path.join(ROOT, "paxray_images_unfiltered", "images_patlas")
ZIP_PATH = os.path.join(ROOT, "labels.zip")
TRAINVAL_JSON = os.path.join(ROOT, "paxray_train_val.json")
TEST_JSON = os.path.join(ROOT, "paxray_test.json")

IMG_SIZE = 512          # native resolution — do not downsample CXR anatomy
VAL_FRACTION = 0.20     # 20% of train_val -> validation (test is untouched)
SPLIT_SEED = 1337

# Derived, compact mask cache (bit-packed 14 target channels). ~448 KB/sample vs
# 40 MB per 159-channel .npy in labels.zip -> removes the I/O bottleneck. Written
# to EXT_DATA_DIR (NEVER the original dataset).
CACHE_DIR = os.path.join(EXT_DATA_DIR, "paxray_lobezone_cache")

# ---------------------------------------------------------------------------
# Target channel specification.
# Each entry: (short_name, group, [source category_ids to OR together])
# Order matters: primary metric is computed over the lobe+zone channels only
# (the first N_PRIMARY entries), lung-context channels are auxiliary/logged.
# ---------------------------------------------------------------------------
TARGET_SPEC = [
    # --- anatomical lobes (5) ---
    ("RUL", "lobe", [153]),      # lung upper lobe right
    ("RML", "lobe", [152]),      # lung middle lobe right
    ("RLL", "lobe", [151]),      # lung lower lobe right
    ("LUL", "lobe", [150]),      # lung upper lobe left
    ("LLL", "lobe", [149]),      # lung lower lobe left
    # --- radiographic zones (6): apical merged into upper ---
    ("R_upper", "zone", [141, 144]),   # right upper zone + right apical
    ("R_mid",   "zone", [142]),
    ("R_base",  "zone", [143]),
    ("L_upper", "zone", [145, 148]),   # left upper zone + left apical
    ("L_mid",   "zone", [146]),
    ("L_base",  "zone", [147]),
    # --- lung context (3): auxiliary, not in primary metric ---
    ("lung",       "context", [134]),
    ("right_lung", "context", [135]),
    ("left_lung",  "context", [136]),
]

CLASS_NAMES = [s[0] for s in TARGET_SPEC]
CLASS_GROUP = [s[1] for s in TARGET_SPEC]
N_CLASSES = len(TARGET_SPEC)                                  # 14
PRIMARY_IDX = [i for i, s in enumerate(TARGET_SPEC) if s[1] in ("lobe", "zone")]  # 0..10 (11)
LOBE_IDX = [i for i, s in enumerate(TARGET_SPEC) if s[1] == "lobe"]
ZONE_IDX = [i for i, s in enumerate(TARGET_SPEC) if s[1] == "zone"]
CONTEXT_IDX = [i for i, s in enumerate(TARGET_SPEC) if s[1] == "context"]


# ---------------------------------------------------------------------------
# On-demand mask reader (do NOT unzip the whole 615G archive)
# ---------------------------------------------------------------------------
class PaxrayLabelZip:
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self._z = None  # lazy per-worker handle (fork-safe)

    @property
    def z(self):
        if self._z is None:
            self._z = zipfile.ZipFile(self.zip_path)
        return self._z

    def get_stack(self, stem):
        with self.z.open(f"labels/{stem}.npy") as f:
            arr = np.load(io.BytesIO(f.read()))   # (159, 512, 512) bool
        return arr

    def _build_targets(self, stem):
        arr = self.get_stack(stem)
        out = np.zeros((N_CLASSES, arr.shape[1], arr.shape[2]), dtype=bool)
        for i, (_, _, srcs) in enumerate(TARGET_SPEC):
            m = arr[srcs[0]]
            for s in srcs[1:]:
                m = m | arr[s]
            out[i] = m
        return out

    def get_targets(self, stem):
        """Return (N_CLASSES, H, W) float32 mask, applying the OR-merges.

        Uses the compact bit-packed cache when present; otherwise builds it from
        labels.zip and writes the cache for next time.
        """
        cpath = os.path.join(CACHE_DIR, stem + ".npy")
        if os.path.exists(cpath):
            packed = np.load(cpath)                       # uint8 packed bits
            bits = np.unpackbits(packed)[: N_CLASSES * IMG_SIZE * IMG_SIZE]
            return bits.reshape(N_CLASSES, IMG_SIZE, IMG_SIZE).astype(np.float32)
        out = self._build_targets(stem)
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = cpath + f".tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                np.save(f, np.packbits(out))   # explicit handle: no .npy munging
            os.replace(tmp, cpath)
        except OSError:
            pass
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Split logic: study-level, frontal-only. A "study stem" (filename without the
# _frontal/_lateral view + extension) groups views of the same study, so no
# study crosses train/val. Frontal-only => effectively image-level, but grouping
# is kept to be leakage-safe. Test split is provided and never touched here.
# ---------------------------------------------------------------------------
def _frontal_stems(json_path):
    with open(json_path) as f:
        meta = json.load(f)
    stems = []
    for im in meta["images"]:
        fn = im["file_name"]
        if fn.endswith("_frontal.png"):
            stems.append(fn[:-len(".png")])          # e.g. "COVID1920_473_frontal"
    return sorted(set(stems))


def _study_key(stem):
    # strip the trailing view token so frontal/lateral of one study share a key
    for v in ("_frontal", "_lateral"):
        if stem.endswith(v):
            return stem[: -len(v)]
    return stem


def make_splits():
    """Returns dict with 'train','val','test' lists of frontal stems."""
    trainval = _frontal_stems(TRAINVAL_JSON)
    test = _frontal_stems(TEST_JSON)

    # group by study, split groups (not images) to prevent leakage
    keys = sorted(set(_study_key(s) for s in trainval))
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(keys))
    n_val = int(round(len(keys) * VAL_FRACTION))
    val_keys = set(keys[i] for i in perm[:n_val])

    train, val = [], []
    for s in trainval:
        (val if _study_key(s) in val_keys else train).append(s)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# ---------------------------------------------------------------------------
# Augmentation — anatomy-preserving. NO horizontal flip (channels are side-
# specific: left/right lung/lobe/zone would swap). Small rotation/scale/shift
# and mild intensity jitter only, reflecting acquisition variation.
# ---------------------------------------------------------------------------
AUG = os.environ.get("AUG", "light")   # default | light | minimal  (run13: light best)

def _augment(img, mask, rng):
    H, W = img.shape
    if AUG == "minimal":
        ang = 0.0; scale = 1.0; tx = ty = 0.0    # intensity-only (below)
    elif AUG == "light":
        ang = rng.uniform(-5, 5); scale = rng.uniform(0.95, 1.05)
        tx = ty = 0.0                       # no translation jitter
    else:
        # geometric: small affine (rotate +-7 deg, scale 0.95-1.05, shift +-4%)
        ang = rng.uniform(-7, 7)
        scale = rng.uniform(0.95, 1.05)
        tx = rng.uniform(-0.04, 0.04) * W
        ty = rng.uniform(-0.04, 0.04) * H
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    # warp all mask channels (nearest to keep binary)
    mask = np.stack([
        cv2.warpAffine(mask[c], M, (W, H), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for c in range(mask.shape[0])
    ], axis=0)
    # intensity: mild brightness/contrast jitter
    if rng.rand() < 0.5:
        alpha = rng.uniform(0.9, 1.1)   # contrast
        beta = rng.uniform(-0.05, 0.05)  # brightness
        img = np.clip(img * alpha + beta, 0.0, 1.0)
    return img, mask


class PaxrayLobeZoneDataset(Dataset):
    def __init__(self, stems, train=False):
        self.stems = stems
        self.train = train
        self.labels = PaxrayLabelZip(ZIP_PATH)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = cv2.imread(os.path.join(IMAGES_ROOT, stem + ".png"),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(stem + ".png")
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mask = self.labels.get_targets(stem)              # (N_CLASSES, H, W) float32

        if self.train:
            rng = np.random.RandomState()  # per-call randomness
            img, mask = _augment(img, mask, rng)

        img = torch.from_numpy(img).unsqueeze(0)          # (1, H, W)
        mask = torch.from_numpy(np.ascontiguousarray(mask))  # (N_CLASSES, H, W)
        return img, mask, stem


def build_datasets():
    sp = make_splits()
    return (
        PaxrayLobeZoneDataset(sp["train"], train=True),
        PaxrayLobeZoneDataset(sp["val"], train=False),
        PaxrayLobeZoneDataset(sp["test"], train=False),
        sp,
    )


if __name__ == "__main__":
    tr, va, te, sp = build_datasets()
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    print("classes:", CLASS_NAMES)
    print("primary idx:", PRIMARY_IDX)
    img, mask, stem = tr[0]
    print("sample", stem, "img", tuple(img.shape), "mask", tuple(mask.shape),
          "mask range", float(mask.min()), float(mask.max()))
