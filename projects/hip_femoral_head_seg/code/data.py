"""
data.py — Pediatric hip AP X-ray femoral-head segmentation.

Dataset layout (see prepare.py DATA_DIR["hip x-ray"]):
    <train|test>/femoral_head_<left|right>/<case>/{image.jpg, mask.png, meta.json}
Both hips of a study share the SAME image file and the SAME case id; the split is
already case-disjoint between train/ and test/. We carve a case-level, cohort-
stratified validation set out of train/ (test/ is NEVER used for selection).

Cohorts (derived from meta.json src_rel), reproducing split_manifest counts exactly:
    reference, mat_removed, metal, morbus_perthes, normal, severe_dysplasia
EASY = {normal, reference};  HARD = the rest (the clinically hard subgroups).
"""
import os, glob, json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from prepare import DATA_DIR

ROOT = DATA_DIR["hip x-ray"]
SIDES = ["femoral_head_left", "femoral_head_right"]
COHORTS = ["reference", "mat_removed", "metal", "morbus_perthes", "normal", "severe_dysplasia"]
EASY_COHORTS = {"normal", "reference"}
HARD_COHORTS = set(COHORTS) - EASY_COHORTS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def cohort_from_src(src_rel: str) -> str:
    s = src_rel.lower()
    for c in ["mat_removed", "morbus_perthes", "severe_dysplasia", "normal", "metal"]:
        if c in s:
            return c
    return "reference"  # references/... and any residual


# --------------------------------------------------------------------------- #
# Index building + splitting
# --------------------------------------------------------------------------- #
def _build_cases(split: str):
    """Return dict case_id -> {cohort, image, left_mask|None, right_mask|None}."""
    cases = {}
    for side in SIDES:
        for d in sorted(glob.glob(os.path.join(ROOT, split, side, "*"))):
            if not os.path.isdir(d):
                continue
            case = os.path.basename(d)
            meta = json.load(open(os.path.join(d, "meta.json")))
            coh = cohort_from_src(meta["src_rel"])
            rec = cases.setdefault(case, {"case": case, "cohort": coh,
                                          "image": os.path.join(d, "image.jpg"),
                                          "left_mask": None, "right_mask": None})
            rec["cohort"] = coh  # both sides agree
            rec["image"] = os.path.join(d, "image.jpg")
            key = "left_mask" if side.endswith("left") else "right_mask"
            rec[key] = os.path.join(d, "mask.png")
    return cases


def make_splits(seed: int = 0, val_frac: float = 0.2):
    """Case-level, cohort-stratified train/val split of train/; test/ loaded as-is."""
    train_cases = _build_cases("train")
    test_cases = _build_cases("test")

    # stratified, deterministic
    rng = np.random.RandomState(seed)
    val_ids = []
    by_coh = {}
    for cid, rec in train_cases.items():
        by_coh.setdefault(rec["cohort"], []).append(cid)
    for coh, ids in by_coh.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_frac))
        val_ids.extend(ids[:n_val])
    val_ids = set(val_ids)

    tr = [rec for cid, rec in train_cases.items() if cid not in val_ids]
    va = [rec for cid, rec in train_cases.items() if cid in val_ids]
    te = list(test_cases.values())
    return tr, va, te


def head_samples(cases):
    """Expand cases -> list of per-head samples (only heads that have a mask)."""
    out = []
    for rec in cases:
        for side, key in [("left", "left_mask"), ("right", "right_mask")]:
            if rec[key] is not None:
                out.append({"case": rec["case"], "cohort": rec["cohort"],
                            "side": side, "image": rec["image"], "mask": rec[key]})
    return out


# --------------------------------------------------------------------------- #
# Image / mask IO
# --------------------------------------------------------------------------- #
def load_image(path, res):
    """Grayscale X-ray -> resized RGB float [H,W,3] in [0,1] (pre-normalization)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.repeat(img[..., None], 3, axis=2)


def load_mask(path, res=None):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if res is not None:
        m = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)


def normalize(img):
    return (img - IMAGENET_MEAN) / IMAGENET_STD


# --------------------------------------------------------------------------- #
# Augmentation (medical-appropriate, restrained). img in [0,1], mask in {0,1}.
# --------------------------------------------------------------------------- #
def augment(img, mask, rng, cfg):
    h, w = img.shape[:2]
    # geometric: rotation + scale + translation via single affine (img+mask together)
    ang = rng.uniform(-cfg["rot"], cfg["rot"])
    sc = rng.uniform(1 - cfg["scale"], 1 + cfg["scale"])
    tx = rng.uniform(-cfg["trans"], cfg["trans"]) * w
    ty = rng.uniform(-cfg["trans"], cfg["trans"]) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    # horizontal flip (pelvis is bilaterally symmetric -> anatomically valid; mask flips too)
    if rng.rand() < cfg["hflip"]:
        img = img[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    # photometric: brightness/contrast + gamma
    if cfg["bc"] > 0:
        b = rng.uniform(-cfg["bc"], cfg["bc"])
        c = rng.uniform(1 - cfg["bc"], 1 + cfg["bc"])
        img = np.clip((img - 0.5) * c + 0.5 + b, 0, 1)
    if cfg["gamma"] > 0:
        g = rng.uniform(1 - cfg["gamma"], 1 + cfg["gamma"])
        img = np.clip(img, 1e-6, 1) ** g
    return img, mask


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
DEFAULT_AUG = dict(rot=10.0, scale=0.10, trans=0.05, hflip=0.5, bc=0.15, gamma=0.20)


class HeadTrainDataset(Dataset):
    """Per-head training samples: full image -> single binary head mask at `res`."""
    def __init__(self, samples, res, aug=DEFAULT_AUG, seed=0):
        self.samples = samples
        self.res = res
        self.aug = aug
        self.base_seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_image(s["image"], self.res)
        mask = load_mask(s["mask"], self.res)
        if self.aug is not None:
            rng = np.random.RandomState((self.base_seed * 1_000_003 + i * 7919 +
                                         int.from_bytes(os.urandom(2), "little")) % (2**31))
            img, mask = augment(img, mask, rng, self.aug)
        img = normalize(img).transpose(2, 0, 1).copy()
        return (torch.from_numpy(img).float(),
                torch.from_numpy(mask[None]).float())


class CaseTrainDataset(Dataset):
    """Bilateral training: full image -> 2-channel target.
    channel 0 = head in LEFT half of image (patient right / right_mask)
    channel 1 = head in RIGHT half of image (patient left / left_mask)
    presence[c]=0 marks a missing side (excluded from the loss).
    hflip mirrors the image AND swaps the two channels (keeps ch=image-side)."""
    def __init__(self, cases, res, aug=DEFAULT_AUG, seed=0):
        self.cases = cases
        self.res = res
        self.aug = aug
        self.base_seed = seed

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        rec = self.cases[i]
        img = load_image(rec["image"], self.res)
        mask = np.zeros((self.res, self.res, 2), np.float32)
        presence = np.zeros(2, np.float32)
        if rec["right_mask"] is not None:  # patient right -> image left half -> ch0
            mask[..., 0] = load_mask(rec["right_mask"], self.res); presence[0] = 1
        if rec["left_mask"] is not None:   # patient left -> image right half -> ch1
            mask[..., 1] = load_mask(rec["left_mask"], self.res); presence[1] = 1
        if self.aug is not None:
            rng = np.random.RandomState((self.base_seed * 1_000_003 + i * 7919 +
                                         int.from_bytes(os.urandom(2), "little")) % (2**31))
            img, mask, presence = augment_case(img, mask, presence, rng, self.aug)
        img = normalize(img).transpose(2, 0, 1).copy()
        return (torch.from_numpy(img).float(),
                torch.from_numpy(mask.transpose(2, 0, 1)).float(),
                torch.from_numpy(presence).float())


def augment_case(img, mask, presence, rng, cfg):
    """Geometric+photometric aug for bilateral 2-channel target (mask HxWx2)."""
    h, w = img.shape[:2]
    ang = rng.uniform(-cfg["rot"], cfg["rot"])
    sc = rng.uniform(1 - cfg["scale"], 1 + cfg["scale"])
    tx = rng.uniform(-cfg["trans"], cfg["trans"]) * w
    ty = rng.uniform(-cfg["trans"], cfg["trans"]) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 2] += tx; M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    if rng.rand() < cfg["hflip"]:
        img = img[:, ::-1].copy()
        mask = mask[:, ::-1, ::-1].copy()   # mirror + swap the two channels
        presence = presence[::-1].copy()
    if cfg["bc"] > 0:
        b = rng.uniform(-cfg["bc"], cfg["bc"]); c = rng.uniform(1 - cfg["bc"], 1 + cfg["bc"])
        img = np.clip((img - 0.5) * c + 0.5 + b, 0, 1)
    if cfg["gamma"] > 0:
        g = rng.uniform(1 - cfg["gamma"], 1 + cfg["gamma"])
        img = np.clip(img, 1e-6, 1) ** g
    return img, mask, presence


class HeadEvalDataset(Dataset):
    """Per-head eval samples: returns model input + NATIVE-res GT mask + meta.
    Native-res GT preserves true shape for boundary metrics."""
    def __init__(self, samples, res):
        self.samples = samples
        self.res = res

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_image(s["image"], self.res)
        img = normalize(img).transpose(2, 0, 1).copy()
        gt_native = load_mask(s["mask"], None)  # native resolution
        return {
            "image": torch.from_numpy(img).float(),
            "gt": gt_native,  # numpy HxW {0,1}
            "cohort": s["cohort"], "case": s["case"], "side": s["side"],
        }


def eval_collate(batch):
    imgs = torch.stack([b["image"] for b in batch])
    meta = [{k: b[k] for k in ("gt", "cohort", "case", "side")} for b in batch]
    return imgs, meta
