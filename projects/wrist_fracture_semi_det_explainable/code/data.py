"""
GRAZPEDWRI-DX paediatric wrist radiographs.

Task: fracture detection with an inherently-interpretable (evidence-map) model.

Supervision available in-scope:
  * image_labels.csv        -> image-level `fracture_visible` for all 16395 TRAIN images (weak)
  * annotations_voc_5pct.csv-> VOC boxes for 824 TRAIN images (strong) + the TEST images
                               (the TEST part is ONLY touched at finalization).
  * train_test_split.json   -> official patient-level train/test split.

Validation = 20% of the TRAIN split, patient-level, stratified so that the
strongly-annotated (boxed) images and the fracture-positive rate are preserved.
The TEST split is never loaded here except through `load_test_index()`, which is
called only by the finalization step.
"""
import os
import json
import csv
import hashlib
import collections
import multiprocessing as mp

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from prepare import DATA_DIR, EXT_DATA_DIR

cv2.setNumThreads(0)

ROOT = DATA_DIR["wrist x-ray"]
IMG_DIR = os.path.join(ROOT, "images")

# ---------------------------------------------------------------------------
# Geometry.  Wrist radiographs are portrait, ~450-700 x 650-1300 px.  Fracture
# boxes have a median extent of ~112 x 77 px, the smallest are ~42 x 31 px, so
# we letterbox into 640x896 (scale factor mostly 0.8-1.2, i.e. near-native
# resolution).  Aggressive downsampling would blur cortical-break detail.
# ---------------------------------------------------------------------------
IMG_W = 640
IMG_H = 896
STRIDE = 8            # output evidence-map stride
OUT_W = IMG_W // STRIDE
OUT_H = IMG_H // STRIDE

CACHE_DIR = os.path.join(EXT_DATA_DIR, f"gpwd_cache_{IMG_W}x{IMG_H}")
CACHE_IMG = os.path.join(CACHE_DIR, "images_u8.npy")
CACHE_META = os.path.join(CACHE_DIR, "meta.json")

FRACTURE_CLASS = "fracture"

# Fracture mimics that the dataset also outlines.  Laterality markers, casts and
# metalwork, periosteal reaction and the pronator sign are exactly the things a
# fracture detector tends to fire on by mistake, so they are given their own
# output channels as an auxiliary task.
AUX_CLASSES = ["text", "periostealreaction", "metal", "other"]
_AUX_MAP = {"text": 0, "periostealreaction": 1, "metal": 2,
            "boneanomaly": 3, "bonelesion": 3, "pronatorsign": 3, "softtissue": 3}


# ---------------------------------------------------------------------------
# Cache construction (one-time, ~4 min on 12 cores).  16-bit PNGs are slow to
# decode (~70 ms each); we letterbox once and keep a uint8 memmap.
# ---------------------------------------------------------------------------
def _letterbox_params(w, h):
    s = min(IMG_W / w, IMG_H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    px, py = (IMG_W - nw) // 2, (IMG_H - nh) // 2
    return s, nw, nh, px, py


_W_MM = None


def _w_init(path, n):
    global _W_MM
    _W_MM = np.memmap(path, dtype=np.uint8, mode="r+", shape=(n, IMG_H, IMG_W))


def _w_one(args):
    i, stem = args
    im = cv2.imread(os.path.join(IMG_DIR, stem + ".png"), cv2.IMREAD_UNCHANGED)
    if im is None:
        return i, 0, 0
    if im.ndim == 3:
        im = im[..., 0]
    h, w = im.shape
    im8 = (im >> 8).astype(np.uint8) if im.dtype == np.uint16 else im.astype(np.uint8)
    s, nw, nh, px, py = _letterbox_params(w, h)
    r = cv2.resize(im8, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((IMG_H, IMG_W), np.uint8)
    canvas[py:py + nh, px:px + nw] = r
    _W_MM[i] = canvas
    return i, w, h


def build_cache(stems):
    os.makedirs(CACHE_DIR, exist_ok=True)
    n = len(stems)
    mm = np.memmap(CACHE_IMG, dtype=np.uint8, mode="w+", shape=(n, IMG_H, IMG_W))
    del mm
    sizes = [None] * n
    with mp.Pool(12, initializer=_w_init, initargs=(CACHE_IMG, n)) as pool:
        for k, (i, w, h) in enumerate(pool.imap_unordered(_w_one, list(enumerate(stems)), chunksize=32)):
            sizes[i] = (w, h)
            if (k + 1) % 2000 == 0:
                print(f"  cached {k+1}/{n}", flush=True)
    meta = {"stems": stems, "sizes": sizes, "img_w": IMG_W, "img_h": IMG_H}
    with open(CACHE_META, "w") as f:
        json.dump(meta, f)
    return meta


def ensure_cache():
    """Cache every image in the dataset (train + test).  Labels are not touched."""
    all_stems = sorted(x[:-4] for x in os.listdir(IMG_DIR) if x.endswith(".png"))
    if os.path.exists(CACHE_META):
        meta = json.load(open(CACHE_META))
        if meta["stems"] == all_stems and os.path.getsize(CACHE_IMG) == len(all_stems) * IMG_H * IMG_W:
            return meta
    print(f"building image cache at {CACHE_DIR} ({len(all_stems)} images) ...", flush=True)
    return build_cache(all_stems)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
def _read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_boxes(stem_filter=None):
    """Fracture boxes and mimic boxes (original pixel coords) per filestem."""
    out = collections.defaultdict(list)
    aux = collections.defaultdict(list)
    seen = set()
    for r in _read_csv(os.path.join(ROOT, "annotations_voc_5pct.csv")):
        stem = r["filestem"]
        if stem_filter is not None and stem not in stem_filter:
            continue
        seen.add(stem)
        box = [float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])]
        if r["class_name"] == FRACTURE_CLASS:
            out[stem].append(box)
        elif r["class_name"] in _AUX_MAP:
            aux[stem].append([float(_AUX_MAP[r["class_name"]])] + box)
    return out, aux, seen


def _patient_of(stem):
    return stem.split("_")[0]


def build_index(seed=42, val_frac=0.2):
    """Patient-level 80/20 split of the official TRAIN split.

    Stratification key per patient: (has any strongly-boxed image,
    fraction of that patient's images that are fracture-positive, bucketed).
    """
    meta = ensure_cache()
    row_of = {s: i for i, s in enumerate(meta["stems"])}
    sizes = meta["sizes"]

    split = json.load(open(os.path.join(ROOT, "train_test_split.json")))
    train_stems = list(split["train"])
    train_set = set(train_stems)

    labels = {r["filestem"]: (int(r["fracture_visible"]), int(r["has_bbox"]))
              for r in _read_csv(os.path.join(ROOT, "image_labels.csv"))}
    boxes, aux_boxes, boxed_stems = load_boxes(train_set)

    per_pat = collections.defaultdict(list)
    for s in train_stems:
        per_pat[_patient_of(s)].append(s)

    strat = {}
    for p, ss in per_pat.items():
        has_box = any(s in boxed_stems for s in ss)
        pos = sum(labels[s][0] for s in ss) / len(ss)
        strat[p] = (has_box, 0 if pos == 0 else (1 if pos < 1 else 2))

    groups = collections.defaultdict(list)
    for p, k in strat.items():
        groups[k].append(p)
    rng = np.random.RandomState(seed)
    val_pat = set()
    for k, ps in groups.items():
        ps = sorted(ps)
        rng.shuffle(ps)
        n_val = int(round(len(ps) * val_frac))
        val_pat.update(ps[:n_val])

    def make(stems):
        recs = []
        for s in stems:
            fv, hb = labels[s]
            w, h = sizes[row_of[s]]
            sc, nw, nh, px, py = _letterbox_params(w, h)
            bb = [[b[0] * sc + px, b[1] * sc + py, b[2] * sc + px, b[3] * sc + py] for b in boxes.get(s, [])]
            ab = [[b[0], b[1] * sc + px, b[2] * sc + py, b[3] * sc + px, b[4] * sc + py]
                  for b in aux_boxes.get(s, [])]
            recs.append({"stem": s, "row": row_of[s], "label": fv,
                         "has_box": int(s in boxed_stems), "boxes": bb, "aux": ab})
        return recs

    tr = make([s for s in train_stems if _patient_of(s) not in val_pat])
    va = make([s for s in train_stems if _patient_of(s) in val_pat])
    return tr, va


def load_test_index():
    """TEST split.  Only called at finalization."""
    meta = ensure_cache()
    row_of = {s: i for i, s in enumerate(meta["stems"])}
    sizes = meta["sizes"]
    split = json.load(open(os.path.join(ROOT, "train_test_split.json")))
    test_stems = list(split["test"])
    boxes, _aux, boxed = load_boxes(set(test_stems))
    recs = []
    for s in test_stems:
        w, h = sizes[row_of[s]]
        sc, nw, nh, px, py = _letterbox_params(w, h)
        bb = [[b[0] * sc + px, b[1] * sc + py, b[2] * sc + px, b[3] * sc + py] for b in boxes.get(s, [])]
        recs.append({"stem": s, "row": row_of[s], "label": int(len(bb) > 0),
                     "has_box": 1, "boxes": bb, "aux": [], "scale": sc, "pad": (px, py),
                     "orig": (w, h)})
    return recs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
MEAN, STD = 0.30, 0.22


class WristDataset(Dataset):
    """Returns the image plus the dense stride-8 targets.

    Targets are built inside the worker processes so the training loop never
    blocks on numpy work.
    """

    def __init__(self, recs, train=True, target_fn=None):
        self.recs = recs
        self.train = train
        self.target_fn = target_fn
        self.mm = None

    def __len__(self):
        return len(self.recs)

    def _img(self, row):
        if self.mm is None:
            n = len(json.load(open(CACHE_META))["stems"])
            self.mm = np.memmap(CACHE_IMG, dtype=np.uint8, mode="r", shape=(n, IMG_H, IMG_W))
        return np.asarray(self.mm[row])

    def __getitem__(self, i):
        r = self.recs[i]
        img = self._img(r["row"]).astype(np.float32) / 255.0
        boxes = np.array(r["boxes"], np.float32).reshape(-1, 4)
        aux = np.array(r.get("aux", []), np.float32).reshape(-1, 5)

        if self.train:
            n = len(boxes)
            allb = np.concatenate([np.concatenate([np.full((n, 1), -1.0, np.float32), boxes], 1),
                                   aux], 0) if (n or len(aux)) else np.zeros((0, 5), np.float32)
            img, allb = _augment(img, allb)
            boxes = allb[allb[:, 0] < 0][:, 1:]
            aux = allb[allb[:, 0] >= 0]

        img = (img - MEAN) / STD
        out = {
            "img": torch.from_numpy(np.ascontiguousarray(img))[None],
            "boxes": torch.from_numpy(boxes),
            "label": torch.tensor(float(r["label"])),
            "has_box": torch.tensor(float(r["has_box"])),
            "idx": i,
        }
        if self.target_fn is not None:
            out.update(self.target_fn(boxes, r["has_box"], r["label"], aux))
        return out


def _augment(img, boxes):
    """`boxes` is (N,5): class marker in column 0, xyxy in columns 1..4."""
    cls = boxes[:, 0].copy() if len(boxes) else np.zeros((0,), np.float32)
    boxes = boxes[:, 1:] if len(boxes) else np.zeros((0, 4), np.float32)
    # horizontal flip: left and right wrists both occur, so mirroring is
    # anatomically valid here.
    if np.random.rand() < 0.5:
        img = img[:, ::-1]
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = IMG_W - boxes[:, [2, 0]]

    # affine: rotation, magnification and positioning all vary between real
    # acquisitions, so wider jitter here is anatomically legitimate and is the
    # main defence against memorising the ~555 outlined fractures.
    ang = np.random.uniform(-15, 15)
    sc = np.random.uniform(0.82, 1.22)
    tx = np.random.uniform(-0.08, 0.08) * IMG_W
    ty = np.random.uniform(-0.08, 0.08) * IMG_H
    M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR, borderValue=0.0)
    if len(boxes):
        c = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [0, 3]], boxes[:, [2, 3]]], 1)
        c = c.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
        c = c.reshape(-1, 4, 2)
        boxes = np.concatenate([c.min(1), c.max(1)], 1).astype(np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, IMG_W - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, IMG_H - 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 8) & (boxes[:, 3] - boxes[:, 1] > 8)
        boxes, cls = boxes[keep], cls[keep]

    # acquisition-style intensity jitter
    img = img * np.random.uniform(0.88, 1.12) + np.random.uniform(-0.06, 0.06)
    if np.random.rand() < 0.25:
        img = np.clip(img, 0, 1) ** np.random.uniform(0.8, 1.25)
    img = np.clip(img, 0, 1).astype(np.float32)
    return img, np.concatenate([cls[:, None], boxes], 1) if len(boxes) else np.zeros((0, 5), np.float32)


TARGET_KEYS = ("hm", "hm_aux", "ltrb", "reg", "dense")


def collate(batch):
    out = {
        "img": torch.stack([b["img"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "has_box": torch.stack([b["has_box"] for b in batch]),
        "idx": torch.tensor([b["idx"] for b in batch]),
    }
    for k in TARGET_KEYS:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


if __name__ == "__main__":
    tr, va = build_index()
    def stats(name, recs):
        nb = sum(r["has_box"] for r in recs)
        npos = sum(r["label"] for r in recs)
        nfb = sum(len(r["boxes"]) for r in recs)
        pats = len({_patient_of(r["stem"]) for r in recs})
        print(f"{name}: {len(recs)} imgs / {pats} patients | fracture+ {npos} "
              f"| boxed {nb} | fracture boxes {nfb}")
    stats("train", tr)
    stats("val  ", va)
    ptr = {_patient_of(r["stem"]) for r in tr}
    pva = {_patient_of(r["stem"]) for r in va}
    print("patient overlap train/val:", len(ptr & pva))
