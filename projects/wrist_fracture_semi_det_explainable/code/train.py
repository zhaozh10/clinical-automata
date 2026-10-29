"""
Inherently-interpretable paediatric wrist-fracture detector.

Design rationale (why this is "explainable by construction" rather than
post-hoc): the network produces a single dense *evidence map* over the
radiograph.  The image-level "is there a fracture" probability is not a
separate classifier head -- it is a fixed, non-learned smooth-max pooling of
that same evidence map.  The image-level decision therefore cannot come from
anywhere other than the locations the map highlights, and those locations are
exactly what is shown to the clinician.  The same map is decoded into boxes.

Supervision is hybrid:
  * 555 fracture boxes on 663 strongly-annotated training images  -> dense
    Gaussian/centre supervision + box-size regression
  * every fracture-negative image                                 -> dense
    "no evidence anywhere" supervision
  * 8.4k fracture-positive images with image-level labels only    -> multiple
    -instance (pooled) supervision
"""
import os
import sys
import time
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import roc_auc_score

import data as D
from prepare import TIME_BUDGET_MINUTES

# ---------------------------------------------------------------------------
RUN_ITER = 12
PROJECT = "projects/20260910_09"
RUN_DIR = f"{PROJECT}/run{RUN_ITER}"

SEED = 0
BACKBONE = "resnet34"
BATCH = 24
LR = 3e-4
WD = 1e-4
WORKERS = 10
W_CLS = 1.0          # weak image-level (MIL) loss weight
W_BOX = 1.0          # dense IoU box loss weight
LSE_R = 6.0          # smooth-max temperature for the pooling readout
W_AUX = 0.5          # auxiliary "fracture mimic" channels
W_SPARSE = 1.0       # concentration penalty on image-level-only positives
SPARSE_R = 6.0       # cells around the retained peaks that stay unpenalised
SPARSE_AFTER = 300.0 # s of training before the penalty switches on
EVAL_EVERY_S = 120   # evaluate every N seconds of *training* time
EMA_DECAY = 0.999    # weight averaging -- the AP@0.5 estimate rests on only 139
                     # held-out boxes, so single-step weights are very jittery
PRIMARY = "fracture_ap50"
DIRECTION = "higher"
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda:0")


class ModelEMA:
    """Exponential moving average of the weights.

    Averaging removes most of the step-to-step jitter that the small
    strongly-annotated validation subset would otherwise turn into noise, and
    usually generalises better than any single iterate.
    """

    def __init__(self, model, decay=EMA_DECAY):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ===========================  model  =======================================
class EvidenceNet(nn.Module):
    def __init__(self, backbone=BACKBONE, ch=128):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=True, features_only=True,
                                     out_indices=(2, 3, 4), in_chans=1)
        cs = self.enc.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in cs])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1),
                                                   nn.GroupNorm(16, ch), nn.ReLU(inplace=True))
                                     for _ in cs])
        self.stem = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1),
                                  nn.GroupNorm(16, ch), nn.ReLU(inplace=True))
        self.heat = nn.Conv2d(ch, 1 + len(D.AUX_CLASSES), 1)   # ch 0 = fracture
        self.ltrb = nn.Conv2d(ch, 4, 1)           # dense distances to the 4 box sides
        nn.init.constant_(self.heat.bias, -4.6)   # prior p ~ 0.01
        nn.init.constant_(self.ltrb.bias, 1.9)    # exp(1.9) ~ 7 cells ~ median half-extent

    def _heads(self, x):
        f = self.enc(x)                                   # strides 8,16,32
        p = self.lat[-1](f[-1])
        for i in range(len(f) - 2, -1, -1):
            p = F.interpolate(p, size=f[i].shape[-2:], mode="nearest") + self.lat[i](f[i])
            p = self.smooth[i](p)
        p = self.stem(p)
        ltrb = torch.exp(self.ltrb(p).clamp(-4, 6))
        return self.heat(p), ltrb

    def forward(self, x):
        heat, ltrb = self._heads(x)
        heat = heat[:, :1]           # only the fracture channel is ever reported
        if not self.training:
            # mirror-image test-time averaging.  Left and right wrists both
            # occur, so the mirrored view is a legitimate second reading of the
            # same film; averaging the two evidence maps steadies both the
            # probability and the marked region.
            h2, l2 = self._heads(torch.flip(x, dims=[-1]))
            heat = (heat + torch.flip(h2[:, :1], dims=[-1])) / 2
            l2 = torch.flip(l2, dims=[-1])[:, [2, 1, 0, 3]]   # left<->right sides
            ltrb = (ltrb + l2) / 2
        return heat, ltrb[:, :2], ltrb[:, 2:]


def pool_logit(heat, r=LSE_R):
    """Smooth-max readout: the image score IS a pooling of the evidence map."""
    b = heat.shape[0]
    h = heat.reshape(b, -1)
    return (torch.logsumexp(r * h, dim=1) - math.log(h.shape[1])) / r


# ===========================  targets  =====================================
def _gauss_radius(h, w, min_overlap=0.7):
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / 2
    return max(1.0, min(r1, r2, r3))


def make_targets(bxs, has_box, label, aux=None):
    """Dense targets for ONE image on the stride-8 grid (runs in the workers).

    Box geometry is supervised at EVERY grid cell near the fracture centre, not
    just at the single centre cell: with only ~555 training outlines, one
    regression sample per fracture is far too few and the size head drifts.

    dense = 1 for images whose evidence map is fully known:
      * strongly annotated images (boxes given), and
      * every fracture-negative image (no evidence anywhere).
    """
    hm = np.zeros((D.OUT_H, D.OUT_W), np.float32)
    ltrb = np.zeros((4, D.OUT_H, D.OUT_W), np.float32)
    reg = np.zeros((D.OUT_H, D.OUT_W), np.float32)
    area_map = np.full((D.OUT_H, D.OUT_W), np.inf, np.float32)
    dense = 1.0 if (has_box > 0.5 or label < 0.5) else 0.0
    ys, xs = np.mgrid[0:D.OUT_H, 0:D.OUT_W].astype(np.float32)
    ys, xs = ys + 0.5, xs + 0.5
    for b in np.asarray(bxs, np.float32).reshape(-1, 4):
        x1, y1, x2, y2 = b / D.STRIDE
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < D.OUT_W and 0 <= iy < D.OUT_H):
            continue
        rad = int(_gauss_radius(math.ceil(h), math.ceil(w)))
        _draw_gauss(hm, ix, iy, rad)
        # centre sampling: cells inside the box and within CENTER_R of its centre
        m = ((xs > x1) & (xs < x2) & (ys > y1) & (ys < y2) &
             (np.abs(xs - cx) < CENTER_R) & (np.abs(ys - cy) < CENTER_R))
        m &= (w * h) < area_map            # smaller box wins on overlap
        if not m.any():
            m = np.zeros_like(m)
            m[iy, ix] = True
        area_map[m] = w * h
        ltrb[0][m] = (xs - x1)[m]
        ltrb[1][m] = (ys - y1)[m]
        ltrb[2][m] = (x2 - xs)[m]
        ltrb[3][m] = (y2 - ys)[m]
        reg[m] = 1.0
    hm_aux = np.zeros((len(D.AUX_CLASSES), D.OUT_H, D.OUT_W), np.float32)
    for a in np.asarray(aux if aux is not None else [], np.float32).reshape(-1, 5):
        c = int(a[0])
        x1, y1, x2, y2 = a[1:] / D.STRIDE
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        ix, iy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= ix < D.OUT_W and 0 <= iy < D.OUT_H:
            _draw_gauss(hm_aux[c], ix, iy, int(_gauss_radius(math.ceil(h), math.ceil(w))))
    return {"hm": torch.from_numpy(hm), "hm_aux": torch.from_numpy(hm_aux),
            "ltrb": torch.from_numpy(ltrb), "reg": torch.from_numpy(reg),
            "dense": torch.tensor(dense)}


CENTER_R = 3.0


def _draw_gauss(hm, cx, cy, rad):
    d = 2 * rad + 1
    sigma = d / 6.0
    y, x = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(np.float32).eps * g.max()] = 0
    H, W = hm.shape
    l, r = min(cx, rad), min(W - cx, rad + 1)
    t, b = min(cy, rad), min(H - cy, rad + 1)
    np.maximum(hm[cy - t:cy + b, cx - l:cx + r], g[rad - t:rad + b, rad - l:rad + r],
               out=hm[cy - t:cy + b, cx - l:cx + r])


def focal_loss(pred_logit, gt, dense_mask):
    """CenterNet penalty-reduced focal loss, restricted to fully-known images."""
    p = torch.sigmoid(pred_logit).clamp(1e-4, 1 - 1e-4)
    m = dense_mask.view(-1, 1, 1)
    pos = (gt == 1).float() * m
    neg = (gt < 1).float() * m
    pl = -((1 - p) ** 2) * torch.log(p) * pos
    nl = -((1 - gt) ** 4) * (p ** 2) * torch.log(1 - p) * neg
    npos = pos.sum().clamp(min=1.0)
    return (pl.sum() + nl.sum()) / npos


def concentration_loss(heat_logit, weak_pos, radius=SPARSE_R, topk=2):
    """Penalise diffuse evidence on films that carry only an image-level label.

    Those films are supervised by a max-pooling readout alone, so any number of
    extra hot-spots is free -- which is exactly where the false positives that
    depress AP come from.  Here everything outside the top-`topk` peaks is
    pushed down, so the map must commit to a small number of locations.  The
    peaks themselves are left alone: their position is never asserted, only
    their uniqueness.
    """
    B, H, W = heat_logit.shape
    with torch.no_grad():
        p = torch.sigmoid(heat_logit)
        nms = (F.max_pool2d(p, 3, 1, 1) == p).float() * p
        _, idx = torch.topk(nms.reshape(B, -1), topk, dim=1)
        py = (idx // W).float().unsqueeze(-1).unsqueeze(-1)
        px = (idx % W).float().unsqueeze(-1).unsqueeze(-1)
        yy = torch.arange(H, device=heat_logit.device).view(1, 1, H, 1)
        xx = torch.arange(W, device=heat_logit.device).view(1, 1, 1, W)
        near = ((yy - py) ** 2 + (xx - px) ** 2) <= radius ** 2      # B,topk,H,W
        far = (~near.any(1)) & weak_pos.view(-1, 1, 1)
    q = torch.sigmoid(heat_logit).clamp(1e-4, 1 - 1e-4)
    loss = -(q ** 2) * torch.log(1 - q) * far.float()
    return loss.sum() / weak_pos.sum().clamp(min=1.0)


def giou_loss(pred, tgt, mask):
    """1 - GIoU on the dense LTRB predictions.  AP@0.5 is an IoU criterion, so
    optimising IoU directly is better aligned than an L1 on width/height."""
    l, tp, r, b = pred.unbind(1)
    lg, tg, rg, bg = tgt.unbind(1)
    iw = torch.min(l, lg) + torch.min(r, rg)
    ih = torch.min(tp, tg) + torch.min(b, bg)
    inter = iw.clamp(min=0) * ih.clamp(min=0)
    ap = (l + r) * (tp + b)
    ag = (lg + rg) * (tg + bg)
    union = ap + ag - inter
    enc = (torch.max(l, lg) + torch.max(r, rg)) * (torch.max(tp, tg) + torch.max(b, bg))
    giou = inter / union.clamp(min=1e-6) - (enc - union) / enc.clamp(min=1e-6)
    loss = (1 - giou) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


# ===========================  decoding  ====================================
def decode(heat_logit, lt, rb, K=20):
    """Peaks of the evidence map -> boxes.  Pure read-out, no extra head."""
    p = torch.sigmoid(heat_logit)
    keep = (F.max_pool2d(p, 3, 1, 1) == p).float()
    p = (p * keep).reshape(p.shape[0], -1)
    K = min(K, p.shape[1])
    scores, idx = torch.topk(p, K, dim=1)
    ys = (idx // D.OUT_W).float() + 0.5
    xs = (idx % D.OUT_W).float() + 0.5
    B = heat_logit.shape[0]
    ltrb = torch.cat([lt, rb], 1).reshape(B, 4, -1)
    l = torch.gather(ltrb[:, 0], 1, idx)
    t_ = torch.gather(ltrb[:, 1], 1, idx)
    r = torch.gather(ltrb[:, 2], 1, idx)
    b = torch.gather(ltrb[:, 3], 1, idx)
    boxes = torch.stack([(xs - l), (ys - t_), (xs + r), (ys + b)], -1) * D.STRIDE
    return boxes, scores, xs * D.STRIDE, ys * D.STRIDE


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


# =====================  EVALUATION  (LOCKED CONTRACT)  =====================
# Must not change during the experiment loop.
@torch.no_grad()
def evaluate(model, loader, recs, collect=False):
    model.eval()
    img_scores, img_labels = [], []
    dets = []          # (img_i, score, box) over strongly annotated images
    n_gt = 0
    n_boxed_img = 0
    hits, hit_den = 0, 0
    per_sample = []
    for batch in loader:
        x = batch["img"].to(dev, non_blocking=True).float()
        hl, wh, off = model(x)
        s = torch.sigmoid(pool_logit(hl)).cpu().numpy()
        boxes, scores, px, py = decode(hl, wh, off)
        boxes = boxes.cpu().numpy(); scores = scores.cpu().numpy()
        px = px.cpu().numpy(); py = py.cpu().numpy()
        hmax = torch.sigmoid(hl).reshape(hl.shape[0], -1).argmax(1).cpu().numpy()
        for k, gi in enumerate(batch["idx"].tolist()):
            r = recs[gi]
            img_scores.append(float(s[k])); img_labels.append(int(r["label"]))
            gt = np.array(r["boxes"], np.float32).reshape(-1, 4)
            if collect:
                per_sample.append({"stem": r["stem"], "score": float(s[k]),
                                   "boxes": boxes[k].tolist(), "scores": scores[k].tolist()})
            if r["has_box"]:
                n_boxed_img += 1
                n_gt += len(gt)
                for j in range(boxes.shape[1]):
                    if scores[k, j] > 0.01:
                        dets.append((n_boxed_img - 1, float(scores[k, j]), boxes[k, j]))
                if len(gt):
                    hit_den += 1
                    ax = (hmax[k] % D.OUT_W) * D.STRIDE + D.STRIDE / 2
                    ay = (hmax[k] // D.OUT_W) * D.STRIDE + D.STRIDE / 2
                    inside = ((gt[:, 0] <= ax) & (ax <= gt[:, 2]) &
                              (gt[:, 1] <= ay) & (ay <= gt[:, 3])).any()
                    hits += int(inside)

    # --- image-level (all validation images) ---
    y = np.array(img_labels); s = np.array(img_scores)
    auc = float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")
    ths = np.unique(np.round(s, 4))
    best_j, best_t = -1, 0.5
    for t in ths:
        pr = s >= t
        tp = int((pr & (y == 1)).sum()); fn = int((~pr & (y == 1)).sum())
        tn = int((~pr & (y == 0)).sum()); fp = int((pr & (y == 0)).sum())
        se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
        if se + sp - 1 > best_j:
            best_j, best_t = se + sp - 1, float(t)
    pr = s >= best_t
    tp = int((pr & (y == 1)).sum()); fn = int((~pr & (y == 1)).sum())
    tn = int((~pr & (y == 0)).sum()); fp = int((pr & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    # --- box-level AP@0.5 on the strongly annotated subset ---
    gt_by_img = {}
    ii = 0
    for r in recs:
        if r["has_box"]:
            gt_by_img[ii] = np.array(r["boxes"], np.float32).reshape(-1, 4)
            ii += 1
    dets.sort(key=lambda d: -d[1])
    used = {i: np.zeros(len(g), bool) for i, g in gt_by_img.items()}
    tps, fps = [], []
    for gi, sc, bx in dets:
        g = gt_by_img.get(gi, np.zeros((0, 4), np.float32))
        ok = 0
        if len(g):
            m = iou_mat(bx[None], g)[0]
            j = int(m.argmax())
            if m[j] >= 0.5 and not used[gi][j]:
                used[gi][j] = True
                ok = 1
        tps.append(ok); fps.append(1 - ok)
    tps = np.cumsum(tps); fps = np.cumsum(fps)
    if n_gt > 0 and len(tps):
        rec = tps / n_gt
        prec = tps / np.maximum(tps + fps, 1e-9)
        mrec = np.concatenate([[0.0], rec, [rec[-1]]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
        # FROC: detection sensitivity at 1 false mark per image
        fpi = fps / max(n_boxed_img, 1)
        j = int(np.searchsorted(fpi, 1.0, side="right")) - 1
        froc = float(rec[j]) if j >= 0 else 0.0
    else:
        ap, froc = 0.0, 0.0

    out = {PRIMARY: ap,
           "metrics": {"auc": auc, "sensitivity": sens, "specificity": spec, "f1": f1,
                       "loc_hit_rate": hits / max(hit_den, 1), "froc_sens1fp": froc,
                       "threshold": best_t, "n_val": int(len(y)), "n_boxed": n_boxed_img,
                       "n_gt_boxes": n_gt}}
    if collect:
        out["per_sample"] = per_sample
    model.train()
    return out
# ===========================================================================


def main():
    os.makedirs(f"{RUN_DIR}/checkpoints", exist_ok=True)
    t_start = time.time()
    tr_recs, va_recs = D.build_index()
    print(f"train {len(tr_recs)} | val {len(va_recs)} "
          f"| train boxed {sum(r['has_box'] for r in tr_recs)} "
          f"| val boxed {sum(r['has_box'] for r in va_recs)}", flush=True)

    w = []
    for r in tr_recs:
        if r["has_box"] and len(r["boxes"]):
            w.append(8.0)
        elif r["has_box"]:
            w.append(2.0)
        elif r["label"] == 1:
            w.append(1.0)
        else:
            w.append(1.5)
    sampler = WeightedRandomSampler(torch.tensor(w), num_samples=len(tr_recs), replacement=True)

    dtr = DataLoader(D.WristDataset(tr_recs, True, make_targets), batch_size=BATCH, sampler=sampler,
                     num_workers=WORKERS, collate_fn=D.collate, pin_memory=True,
                     drop_last=True, persistent_workers=True, prefetch_factor=6)
    dva = DataLoader(D.WristDataset(va_recs, False), batch_size=32, shuffle=False,
                     num_workers=WORKERS, collate_fn=D.collate, pin_memory=True)

    model = EvidenceNet().to(dev)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema = ModelEMA(model)
    budget = TIME_BUDGET_MINUTES * 60

    def set_lr(step, train_s):
        """Time-based cosine decay: the budget is wall-clock, not epochs, so the
        schedule is driven by elapsed training seconds rather than step count."""
        warm = min(1.0, (step + 1) / 300)
        frac = min(train_s / budget, 1.0)
        scale = 0.5 * (1 + math.cos(math.pi * frac))
        lr = LR * warm * max(scale, 0.02)
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    best = -1e9 if DIRECTION == "higher" else 1e9
    best_metrics = None
    train_s = 0.0
    next_eval = EVAL_EVERY_S
    step = 0
    bce = nn.BCEWithLogitsLoss()
    model.train()
    stop = False
    while not stop:
        for batch in dtr:
            t0 = time.time()
            cur_lr = set_lr(step, train_s)
            x = batch["img"].to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            lab = batch["label"].to(dev, non_blocking=True)
            hm = batch["hm"].to(dev, non_blocking=True)
            hm_aux = batch["hm_aux"].to(dev, non_blocking=True)
            ltrb_t = batch["ltrb"].to(dev, non_blocking=True)
            reg = batch["reg"].to(dev, non_blocking=True)
            dense = batch["dense"].to(dev, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                heat_all, ltrb_p = model._heads(x)
                lt, rb = ltrb_p[:, :2], ltrb_p[:, 2:]
                hl = heat_all[:, :1]
            heat_all, hl, lt, rb = heat_all.float(), hl.float(), lt.float(), rb.float()
            l_heat = focal_loss(hl[:, 0], hm, dense)
            l_cls = bce(pool_logit(hl[:, 0]), lab)
            l_wh = giou_loss(torch.cat([lt, rb], 1), ltrb_t, reg)
            weak_pos = (lab > 0.5) & (dense < 0.5)
            if train_s >= SPARSE_AFTER and weak_pos.any():
                l_sp = concentration_loss(hl[:, 0], weak_pos)
            else:
                l_sp = hl.sum() * 0.0
            # mimics are only annotated on the strongly-labelled films
            hb = batch["has_box"].to(dev, non_blocking=True)
            l_aux = sum(focal_loss(heat_all[:, 1 + c], hm_aux[:, c], hb)
                        for c in range(len(D.AUX_CLASSES))) / len(D.AUX_CLASSES)
            loss = l_heat + W_CLS * l_cls + W_BOX * l_wh + W_SPARSE * l_sp + W_AUX * l_aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            torch.cuda.synchronize()
            train_s += time.time() - t0
            step += 1

            if step % 100 == 0:
                print(f"step {step} t={train_s:.0f}s lr {cur_lr:.2e} loss {loss.item():.4f} "
                      f"(heat {l_heat.item():.4f} cls {l_cls.item():.4f} wh {l_wh.item():.3f} "
                      f"sp {l_sp.item():.3f} aux {l_aux.item():.3f})", flush=True)

            if train_s >= next_eval or train_s >= budget:
                res = evaluate(ema.ema, dva, va_recs)
                cur = res[PRIMARY]
                mm = res["metrics"]
                print(f"[eval] t={train_s:.0f}s {PRIMARY}={cur:.4f} auc={mm['auc']:.4f} "
                      f"sens={mm['sensitivity']:.3f} spec={mm['specificity']:.3f} "
                      f"loc={mm['loc_hit_rate']:.3f} froc={mm['froc_sens1fp']:.3f}", flush=True)
                torch.save({"model": ema.ema.state_dict(), "step": step}, f"{RUN_DIR}/checkpoints/last.pth")
                better = cur > best if DIRECTION == "higher" else cur < best
                if better:
                    best, best_metrics = cur, res
                    torch.save({"model": ema.ema.state_dict(), "step": step, "metrics": res},
                               f"{RUN_DIR}/checkpoints/best.pth")
                next_eval = train_s + EVAL_EVERY_S
            if train_s >= budget:
                stop = True
                break

    peak = torch.cuda.max_memory_allocated() / 1e6
    total = time.time() - t_start
    mm = best_metrics["metrics"]
    print("\n---")
    print(f"{PRIMARY}:      {best:.4f}\n")
    print("metrics:")
    for k, v in mm.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\ntraining_seconds: {train_s:.1f}")
    print(f"total_seconds:    {total:.1f}")
    print(f"peak_vram_mb:     {peak:.1f}")
    with open(f"{RUN_DIR}/best_metrics.json", "w") as f:
        json.dump({PRIMARY: best, "metrics": mm, "training_seconds": train_s,
                   "total_seconds": total, "peak_vram_mb": peak}, f, indent=2)


if __name__ == "__main__":
    main()
