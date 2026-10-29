"""PENGWIN 2026 Task 1 — fracture fragment instance segmentation.

Two-headed 3D U-Net:  semantic head (bg / sacrum / left hip / right hip / femur)
+ contact-surface head (voxels on the interface between two fragments).
Instances are decoded by suppressing the contact surface, running connected
components, then re-growing the cores to fill the semantic mask.

A small region classifier (pelvic vs femur scan) provides the conditioning
channel at inference time, because the two subsets of the dataset use
contradictory labelling conventions for the same anatomy.

Run:   uv run train.py          (self-launches DDP across all visible GPUs)
"""

import json
import os
import sys
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import TIME_BUDGET_MINUTES
import data as D

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
PROJECT = "projects/20260912_15"
RUN_DIR = os.environ.get("RUN_DIR", "")

PATCH = (128, 128, 128)
BATCH_PER_GPU = 2
BASE_CH = 32
LR = 1e-3
WD = 1e-5
WARMUP = 200
CS_LOSS_W = 1.0
CS_POS_WEIGHT = 6.0

N_EVALS = int(os.environ.get("SMOKE_N_EVALS", "2"))                       # evaluations spread over the training budget
SW_STEP = 0.75                   # sliding-window step as a fraction of the patch
SW_BATCH = 4                     # sliding-window patches per forward pass
CS_TAU = 0.05                    # contact-surface probability threshold for splitting
HU_GATE = -10000.0               # carve seed cores below this attenuation (off by default)
# Watershed marker depth in mm, per bone.  Tuned offline on the validation set
# re-tuned on the run5 weights: the pelvis needs shallow markers (complex flat
# bone produces spurious deep distance maxima) while the femur needs deep ones.
WS_H_PELVIC = 1.0
WS_H_FEMUR = 3.0
WS_H = 0.0                       # sweep override; 0 means "use the per-bone values"
WS_W = 1.0                       # weight on the distance term of the watershed elevation
MARKER_MODE = os.environ.get("MARKER_MODE", "abs")   # "abs" | "hmax"
MIN_PRED_MM3 = 500.0             # match the challenge omission rule; tuned offline on val

AUG_AFFINE_P = float(os.environ.get("AUG_AFFINE_P", "0.0"))   # off by default (run6); used for ensemble diversity
AUG_ROT_DEG = 15.0
AUG_SCALE = 0.15

REGION_HEADS = os.environ.get("REGION_HEADS", "0") == "1"
DEEP_SUP = True
DS_WEIGHTS = (1.0, 0.5, 0.25)

SEED = int(os.environ.get("SEED", "20260912"))

# ---- evaluation contract (FIXED for the whole experiment loop) -------------
PRIMARY_METRIC = "frag_dice"
METRIC_DIRECTION = "higher"
SCORED_FRAG_MM3 = 500.0          # challenge guideline: smaller fragments are omitted
MAX_FRAG_PER_BONE = 50           # challenge encoding: 50 label values per bone


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.InstanceNorm3d(co, affine=True),
        nn.LeakyReLU(1e-2, inplace=True),
        nn.Conv3d(co, co, 3, padding=1, bias=False), nn.InstanceNorm3d(co, affine=True),
        nn.LeakyReLU(1e-2, inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_ch=2, base=BASE_CH, n_sem=D.N_CLS, chans=None):
        super().__init__()
        chans = chans or [base, base * 2, base * 4, base * 8, min(base * 10, 320)]
        self.enc = nn.ModuleList()
        ci = in_ch
        for c in chans:
            self.enc.append(conv_block(ci, c))
            ci = c
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(chans) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(chans[i], chans[i - 1], 2, stride=2))
            self.dec.append(conv_block(chans[i - 1] * 2, chans[i - 1]))
        # One output head per region.  The two subsets label the same anatomy by
        # opposite conventions, and their optimal decoding differs sharply, so
        # give them separate heads over a shared trunk.
        self.n_out = n_sem + 1
        self.head = nn.Conv3d(chans[0], 2 * self.n_out, 1) if REGION_HEADS \
            else nn.Conv3d(chans[0], self.n_out, 1)
        # deep supervision: extra heads on the two coarser decoder levels
        self.ds_heads = nn.ModuleList([nn.Conv3d(chans[i], n_sem + 1, 1)
                                       for i in (1, 2)]) if DEEP_SUP else None

    def forward(self, x, reg=None):
        skips = []
        for i, e in enumerate(self.enc):
            if i > 0:
                x = self.pool(x)
            x = e(x)
            skips.append(x)
        x = skips[-1]
        feats = []
        for j, (u, d) in enumerate(zip(self.up, self.dec)):
            x = u(x)
            s = skips[-2 - j]
            x = d(torch.cat([x, s], 1))
            feats.append(x)
        o = self.head(x)
        if REGION_HEADS:
            B = o.shape[0]
            o = o.view(B, 2, self.n_out, *o.shape[2:])
            if reg is None:
                reg = torch.zeros(B, dtype=torch.long, device=o.device)
            o = o[torch.arange(B, device=o.device), reg]
        if self.ds_heads is not None and self.training:
            # feats[-2] is at stride 2, feats[-3] at stride 4
            aux = [self.ds_heads[0](feats[-2]), self.ds_heads[1](feats[-3])]
            return o[:, : D.N_CLS], o[:, D.N_CLS:], aux
        return o[:, : D.N_CLS], o[:, D.N_CLS:]


class _Ensemble(nn.Module):
    """Average the semantic and contact-surface probabilities of several models."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, reg=None):
        sp = cp = None
        for m in self.models:
            global REGION_HEADS
            saved = REGION_HEADS
            REGION_HEADS = getattr(m, "_region_heads", False)
            try:
                s, c = m(x, reg)
            finally:
                REGION_HEADS = saved
            s = s.float().softmax(1)
            c = torch.sigmoid(c.float())
            sp = s if sp is None else sp + s
            cp = c if cp is None else cp + c
        n = len(self.models)
        # predict_volume applies softmax/sigmoid again, so hand back logits that
        # reproduce the averaged probabilities
        return (sp / n).clamp_min(1e-8).log(), ((cp / n).clamp(1e-6, 1 - 1e-6)
                                                .logit())


class RegionNet(nn.Module):
    """pelvic vs femur from a 48^3 coarse whole-volume view."""

    def __init__(self):
        super().__init__()
        ch = [1, 16, 32, 64, 64]
        layers = []
        for i in range(4):
            layers += [nn.Conv3d(ch[i], ch[i + 1], 3, padding=1, bias=False),
                       nn.InstanceNorm3d(ch[i + 1], affine=True),
                       nn.LeakyReLU(1e-2, inplace=True), nn.MaxPool3d(2)]
        self.f = nn.Sequential(*layers)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        h = self.f(x).mean((2, 3, 4))
        return self.fc(h)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def soft_dice_ce(logits, target, n_cls):
    ce = F.cross_entropy(logits.float(), target)
    p = logits.float().softmax(1)
    t = F.one_hot(target, n_cls).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    dice = (2 * inter + 1e-5) / (denom + 1e-5)
    return ce + (1.0 - dice[1:].mean())


def cs_loss(logit, target):
    logit = logit.float().squeeze(1)
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(
        logit, t, pos_weight=torch.tensor(CS_POS_WEIGHT, device=logit.device))
    p = torch.sigmoid(logit)
    inter = (p * t).sum()
    dice = (2 * inter + 1e-5) / (p.sum() + t.sum() + 1e-5)
    return bce + (1.0 - dice)


# ---------------------------------------------------------------------------
# GPU-side batch preparation + augmentation
# ---------------------------------------------------------------------------
LR_SWAP = torch.tensor([0, 1, 3, 2, 4])          # left hip <-> right hip on x-flip


def prep_batch(img, lab, cs, reg, device, train=True):
    img = img.to(device, non_blocking=True).float()
    lab = lab.to(device, non_blocking=True).long()
    cs = cs.to(device, non_blocking=True).float()
    reg = reg.to(device, non_blocking=True)

    sem = torch.zeros_like(lab)
    for lo, hi, c in D.VAL_RANGES:
        sem[(lab >= lo) & (lab <= hi)] = c
    sem = sem.squeeze(1)
    cs = cs.squeeze(1)

    img = img.clamp(D.HU_LO, D.HU_HI).sub_(D.HU_MEAN).div_(D.HU_STD)

    if train:
        B = img.shape[0]
        # left-right flip (x is the last axis, anatomically patient-left)
        f = torch.rand(B, device=device) < 0.5
        if f.any():
            idx = f.nonzero(as_tuple=True)[0]
            img[idx] = torch.flip(img[idx], dims=[4])
            sem[idx] = LR_SWAP.to(device)[torch.flip(sem[idx], dims=[3])]
            cs[idx] = torch.flip(cs[idx], dims=[3])
        # random rotation + isotropic scale (patient positioning varies, and
        # 238 training cases is thin for a 3-D task)
        r = torch.rand(B, device=device) < AUG_AFFINE_P
        if r.any():
            ang = (torch.rand(B, 3, device=device) * 2 - 1) * (AUG_ROT_DEG * math.pi / 180)
            ang = ang * r[:, None]
            sc = 1.0 + (torch.rand(B, device=device) * 2 - 1) * AUG_SCALE
            sc = torch.where(r, sc, torch.ones_like(sc))
            cz, sz = torch.cos(ang[:, 0]), torch.sin(ang[:, 0])
            cy, sy = torch.cos(ang[:, 1]), torch.sin(ang[:, 1])
            cx, sx = torch.cos(ang[:, 2]), torch.sin(ang[:, 2])
            Rz = torch.zeros(B, 3, 3, device=device)
            Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz; Rz[:, 2, 2] = 1
            Ry = torch.zeros(B, 3, 3, device=device)
            Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy; Ry[:, 1, 1] = 1
            Rx = torch.zeros(B, 3, 3, device=device)
            Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx; Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx; Rx[:, 0, 0] = 1
            M = (Rz @ Ry @ Rx) / sc[:, None, None]
            theta = torch.cat([M, torch.zeros(B, 3, 1, device=device)], 2)
            grid = F.affine_grid(theta, img.shape, align_corners=False)
            img = F.grid_sample(img, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
            semf = F.grid_sample(sem[:, None].float(), grid, mode="nearest",
                                 padding_mode="border", align_corners=False)
            sem = semf[:, 0].long()
            cs = F.grid_sample(cs[:, None], grid, mode="nearest",
                               padding_mode="border", align_corners=False)[:, 0]

        # intensity jitter
        g = 1.0 + 0.15 * (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1)
        b = 0.10 * (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1)
        img = img * g + b
        n = torch.rand(B, 1, 1, 1, 1, device=device) < 0.2
        img = img + n * torch.randn_like(img) * 0.05

    rch = reg.float().view(-1, 1, 1, 1, 1).expand(-1, 1, *img.shape[2:])
    x = torch.cat([img, rch], 1).contiguous(memory_format=torch.channels_last_3d)
    return x, sem, cs


# ---------------------------------------------------------------------------
# sliding-window inference
# ---------------------------------------------------------------------------
def _gaussian(patch, device, sigma_scale=0.125):
    grids = []
    for p in patch:
        c = (p - 1) / 2.0
        x = torch.arange(p, device=device, dtype=torch.float32)
        grids.append(torch.exp(-((x - c) ** 2) / (2 * (p * sigma_scale) ** 2)))
    g = grids[0][:, None, None] * grids[1][None, :, None] * grids[2][None, None, :]
    return (g / g.max()).clamp_min(1e-3)


@torch.no_grad()
def predict_volume(model, case, region, device, gw=None):
    """-> (sem_prob[5,Z,Y,X] fp16, cs_prob[Z,Y,X] fp16) on GPU."""
    shp = case.shape
    P = [min(PATCH[i], shp[i]) for i in range(3)]
    # the network pools 4x, so the tensor fed to it must be a multiple of 16;
    # a volume thinner than the patch is padded with air and cropped back.
    Pn = [int(math.ceil(v / 16) * 16) for v in P]
    if gw is None or tuple(gw.shape) != tuple(P):
        gw = _gaussian(P, device)
    step = [max(1, int(round(P[i] * SW_STEP))) for i in range(3)]
    starts = []
    for i in range(3):
        if shp[i] <= P[i]:
            starts.append([0])
        else:
            n = int(math.ceil((shp[i] - P[i]) / step[i])) + 1
            starts.append([int(round(k * (shp[i] - P[i]) / (n - 1))) for k in range(n)])

    acc = torch.zeros((D.N_CLS,) + tuple(shp), dtype=torch.float16, device=device)
    acc_cs = torch.zeros(shp, dtype=torch.float16, device=device)
    wsum = torch.zeros(shp, dtype=torch.float16, device=device)
    rch_val = 1.0 if region == "femur" else 0.0
    reg_idx = int(rch_val)

    locs = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    for b0 in range(0, len(locs), SW_BATCH):
        chunk = locs[b0:b0 + SW_BATCH]
        buf = torch.full((len(chunk), 2) + tuple(Pn), -1024.0,
                         dtype=torch.float32, device=device)
        for k, (z, y, x) in enumerate(chunk):
            sl = (slice(z, z + P[0]), slice(y, y + P[1]), slice(x, x + P[2]))
            raw = np.ascontiguousarray(case.img[sl])
            buf[k, 0, :P[0], :P[1], :P[2]] = torch.from_numpy(raw).to(
                device, non_blocking=True).float()
            buf[k, 1] = rch_val
        buf[:, 0] = buf[:, 0].clamp(D.HU_LO, D.HU_HI).sub_(D.HU_MEAN).div_(D.HU_STD)
        buf = buf.contiguous(memory_format=torch.channels_last_3d)
        rb = torch.full((len(chunk),), reg_idx, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_l, c_l = model(buf, rb)
        s_l = s_l[:, :, :P[0], :P[1], :P[2]]
        c_l = c_l[:, :, :P[0], :P[1], :P[2]]
        sp = s_l.float().softmax(1).half()
        cp = torch.sigmoid(c_l.float())[:, 0].half()
        for k, (z, y, x) in enumerate(chunk):
            sl = (slice(z, z + P[0]), slice(y, y + P[1]), slice(x, x + P[2]))
            acc[(slice(None),) + sl] += sp[k] * gw
            acc_cs[sl] += cp[k] * gw
            wsum[sl] += gw
    wsum.clamp_min_(1e-3)
    acc /= wsum
    acc_cs /= wsum
    return acc, acc_cs, gw


# ---------------------------------------------------------------------------
# instance decoding
# ---------------------------------------------------------------------------
def _grow_labels(lbl, mask, iters=12):
    """Dilate integer labels inside `mask` (torch int32 3-D tensors on GPU)."""
    cur = lbl.float()[None, None]
    m = mask[None, None]
    for _ in range(iters):
        nxt = F.max_pool3d(cur, 3, stride=1, padding=1)
        cur = torch.where((cur == 0) & m, nxt, cur)
        if bool(((cur == 0) & m).sum() == 0):
            break
    return cur[0, 0].to(torch.int32)


def _watershed_labels(mask_np, core_np, h_mm, cs_np=None, w_dist=WS_W,
                      voxel_mm=D.CACHE_SPACING):
    """Marker-based watershed inside `mask_np`.

    Connected components require the predicted fracture surface to close
    completely -- a single missed voxel merges two fragments, which is the
    dominant recall failure.  Seeding the watershed from the deep interior of
    each core (distance >= h) and flooding the distance map instead tolerates
    an incomplete surface.
    """
    import scipy.ndimage as ndi
    from skimage.segmentation import watershed

    d = ndi.distance_transform_edt(core_np, sampling=(voxel_mm,) * 3)
    if MARKER_MODE == "hmax":
        # An absolute depth threshold drops any fragment whose whole interior is
        # shallower than h, which is exactly how thin fragments get absorbed by
        # a thicker neighbour.  Regional maxima of prominence h keep one seed per
        # bump regardless of its absolute depth.
        from skimage.morphology import h_maxima
        markers, n = ndi.label(h_maxima(d, h_mm) > 0)
    else:
        markers, n = ndi.label(d >= h_mm)
    core_cc, ncore = ndi.label(core_np)
    if ncore == 0:
        return mask_np.astype(np.int32)
    # a fragment thinner than h has no deep interior and would vanish entirely,
    # so give every seed component that lost its marker one of its own
    present = set(int(v) for v in np.unique(core_cc[markers > 0]))
    for c in range(1, ncore + 1):
        if c not in present:
            n += 1
            markers[core_cc == c] = n
    if n == 0:
        markers, n = core_cc, ncore
    dm = ndi.distance_transform_edt(mask_np, sampling=(voxel_mm,) * 3)
    # Flooding a pure distance map puts the split at the geometric mid-surface
    # rather than on the predicted fracture; blend in the contact-surface
    # probability so the boundary follows the fracture where one was seen.
    elev = -dm / max(float(dm.max()), 1e-6)
    if cs_np is not None and w_dist < 1.0:
        elev = w_dist * elev + (1.0 - w_dist) * cs_np.astype(np.float32)
    return watershed(elev, markers, mask=mask_np).astype(np.int32)


def decode_instances(sem_prob, cs_prob, device, cs_tau=CS_TAU,
                     min_mm3=MIN_PRED_MM3, voxel_mm3=D.CACHE_SPACING ** 3,
                     img_hu=None, hu_gate=HU_GATE, ws_h=0.0, ws_w=WS_W):
    """-> int16 numpy instance map using the challenge output encoding."""
    import cc3d

    sem = sem_prob.argmax(0).to(torch.uint8)
    csb = cs_prob > cs_tau
    if img_hu is not None and hu_gate > -1000:
        # fracture gaps are filled with soft tissue / haematoma, so carve the
        # seed cores at low attenuation.  The network's mask tends to bridge
        # thin fracture lines, which is the dominant under-splitting failure.
        csb = csb | (img_hu < hu_gate)
    out = torch.zeros(sem.shape, dtype=torch.int32, device=device)
    for c in (1, 2, 3, 4):
        mask = sem == c
        nvox = int(mask.sum())
        if nvox == 0:
            continue
        core = mask & (~csb)
        # crop to the class bounding box: the CPU-side decode then costs a few
        # million voxels instead of the whole volume
        idx = torch.nonzero(mask)
        lo = idx.min(0).values.tolist()
        hi = (idx.max(0).values + 1).tolist()
        sl = tuple(slice(lo[i], hi[i]) for i in range(3))
        mask_np = mask[sl].cpu().numpy()
        core_np = core[sl].cpu().numpy()
        cs_np = cs_prob[sl].float().cpu().numpy()
        h = ws_h if ws_h > 0 else (WS_H_FEMUR if c == 4 else WS_H_PELVIC)
        if h > 0:
            cc = _watershed_labels(mask_np, core_np, h, cs_np, ws_w)
        else:
            cc = cc3d.connected_components(core_np, connectivity=26)
            if int(cc.max()) == 0:
                cc = mask_np.astype(np.int32)
        cc_full = np.zeros(tuple(mask.shape), np.int32)
        cc_full[sl] = cc
        lbl = torch.from_numpy(cc_full).to(device)
        grown = _grow_labels(lbl, mask, iters=12)
        grown = torch.where(mask, grown, torch.zeros_like(grown))
        # drop specks, renumber, apply challenge value offsets
        ids, cnts = torch.unique(grown[grown > 0], return_counts=True)
        big = (cnts.float() * voxel_mm3) >= min_mm3
        ids, cnts = ids[big], cnts[big]
        # the challenge encoding reserves 50 values per bone -> keep the 50
        # largest fragments; anything else is absorbed by its neighbours below.
        if len(ids) > MAX_FRAG_PER_BONE:
            sel = torch.argsort(cnts, descending=True)[:MAX_FRAG_PER_BONE]
            ids, cnts = ids[sel], cnts[sel]
        keep = ids
        if len(keep) == 0:
            allids, allc = torch.unique(grown[grown > 0], return_counts=True)
            if len(allids):
                keep = allids[allc.argmax()][None]
        remap = torch.zeros(int(grown.max()) + 1, dtype=torch.int32, device=device)
        base = D.CLS_BASE[c]
        for k, v in enumerate(keep.tolist()):
            remap[v] = base + k
        newl = remap[grown.clamp_min(0).long()]
        # specks that were dropped get absorbed into their neighbours
        dropped = mask & (newl == 0)
        if bool(dropped.any()) and len(keep) > 0:
            newl = _grow_labels(newl, mask, iters=6)
            newl = torch.where(mask, newl, torch.zeros_like(newl))
        out = torch.where(mask, newl, out)
    return out.to(torch.int16).cpu().numpy()


# ---------------------------------------------------------------------------
# metric  (FIXED — do not change during the experiment loop)
# ---------------------------------------------------------------------------
def _assd_from_coords(coords, gsel, psel, spacing):
    """Average symmetric surface distance for one matched pair, reconstructed
    from the sparse voxel list so cost scales with fragment size, not volume."""
    import scipy.ndimage as ndi

    sel = gsel | psel
    if not gsel.any() or not psel.any():
        return np.nan
    zc = [coords[i][sel] for i in range(3)]
    lo = [int(z.min()) - 2 for z in zc]
    shp = tuple(int(zc[i].max()) - lo[i] + 3 for i in range(3))
    g = np.zeros(shp, bool)
    p = np.zeros(shp, bool)
    idx = tuple(coords[i][sel] - lo[i] for i in range(3))
    g[idx] = gsel[sel]
    p[idx] = psel[sel]
    # ASSD is a secondary metric: evaluate it on a 2x coarser grid so that large
    # fragments stay affordable.  Fragments that vanish at 2 mm fall back to 1 mm.
    g2, p2 = g[::2, ::2, ::2], p[::2, ::2, ::2]
    if g2.any() and p2.any():
        g, p = g2, p2
        spacing = tuple(v * 2 for v in spacing)
    dg = ndi.distance_transform_edt(~g, sampling=spacing)
    dp = ndi.distance_transform_edt(~p, sampling=spacing)
    gs = g & ~ndi.binary_erosion(g)
    ps = p & ~ndi.binary_erosion(p)
    if not gs.any() or not ps.any():
        return np.nan
    return float((dp[gs].sum() + dg[ps].sum()) / (gs.sum() + ps.sum()))


def score_case(pred, gt, frag_vol_mm3, voxel_mm3=D.CACHE_SPACING ** 3):
    """Per-case fragment-level score; see PRIMARY_METRIC docs in the usage manual.

    Works on the sparse union of GT and prediction so that a degenerate
    prediction with thousands of fragments stays cheap to score.
    """
    from scipy.optimize import linear_sum_assignment

    sp = (D.CACHE_SPACING,) * 3
    nz = (gt > 0) | (pred > 0)
    coords = np.nonzero(nz)
    g_all = gt[nz].astype(np.int32)
    p_all = pred[nz].astype(np.int32)
    MAXV = 256
    np.clip(g_all, 0, MAXV - 1, out=g_all)
    np.clip(p_all, 0, MAXV - 1, out=p_all)
    gcnt_all = np.bincount(g_all, minlength=MAXV)
    pcnt_all = np.bincount(p_all, minlength=MAXV)
    sem_lut = np.zeros(MAXV, np.int8)
    for lo, hi, c in D.VAL_RANGES:
        sem_lut[lo:hi + 1] = c
    gcls = sem_lut[g_all]
    pcls = sem_lut[p_all]

    per_cls, sem_d, cnt_err, assds = {}, {}, [], []
    small_tot = small_hit = 0
    n_gt_scored = n_matched = n_fp_tot = n_pred_tot = 0
    matched_dices = []
    for lo, hi, c in D.VAL_RANGES:
        gt_ids = np.nonzero(gcnt_all[lo:hi + 1])[0] + lo
        pr_ids = np.nonzero(pcnt_all[lo:hi + 1])[0] + lo
        if not len(gt_ids) and not len(pr_ids):
            continue
        gin = gcls == c
        pin = pcls == c
        if len(gt_ids):
            inter = float((gin & pin).sum())
            sem_d[D.CLS_NAMES[c]] = 2 * inter / max(float(gin.sum() + pin.sum()), 1e-9)
        if not len(gt_ids):
            continue
        vols = gcnt_all[gt_ids].astype(np.float64) * voxel_mm3
        order = np.argsort(-vols)
        scored = gt_ids[order][vols[order] >= SCORED_FRAG_MM3]
        ignored = gt_ids[order][vols[order] < SCORED_FRAG_MM3]
        small_tot += len(ignored)
        if not len(scored):
            continue
        cnt_err.append(abs(len(pr_ids) - len(scored)))
        if not len(pr_ids):
            per_cls[D.CLS_NAMES[c]] = 0.0
            continue

        ids = np.concatenate([scored, ignored])
        gmap = np.zeros(MAXV, np.int64)
        gmap[ids] = np.arange(1, len(ids) + 1)
        pmap = np.zeros(MAXV, np.int64)
        pmap[pr_ids] = np.arange(1, len(pr_ids) + 1)
        gl = gmap[g_all]
        pl = pmap[p_all]
        ng, npd = len(ids), len(pr_ids)
        hist = np.bincount(gl * (npd + 1) + pl,
                           minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
        inter = hist[1:, 1:].astype(np.float64)
        gsz = hist[1:, :].sum(1)[:, None].astype(np.float64)
        psz_all = hist[:, 1:].sum(0).astype(np.float64)
        Dm = 2 * inter / np.maximum(gsz + psz_all[None, :], 1e-9)

        Ds = Dm[: len(scored)]
        r, cc = linear_sum_assignment(-Ds)
        matched, used = 0.0, set()
        for i, j in zip(r, cc):
            if Ds[i, j] > 0:
                matched += Ds[i, j]
                used.add(int(j))
                a = _assd_from_coords(coords, gl == (i + 1), pl == (j + 1), sp)
                if not np.isnan(a):
                    assds.append(a)
        if len(ignored):
            small_hit += int((Dm[len(scored):].max(1) > 0.1).sum())
        ign_sz = (hist[len(scored) + 1:, 1:].sum(0).astype(np.float64)
                  if len(ignored) else np.zeros(npd))
        frac_ign = np.divide(ign_sz, np.maximum(psz_all, 1e-9))
        fp = int(sum(1 for j in range(npd) if j not in used and frac_ign[j] < 0.5))
        n_gt_scored += len(scored)
        n_matched += len(used)
        n_fp_tot += fp
        n_pred_tot += npd
        matched_dices += [float(Ds[i, j]) for i, j in zip(r, cc) if Ds[i, j] > 0]
        per_cls[D.CLS_NAMES[c]] = matched / max(len(scored) + fp, 1)
    case_score = float(np.mean(list(per_cls.values()))) if per_cls else np.nan
    return {
        "frag_dice": case_score,
        "per_cls": per_cls,
        "sem_dice": sem_d,
        "count_mae": float(np.mean(cnt_err)) if cnt_err else np.nan,
        "assd": float(np.mean(assds)) if assds else np.nan,
        "small_tot": small_tot,
        "small_hit": small_hit,
        "n_gt_scored": n_gt_scored,
        "n_matched": n_matched,
        "n_fp": n_fp_tot,
        "n_pred_tot": n_pred_tot,
        "mean_matched_dice": float(np.mean(matched_dices)) if matched_dices else np.nan,
    }


@torch.no_grad()
def predict_regions(regnet, cases, device):
    """Region conditioning is fixed for the whole run -> predict it once."""
    regnet.eval()
    out = {}
    with torch.no_grad():
        for c in cases:
            cv = D.coarse_view(c).to(device)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = regnet(cv)
            out[c.sid] = "femur" if int(pr.argmax(1)) == 1 else "pelvic"
    return out


@torch.no_grad()
def evaluate(model, region_map, cases, device, rank, world, save_dir=None):
    model.eval()
    mine = cases[rank::world]
    rows, gw = [], None
    for case in mine:
        region = region_map[case.sid]
        sem, cs, gw = predict_volume(model, case, region, device, gw)
        hu = (torch.from_numpy(np.asarray(case.img)).to(device).float()
              if HU_GATE > -1000 else None)
        inst = decode_instances(sem, cs, device, img_hu=hu)
        del sem, cs, hu
        torch.cuda.empty_cache()
        gt = np.asarray(case.lab)
        s = score_case(inst, gt, case.meta["frag_vol_mm3"])
        s["sid"] = case.sid
        s["region"] = case.region
        s["region_pred"] = region
        s["n_gt"] = case.meta["n_frag"]
        s["n_pred"] = int(len(np.unique(inst)) - 1)
        rows.append(s)
        if save_dir:
            np.save(os.path.join(save_dir, f"{case.sid}_pred.npy"), inst)
    if world > 1:
        buf = [None] * world
        dist.all_gather_object(buf, rows)
        rows = [r for b in buf for r in b]
    model.train()
    return rows


def aggregate(rows):
    f = np.array([r["frag_dice"] for r in rows], float)
    pel = np.array([r["frag_dice"] for r in rows if r["region"] == "pelvic"], float)
    fem = np.array([r["frag_dice"] for r in rows if r["region"] == "femur"], float)
    sem = {}
    for r in rows:
        for k, v in r["sem_dice"].items():
            sem.setdefault(k, []).append(v)
    st = sum(r["small_tot"] for r in rows)
    sh = sum(r["small_hit"] for r in rows)
    m = {
        "frag_dice_pelvic": float(np.nanmean(pel)) if len(pel) else float("nan"),
        "frag_dice_femur": float(np.nanmean(fem)) if len(fem) else float("nan"),
        "count_mae": float(np.nanmean([r["count_mae"] for r in rows])),
        "assd_mm": float(np.nanmean([r["assd"] for r in rows])),
        "small_frag_recall": float(sh / st) if st else float("nan"),
        "region_acc": float(np.mean([r["region"] == r["region_pred"] for r in rows])),
    }
    for k in D.CLS_NAMES[1:]:
        m[f"sem_dice_{k}"] = float(np.nanmean(sem[k])) if k in sem else float("nan")
    return float(np.nanmean(f)), m


# ---------------------------------------------------------------------------
# region classifier training (cheap, ~1% of the budget)
# ---------------------------------------------------------------------------
def train_region(train_cases, val_cases, device, rank, steps=300):
    net = RegionNet().to(device)
    if rank == 0:
        print("[region] preparing coarse views", flush=True)
    X = torch.stack([D.coarse_view(c) for c in train_cases]).to(device)
    y = torch.tensor([1 if c.region == "femur" else 0 for c in train_cases], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    net.train()
    g = torch.Generator(device="cpu").manual_seed(SEED + rank)
    for i in range(steps):
        idx = torch.randint(0, len(X), (16,), generator=g).to(device)
        xb = X[idx]
        if random.random() < 0.5:
            xb = torch.flip(xb, dims=[4])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net(xb), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        Xv = torch.stack([D.coarse_view(c) for c in val_cases]).to(device)
        yv = torch.tensor([1 if c.region == "femur" else 0 for c in val_cases], device=device)
        acc = float((net(Xv).argmax(1) == yv).float().mean())
    if rank == 0:
        print(f"[region] val accuracy {acc:.4f}", flush=True)
    del X, Xv
    torch.cuda.empty_cache()
    return net, acc


# ---------------------------------------------------------------------------
# main worker
# ---------------------------------------------------------------------------
def worker(rank, world, run_dir, port):
    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)
    random.seed(SEED + rank)
    if world > 1:
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                                rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t_start = time.time()
    train_cases = D.load_cases("train")
    val_cases = D.load_cases("val")
    _vn = int(os.environ.get("SMOKE_VAL_N", "0"))      # smoke-test shortcut only
    if _vn:
        val_cases = val_cases[:_vn]
    if rank == 0:
        print(f"train {len(train_cases)} val {len(val_cases)} cases | world {world} "
              f"| patch {PATCH} | budget {TIME_BUDGET_MINUTES} min", flush=True)

    regnet, reg_acc = train_region(train_cases, val_cases, device, rank)
    region_map = predict_regions(regnet, val_cases, device)

    model = UNet3D().to(device).to(memory_format=torch.channels_last_3d)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[model] UNet3D params {n/1e6:.1f} M", flush=True)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    raw = model.module if world > 1 else model

    ds = D.PatchDataset(train_cases, PATCH, steps=10 ** 7, seed=SEED + rank * 7919)
    dl = DataLoader(ds, batch_size=BATCH_PER_GPU, num_workers=4, collate_fn=D.collate,
                    pin_memory=True, persistent_workers=True, prefetch_factor=3)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    budget = float(os.environ.get("SMOKE_BUDGET_SEC", TIME_BUDGET_MINUTES * 60.0))
    eval_at = [budget * (k + 1) / N_EVALS for k in range(N_EVALS)]
    train_sec, eval_sec = 0.0, 0.0
    best = -1e18
    best_rows = None
    step = 0
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # LR-schedule length estimate, measured after cuDNN autotuning has settled
    est_total = 12000
    t_warm = None
    loss_acc, t_log = 0.0, time.time()
    it = iter(dl)
    while True:
        img, lab, cs, reg = next(it)
        t0 = time.time()
        x, sem, csb = prep_batch(img, lab, cs, reg, device, train=True)
        lr = LR * min(1.0, (step + 1) / WARMUP) * (1 - min(step / est_total, 0.999)) ** 0.9
        for gp in opt.param_groups:
            gp["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, reg.to(device))
        if len(out) == 3:
            s_log, c_log, aux = out
        else:
            (s_log, c_log), aux = out, []
        loss = soft_dice_ce(s_log, sem, D.N_CLS) + CS_LOSS_W * cs_loss(c_log, csb)
        wsum_ds = DS_WEIGHTS[0]
        for lv, a in enumerate(aux):
            w = DS_WEIGHTS[lv + 1]
            f = 2 ** (lv + 1)
            sem_d = sem[:, ::f, ::f, ::f]
            cs_d = F.max_pool3d(csb[:, None], f)[:, 0]
            loss = loss + w * (soft_dice_ce(a[:, : D.N_CLS], sem_d, D.N_CLS)
                               + CS_LOSS_W * cs_loss(a[:, D.N_CLS:], cs_d))
            wsum_ds += w
        loss = loss / wsum_ds
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        opt.step()
        torch.cuda.synchronize()
        train_sec += time.time() - t0
        loss_acc += float(loss.detach())
        step += 1

        if step == 100:
            t_warm = train_sec
        if step == 300:
            sps = (train_sec - t_warm) / 200.0
            est_total = max(500, int(budget / sps))
            if rank == 0:
                print(f"[speed] {sps*1000:.0f} ms/step -> ~{est_total} steps in budget",
                      flush=True)

        if rank == 0 and step % 200 == 0:
            print(f"step {step} loss {loss_acc/200:.4f} lr {lr:.2e} "
                  f"train_s {train_sec:.0f}/{budget:.0f}", flush=True)
            loss_acc = 0.0

        # The two ranks accumulate slightly different training times, so the
        # decision to evaluate / stop must be collective -- otherwise one rank
        # enters the eval all-gather while the other is still all-reducing
        # gradients and NCCL deadlocks.  Checked on a stride to stay cheap.
        do_eval = do_stop = False
        if step % 25 == 0:
            flag = torch.tensor(
                [1.0 if (eval_at and train_sec >= eval_at[0]) else 0.0,
                 1.0 if train_sec >= budget else 0.0], device=device)
            if world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            do_eval, do_stop = bool(flag[0] > 0), bool(flag[1] > 0)

        if do_eval:
            if eval_at:
                eval_at.pop(0)
            te = time.time()
            rows = evaluate(raw, region_map, val_cases, device, rank, world)
            prim, met = aggregate(rows)
            eval_sec += time.time() - te
            if rank == 0:
                print(f"[eval] step {step} train_s {train_sec:.0f} {PRIMARY_METRIC} "
                      f"{prim:.4f} | pelvic {met['frag_dice_pelvic']:.4f} "
                      f"femur {met['frag_dice_femur']:.4f} | took {time.time()-te:.0f}s",
                      flush=True)
                torch.save({"model": raw.state_dict(), "region": regnet.state_dict(),
                            "step": step}, os.path.join(ckpt_dir, "last.pth"))
                better = prim > best if METRIC_DIRECTION == "higher" else prim < best
                if better or best_rows is None:
                    best = prim
                    best_rows = (rows, met)
                    torch.save({"model": raw.state_dict(), "region": regnet.state_dict(),
                                "step": step, PRIMARY_METRIC: prim, "metrics": met},
                               os.path.join(ckpt_dir, "best.pth"))
            if world > 1:
                dist.barrier()
        if do_stop:
            break

    if rank == 0:
        rows, met = best_rows
        met["region_acc_trainhead"] = reg_acc
        with open(os.path.join(run_dir, "val_cases.json"), "w") as f:
            json.dump(rows, f, indent=1)
        peak = torch.cuda.max_memory_allocated() / 1e6
        print("\n---")
        print(f"{PRIMARY_METRIC}:      {best:.4f}\n")
        print("metrics:")
        for k, v in met.items():
            print(f"{k}:  {v:.4f}")
        print(f"\nsteps:            {step}")
        print(f"training_seconds: {train_sec:.1f}")
        print(f"eval_seconds:     {eval_sec:.1f}")
        print(f"total_seconds:    {time.time()-t_start:.1f}")
        print(f"peak_vram_mb:     {peak:.1f}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def package_ensemble(ckpts, out_path):
    """Bundle several members plus the region classifier into one checkpoint, so
    the deliverable is a single self-contained file."""
    members, cfgs = [], []
    for c in ckpts:
        st = torch.load(c, map_location="cpu")
        w = st["model"]["head.weight"]
        cfgs.append({"base": int(st["model"]["enc.0.0.weight"].shape[0]),
                     "region_heads": bool(w.shape[0] == 2 * (D.N_CLS + 1)),
                     "source": c})
        members.append(st["model"])
    st0 = torch.load(ckpts[0], map_location="cpu")
    torch.save({"members": members, "member_cfg": cfgs, "region": st0["region"],
                "n_members": len(members)}, out_path)
    return cfgs


def _build_member(sd, cfg, device):
    global REGION_HEADS
    saved = REGION_HEADS
    REGION_HEADS = cfg["region_heads"]
    try:
        m = UNet3D(base=cfg["base"]).to(device).to(memory_format=torch.channels_last_3d)
        m.load_state_dict(sd, strict=False)
    finally:
        REGION_HEADS = saved
    m.eval()
    m._region_heads = cfg["region_heads"]
    return m


def _load_model(ckpt, device):
    """Load a member, reading its architecture back off the checkpoint so that
    ensembles can mix runs trained with different head layouts."""
    st = torch.load(ckpt, map_location="cpu")
    w = st["model"]["head.weight"]
    base = st["model"]["enc.0.0.weight"].shape[0]
    two_headed = w.shape[0] == 2 * (D.N_CLS + 1)
    global REGION_HEADS
    saved = REGION_HEADS
    REGION_HEADS = two_headed
    try:
        m = UNet3D(base=base).to(device).to(memory_format=torch.channels_last_3d)
        m.load_state_dict(st["model"], strict=False)   # older runs lack the deep-supervision heads
    finally:
        REGION_HEADS = saved
    m.eval()
    m._region_heads = two_headed
    return m, st


def sweep(ckpt, shard, nshard, gpu, out, taus=None, mins=None, gates=None,
          hs=None, ws=None):
    """Offline post-processing sweep on the validation set.

    Runs the (expensive) network once per case and then re-decodes the cached
    probability maps under several settings, so decoding can be tuned without
    spending a training run.  The scoring function is untouched.
    """
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    ckpts = [c for c in ckpt.split(",") if c]
    models, sts = zip(*[_load_model(c, device) for c in ckpts])
    model = models[0] if len(models) == 1 else _Ensemble(list(models))
    st = sts[0]
    regnet = RegionNet().to(device)
    regnet.load_state_dict(st["region"])
    val = D.load_cases("val")
    region_map = predict_regions(regnet, val, device)
    taus = taus or [0.3, 0.4, 0.5, 0.6, 0.7]
    mins = mins or [60.0, 250.0, 500.0]
    gates = gates or [HU_GATE]
    hs = hs or [WS_H]
    ws = ws or [WS_W]
    grid = [(t, m, g, h, w) for t in taus for m in mins for g in gates
            for h in hs for w in ws]
    acc = {f"{t}_{m}_{g}_{h}_{w}": [] for t, m, g, h, w in grid}
    gw = None
    for case in val[shard::nshard]:
        sem, cs, gw = predict_volume(model, case, region_map[case.sid], device, gw)
        gt = np.asarray(case.lab)
        hu = torch.from_numpy(np.asarray(case.img)).to(device).float()
        for t, m, g, h, w in grid:
            inst = decode_instances(sem, cs, device, cs_tau=t, min_mm3=m,
                                    img_hu=hu, hu_gate=g, ws_h=h, ws_w=w)
            r = score_case(inst, gt, case.meta["frag_vol_mm3"])
            acc[f"{t}_{m}_{g}_{h}_{w}"].append(
                {"sid": case.sid, "region": case.region, "frag_dice": r["frag_dice"],
                 "n_pred": int(len(np.unique(inst)) - 1), "n_gt": case.meta["n_frag"],
                 "n_gt_scored": r["n_gt_scored"], "n_matched": r["n_matched"],
                 "n_fp": r["n_fp"], "mean_matched_dice": r["mean_matched_dice"]})
        del sem, cs, hu
        torch.cuda.empty_cache()
        print(f"  {case.sid} done", flush=True)
    with open(out, "w") as f:
        json.dump(acc, f)
    print("wrote", out, flush=True)


def to_original_grid(inst, meta, out_path=None):
    """Place a cache-grid instance map back on the original image grid.

    The cache is 1 mm isotropic, cropped, and reoriented to LPS; the challenge
    requires the output to carry the input's dimensions and spacing exactly.
    """
    import SimpleITK as sitk

    full = np.zeros(meta["grid_size_full"][::-1], np.uint8)      # (Z, Y, X)
    lo = meta["crop_lo"]
    sh = meta["shape"]
    full[lo[0]:lo[0] + sh[0], lo[1]:lo[1] + sh[1], lo[2]:lo[2] + sh[2]] = \
        inst.astype(np.uint8)
    img = sitk.GetImageFromArray(full)
    img.SetSpacing(tuple(meta["grid_spacing"]))
    img.SetOrigin(tuple(meta["grid_origin"]))
    img.SetDirection(tuple(meta["grid_direction"]))

    o = meta["orig"]
    ref = sitk.Image([int(v) for v in o["size"]], sitk.sitkUInt8)
    ref.SetSpacing(tuple(o["spacing"]))
    ref.SetOrigin(tuple(o["origin"]))
    ref.SetDirection(tuple(o["direction"]))
    out = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                        sitk.sitkUInt8)
    if out_path:
        sitk.WriteImage(out, out_path, True)
    return out


def finalize(ckpt, gpu, shard, nshard, outdir, split="test"):
    """Test-set inference: challenge-format masks + per-case scores."""
    import SimpleITK as sitk

    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    ckpts = [c for c in ckpt.split(",") if c]
    if len(ckpts) == 1 and "members" in torch.load(ckpts[0], map_location="cpu"):
        st = torch.load(ckpts[0], map_location="cpu")
        models = [_build_member(sd, cfg, device)
                  for sd, cfg in zip(st["members"], st["member_cfg"])]
        model = models[0] if len(models) == 1 else _Ensemble(models)
        sts = [st]
    else:
        models, sts = zip(*[_load_model(c, device) for c in ckpts])
        model = models[0] if len(models) == 1 else _Ensemble(list(models))
    regnet = RegionNet().to(device)
    regnet.load_state_dict(sts[0]["region"])
    cases = D.load_cases(split)
    region_map = predict_regions(regnet, cases, device)
    os.makedirs(os.path.join(outdir, "masks"), exist_ok=True)
    rows, gw = [], None
    for case in cases[shard::nshard]:
        t0 = time.time()
        sem, cs, gw = predict_volume(model, case, region_map[case.sid], device, gw)
        inst = decode_instances(sem, cs, device)
        del sem, cs
        torch.cuda.empty_cache()

        gt = np.asarray(case.lab)
        r = score_case(inst, gt, case.meta["frag_vol_mm3"])

        mask_path = os.path.join(outdir, "masks", f"{case.sid}.mha")
        to_original_grid(inst, case.meta, mask_path)

        # second opinion on the native grid, against the untouched original label
        native = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int16)
        gt_native = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(D.subject_dir(case.sid), "label.mha"))
        ).astype(np.int16)
        sp = np.array(case.meta["orig"]["spacing"], float)
        vox = float(np.prod(sp))
        rn = score_case(native, gt_native, None, voxel_mm3=vox)

        rows.append({
            "sample_id": case.sid, "patient_id": case.sid, "region": case.region,
            "region_pred": region_map[case.sid],
            "n_gt_fragments": case.meta["n_frag"],
            "n_gt_scored": r["n_gt_scored"], "n_pred_fragments": r["n_pred_tot"],
            "n_matched": r["n_matched"], "n_false_positive": r["n_fp"],
            "frag_dice": r["frag_dice"], "frag_dice_native_grid": rn["frag_dice"],
            "mean_matched_dice": r["mean_matched_dice"],
            "assd_mm": r["assd"], "count_mae": r["count_mae"],
            "sem_dice": r["sem_dice"], "per_cls": r["per_cls"],
            "prediction": mask_path, "seconds": round(time.time() - t0, 1),
        })
        print(f"  {case.sid} {case.region} frag_dice {r['frag_dice']:.3f} "
              f"(native {rn['frag_dice']:.3f}) {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(outdir, f"test_rows_{shard}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote shard", shard, flush=True)


def main():
    if "--finalize" in sys.argv:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--finalize", action="store_true")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--gpu", type=int, default=0)
        ap.add_argument("--shard", type=int, default=0)
        ap.add_argument("--nshard", type=int, default=1)
        ap.add_argument("--outdir", required=True)
        ap.add_argument("--split", default="test")
        a = ap.parse_args()
        finalize(a.ckpt, a.gpu, a.shard, a.nshard, a.outdir, a.split)
        return
    if "--sweep" in sys.argv:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--sweep", action="store_true")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--shard", type=int, default=0)
        ap.add_argument("--nshard", type=int, default=1)
        ap.add_argument("--gpu", type=int, default=0)
        ap.add_argument("--out", required=True)
        ap.add_argument("--taus", default="")
        ap.add_argument("--mins", default="")
        ap.add_argument("--gates", default="")
        ap.add_argument("--hs", default="")
        ap.add_argument("--ws", default="")
        a = ap.parse_args()
        sweep(a.ckpt, a.shard, a.nshard, a.gpu, a.out,
              [float(v) for v in a.taus.split(",") if v],
              [float(v) for v in a.mins.split(",") if v],
              [float(v) for v in a.gates.split(",") if v],
              [float(v) for v in a.hs.split(",") if v],
              [float(v) for v in a.ws.split(",") if v])
        return
    run_dir = RUN_DIR or os.path.join(
        PROJECT, "run%d" % (1 + max([int(d[3:]) for d in os.listdir(PROJECT)
                                     if d.startswith("run") and d[3:].isdigit()] or [0])))
    os.makedirs(run_dir, exist_ok=True)
    world = torch.cuda.device_count()
    print(f"run_dir {run_dir} | gpus {world}", flush=True)
    if world <= 1:
        worker(0, 1, run_dir, 0)
    else:
        port = random.randint(29600, 29999)
        mp.spawn(worker, args=(world, run_dir, port), nprocs=world, join=True)


if __name__ == "__main__":
    main()
