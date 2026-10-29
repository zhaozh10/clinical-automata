"""Cardiac chamber / great-vessel / valve-plane segmentation on projection radiographs.

Primary metric: mean Dice over the four cardiac chambers (LA, RA, LV, RV), higher is better.
The evaluation function below is FIXED for the whole experiment loop.
"""

import json
import os
import re
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D
from prepare import TIME_BUDGET_MINUTES

PROJECT_DIR = "projects/20260723_07"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG = dict(
    encoder="convnext_small",
    batch_size=16,          # per GPU
    lr=4.5e-4,
    encoder_lr_mult=0.25,
    weight_decay=0.02,
    workers=6,
    val_per_view=300,
    eval_every_min=4.0,
    dice_weight=1.0,
    bce_weight=1.0,
    chamber_weight=4.0,
    ema=0.999,             # evaluate/checkpoint an EMA of the weights
    hier_weight=0.0,       # child-inside-parent containment (tested: no gain)
    boundary_alpha=0.0,    # extra BCE weight in a band around each structure edge
    warmup_frac=0.03,
    seed=1234,
    work_size=0,           # encoder grid; 0 = native 512 (768 tested: worse)
    dinov3_init=True,      # DINOv3 self-supervised weights instead of ImageNet
    gpu_aug=True,
    ddp=True,              # use every visible GPU via DistributedDataParallel
)


def run_dir():
    if os.environ.get("RUN_DIR"):
        return os.environ["RUN_DIR"]
    i = 1
    while os.path.exists(os.path.join(PROJECT_DIR, f"run{i}")):
        i += 1
    return os.path.join(PROJECT_DIR, f"run{i}")


# ---------------------------------------------------------------------------
# Model: U-Net with a pretrained hierarchical encoder
# ---------------------------------------------------------------------------
def _groups(ch):
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


DINOV3_DIR = "/path/to/your/datasets/pretrained/dinov3"
DINOV3_CONVNEXT = {
    "convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "convnext_small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
}


def _remap_dinov3_convnext(sd, in_chans=2):
    """Original ConvNeXt (FB) parameter names -> timm features_only names."""
    out = {}
    for k, v in sd.items():
        if k.startswith("downsample_layers.0."):
            out[k.replace("downsample_layers.0.", "stem_")] = v
        elif k.startswith("downsample_layers."):
            i = k.split(".")[1]
            out[re.sub(rf"^downsample_layers\.{i}\.", f"stages_{i}.downsample.", k)] = v
        elif k.startswith("stages."):
            m = re.match(r"stages\.(\d+)\.(\d+)\.(.*)", k)
            if not m:
                continue
            i, j, rest = m.groups()
            rest = (rest.replace("dwconv.", "conv_dw.")
                        .replace("pwconv1.", "mlp.fc1.")
                        .replace("pwconv2.", "mlp.fc2."))
            out[f"stages_{i}.blocks.{j}.{rest}"] = v
    w = out.get("stem_0.weight")
    if w is not None and w.shape[1] == 3 and in_chans != 3:
        out["stem_0.weight"] = w.sum(1, keepdim=True).repeat(1, in_chans, 1, 1) / in_chans
    return out


class UNet(nn.Module):
    def __init__(self, encoder="convnext_small", num_classes=D.NUM_CHANNELS,
                 decoder_ch=(256, 128, 64, 48, 32), work_size=0):
        super().__init__()
        self.work_size = work_size
        use_dinov3 = CFG.get("dinov3_init") and encoder in DINOV3_CONVNEXT
        self.encoder = timm.create_model(encoder, pretrained=not use_dinov3, features_only=True,
                                         in_chans=2, out_indices=(0, 1, 2, 3))
        if use_dinov3:
            ck = torch.load(os.path.join(DINOV3_DIR, DINOV3_CONVNEXT[encoder]), map_location="cpu")
            miss, unexp = self.encoder.load_state_dict(
                _remap_dinov3_convnext(ck.get("model", ck)), strict=False)
            assert not miss and not unexp, (len(miss), len(unexp))
        chs = [f["num_chs"] for f in self.encoder.feature_info]  # strides 4, 8, 16, 32
        d = decoder_ch
        self.dec3 = DecoderBlock(chs[3], chs[2], d[0])
        self.dec2 = DecoderBlock(d[0], chs[1], d[1])
        self.dec1 = DecoderBlock(d[1], chs[0], d[2])
        self.dec0 = DecoderBlock(d[2], 0, d[3])   # 1/4 -> 1/2
        self.dec_full = DecoderBlock(d[3], 0, d[4])  # 1/2 -> 1/1
        self.head = nn.Sequential(
            nn.Conv2d(d[4], d[4], 3, padding=1, bias=False),
            nn.GroupNorm(_groups(d[4]), d[4]),
            nn.GELU(),
            nn.Conv2d(d[4], num_classes, 1),
        )

    def forward(self, x):
        out_size = x.shape[-2:]
        if self.work_size and self.work_size != x.shape[-1]:
            # run the encoder at a larger grid: the stride-32 bottleneck is where the
            # global shape reasoning happens, and 16x16 may be too coarse for the
            # invisible chamber borders. Logits are returned at the input resolution.
            x = F.interpolate(x, size=(self.work_size, self.work_size), mode="bilinear",
                              align_corners=False)
        f0, f1, f2, f3 = self.encoder(x)
        y = self.dec3(f3, f2)
        y = self.dec2(y, f1)
        y = self.dec1(y, f0)
        y = self.dec0(y)
        y = self.dec_full(y)
        y = self.head(y)
        if y.shape[-2:] != out_size:
            y = F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)
        return y


class Ensemble(nn.Module):
    """Average the logits of several trained models - the members disagree most exactly
    where the projection hides the chamber border, and averaging there helps."""

    def __init__(self, checkpoints, map_location="cpu"):
        super().__init__()
        self.models = nn.ModuleList()
        for path in checkpoints:
            ck = torch.load(path, map_location=map_location)
            c = ck.get("cfg", CFG)
            m = UNet(c.get("encoder", "convnext_small"), work_size=c.get("work_size", 0))
            m.load_state_dict(ck["model"])
            self.models.append(m.eval())

    def forward(self, x):
        return torch.stack([m(x) for m in self.models]).mean(0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def channel_weights(chamber_weight=1.0, device="cuda"):
    """Per-channel loss weight; >1 for the chambers steers capacity to the primary metric."""
    w = torch.ones(D.NUM_CHANNELS, device=device)
    for i in D.CHAMBER_IDX:
        w[i] = chamber_weight
    return w


# Anatomical containment that holds exactly in the labels (verified on the training
# ground truth: 0.00% of chamber pixels fall outside the heart mask).
HIERARCHY_PAIRS = [(D.IDX[c], D.IDX[p]) for c, p in [
    ("atrium_left", "heart"), ("atrium_right", "heart"),
    ("ventricle_left", "heart"), ("ventricle_right", "heart"),
    ("myocardium", "heart"), ("heart", "cardiomediastinum"),
    ("pulmonary_trunk", "pulmonary_artery"), ("carina", "trachea"),
]]


def hierarchy_loss(prob):
    """Penalise a child structure claiming a pixel its parent does not."""
    l = 0.0
    for c, p in HIERARCHY_PAIRS:
        l = l + torch.relu(prob[:, c] - prob[:, p]).mean()
    return l / len(HIERARCHY_PAIRS)


def boundary_weight(target, k=7, alpha=3.0):
    """Up-weight pixels near a structure's edge - that is where the Dice error lives,
    and on a projection the chamber edge is often the only inferable cue."""
    d = F.max_pool2d(target, k, 1, k // 2)
    e = -F.max_pool2d(-target, k, 1, k // 2)
    return 1.0 + alpha * (d - e)


def masked_loss(logits, target, valid, dice_w=1.0, bce_w=1.0, chan_w=None, hier_w=0.0,
                boundary_alpha=0.0):
    """valid: (B, C) - 0 for channels without annotation on that view."""
    if chan_w is not None:
        valid = valid * chan_w[None, :]
    v = valid[:, :, None, None]
    if boundary_alpha:
        v = v * boundary_weight(target, alpha=boundary_alpha)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * v).sum() / (v.expand_as(bce).sum() + 1e-6)

    p = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    dice = (dice * valid).sum() / (valid.sum() + 1e-6)
    loss = bce_w * bce + dice_w * dice
    if hier_w:
        loss = loss + hier_w * hierarchy_loss(p)
    return loss


# ---------------------------------------------------------------------------
# Evaluation  (FIXED - do not change during the experiment loop)
# ---------------------------------------------------------------------------
BORDER_CANDIDATES = ["aorta_ascending", "aorta_arch", "aorta_descending", "pulmonary_trunk",
                     "atrium_right", "atrium_left", "ventricle_left", "ventricle_right",
                     "vena_cava_inferior"]
BORDER_IDX = [D.IDX[n] for n in BORDER_CANDIDATES]
VALVE_MISS_PENALTY_PX = 64.0
# Cohort-relative thresholds from the TRAINING ground truth (reference_stats.json).
# PAXRay renders DRRs from CT, whose projection geometry gives systematically larger
# CTR than a PA film at 2 m (cohort median CTR 0.51), so the classical 0.5 cut-off
# does not separate large hearts here - the cohort's own upper quartile does.
ENLARGED_CTR = 0.55          # ~p75 of the training cohort
LV_DOMINANT_FRAC = 0.251     # ~p90 of LV share of total chamber area


def _width(m):
    cols = np.nonzero(m.any(0))[0]
    return 0 if cols.size == 0 else int(cols[-1] - cols[0] + 1)


def _ctr(heart, lungs):
    """Cardiothoracic ratio: max transverse cardiac diameter / inner thoracic width."""
    w_h, w_t = _width(heart), _width(lungs)
    if w_h == 0 or w_t == 0:
        return None
    return w_h / w_t


def _silhouette_rows(dense, K):
    """Rows spanned by the cardiac silhouette: top of arch/ascending aorta -> bottom of heart."""
    core = dense[0] | dense[1] | dense[K]      # ascending aorta | arch | heart
    rows = np.nonzero(core.any(1))[0]
    m = np.zeros(dense.shape[1], bool)
    if rows.size:
        m[rows[0]:rows[-1] + 1] = True
    return m


def _border_labels(cand, row_mask=None):
    """For each image row: which structure forms the right / left cardiac border.

    cand: (K, H, W) uint8 stack of border-capable structures.
    row_mask: (H,) bool - rows belonging to the cardiac silhouette (aortic arch down
    to the diaphragm); rows outside are not attributed.
    Returns (2, H) int32 with the candidate index, -1 where no silhouette.
    """
    sil = cand.any(0)                                  # (H, W)
    if row_mask is not None:
        sil = sil & row_mask[:, None]
    H, W = sil.shape
    rows_any = sil.any(1)
    first = np.argmax(sil, axis=1)                     # right border (patient's right = min x)
    last = W - 1 - np.argmax(sil[:, ::-1], axis=1)     # left border
    rowcount = cand.sum(2).astype(np.int32)            # (K, H) tie-break: dominant structure in row
    ys = np.arange(H)
    out = np.full((2, H), -1, np.int32)
    for side, xs in ((0, first), (1, last)):
        owners = cand[:, ys, xs] > 0                   # (K, H)
        score = np.where(owners, rowcount[:, ys], -1)
        lab = np.argmax(score, axis=0)
        has = owners.any(0) & rows_any
        out[side] = np.where(has, lab, -1)
    return out


@torch.no_grad()
def evaluate(model, loader, device, val_records, threshold=0.5):
    model.eval()
    C = D.NUM_CHANNELS
    inter = np.zeros(C); psum = np.zeros(C); tsum = np.zeros(C); npos = np.zeros(C)
    valve_err = {n: [] for n, _, _ in D.VALVE_CHANNELS}
    valve_hit = {n: [] for n, _, _ in D.VALVE_CHANNELS}
    border_ok = 0; border_tot = 0
    ctr_abs = []; ctr_pairs = []
    chamber_dice_enlarged = []; chamber_dice_normal = []
    chamber_dice_lvdom = []; lv_dice_lvdom = []

    # channels that are needed as dense CPU arrays (border attribution + CTR)
    dense_idx = BORDER_IDX + [D.IDX["heart"], D.IDX["lung_right"], D.IDX["lung_left"]]
    K = len(BORDER_IDX)
    yy = torch.arange(D.IMG_SIZE, device=device, dtype=torch.float32)

    for x, y, valid, idxs in loader:
        x = x.to(device, non_blocking=True)
        yg = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        prob = torch.sigmoid(logits.float())                  # metrics in fp32
        p = (prob > threshold).float()
        g = (yg > 0.5).float()
        v = valid.to(device)

        # per-channel reductions on GPU
        i_b = (p * g).sum((2, 3))                             # (B, C)
        p_b = p.sum((2, 3))
        g_b = g.sum((2, 3))
        # valve centroids on GPU
        cy_p = (p * yy[None, None, :, None]).sum((2, 3))
        cx_p = (p * yy[None, None, None, :]).sum((2, 3))
        cy_g = (g * yy[None, None, :, None]).sum((2, 3))
        cx_g = (g * yy[None, None, None, :]).sum((2, 3))

        i_b, p_b, g_b = i_b.cpu().numpy(), p_b.cpu().numpy(), g_b.cpu().numpy()
        cy_p, cx_p, cy_g, cx_g = [t.cpu().numpy() for t in (cy_p, cx_p, cy_g, cx_g)]
        val = valid.numpy()
        dense_p = p[:, dense_idx].to(torch.uint8).cpu().numpy()
        dense_g = g[:, dense_idx].to(torch.uint8).cpu().numpy()

        for b in range(i_b.shape[0]):
            rec = val_records[int(idxs[b])]
            vb = val[b]
            inter += i_b[b] * vb
            psum += p_b[b] * vb
            tsum += g_b[b] * vb
            npos += (g_b[b] > 0) * vb

            # per-image chamber dice for subgroup analysis
            cd = [2 * i_b[b, c] / (p_b[b, c] + g_b[b, c] + 1e-6)
                  for c in D.CHAMBER_IDX if g_b[b, c] > 0]
            cd = float(np.mean(cd)) if cd else None

            for name, _, _ in D.VALVE_CHANNELS:
                c = D.IDX[name]
                if g_b[b, c] == 0:
                    continue
                if p_b[b, c] == 0:
                    valve_err[name].append(VALVE_MISS_PENALTY_PX)
                    valve_hit[name].append(0.0)
                else:
                    d = np.hypot(cy_g[b, c] / g_b[b, c] - cy_p[b, c] / p_b[b, c],
                                 cx_g[b, c] / g_b[b, c] - cx_p[b, c] / p_b[b, c])
                    valve_err[name].append(float(d))
                    valve_hit[name].append(1.0)

            if rec["view"] == "frontal":
                # the cardiac silhouette discussed clinically spans aortic arch -> diaphragm
                rm_g = _silhouette_rows(dense_g[b], K)
                bg = _border_labels(dense_g[b, :K], rm_g)
                bp = _border_labels(dense_p[b, :K], rm_g)
                m = bg >= 0
                border_tot += int(m.sum())
                border_ok += int(((bg == bp) & m).sum())
                cg = _ctr(dense_g[b, K], dense_g[b, K + 1] | dense_g[b, K + 2])
                cp = _ctr(dense_p[b, K], dense_p[b, K + 1] | dense_p[b, K + 2])
                if cg is not None:
                    cp = cp if cp is not None else 0.0
                    ctr_pairs.append((cg, cp))
                    ctr_abs.append(abs(cg - cp))
                    if cd is not None:
                        (chamber_dice_enlarged if cg >= ENLARGED_CTR
                         else chamber_dice_normal).append(cd)
                # LV-dominant hearts: does the model still resolve the enlarged LV?
                chamber_area = sum(g_b[b, c] for c in D.CHAMBER_IDX) + 1e-6
                if g_b[b, D.IDX["ventricle_left"]] / chamber_area >= LV_DOMINANT_FRAC:
                    if cd is not None:
                        chamber_dice_lvdom.append(cd)
                    lv = D.IDX["ventricle_left"]
                    if g_b[b, lv] > 0:
                        lv_dice_lvdom.append(2 * i_b[b, lv] / (p_b[b, lv] + g_b[b, lv] + 1e-6))

    dice = np.where(npos > 0, 2 * inter / np.maximum(psum + tsum, 1e-6), np.nan)
    per_class = {D.CHANNEL_NAMES[i]: float(dice[i]) for i in range(C)}
    mean_chambers = float(np.nanmean([dice[i] for i in D.CHAMBER_IDX]))
    metrics = {
        "mean_dice_chambers": mean_chambers,
        "mean_dice_vessels": float(np.nanmean([dice[i] for i in D.VESSEL_IDX])),
        "mean_dice_valves": float(np.nanmean([dice[i] for i in D.VALVE_IDX])),
        "mean_dice_all": float(np.nanmean(dice)),
        "valve_centroid_err_px": float(np.mean([np.mean(v) for v in valve_err.values() if v])),
        "valve_detection_rate": float(np.mean([np.mean(v) for v in valve_hit.values() if v])),
        "border_attribution_acc": float(border_ok / max(border_tot, 1)),
        "ctr_mae": float(np.mean(ctr_abs)) if ctr_abs else float("nan"),
        "mean_dice_chambers_enlarged": float(np.mean(chamber_dice_enlarged)) if chamber_dice_enlarged else float("nan"),
        "mean_dice_chambers_normal": float(np.mean(chamber_dice_normal)) if chamber_dice_normal else float("nan"),
        "mean_dice_chambers_lv_dominant": float(np.mean(chamber_dice_lvdom)) if chamber_dice_lvdom else float("nan"),
        "dice_lv_in_lv_dominant": float(np.mean(lv_dice_lvdom)) if lv_dice_lvdom else float("nan"),
        "ctr_bias": float(np.mean([p - g for g, p in ctr_pairs])) if ctr_pairs else float("nan"),
        "n_enlarged": len(chamber_dice_enlarged),
        "n_lv_dominant": len(chamber_dice_lvdom),
    }
    for n in D.CHANNEL_NAMES:
        metrics[f"dice_{n}"] = per_class[n]
    for n, v in valve_err.items():
        metrics[f"err_{n}_px"] = float(np.mean(v)) if v else float("nan")
    model.train()
    return {"mean_dice_chambers": mean_chambers, "metrics": metrics}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def worker(rank, world, rd, t_start):
    ddp = world > 1
    if ddp:
        import torch.distributed as dist
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)
    is_main = rank == 0
    torch.manual_seed(CFG["seed"] + rank); np.random.seed(CFG["seed"] + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main:
        print(f"run dir: {rd}  world size {world}", flush=True)
        print("config:", json.dumps(CFG), flush=True)

    device = f"cuda:{rank}" if ddp else "cuda"
    train_loader, val_loader, train_recs, val_recs = D.build_loaders(
        batch_size=CFG["batch_size"], workers=CFG["workers"],
        val_per_view=CFG["val_per_view"], seed=CFG["seed"], gpu_aug=CFG["gpu_aug"],
        rank=rank, world=world)
    if is_main:
        print(f"train imgs {len(train_recs)}  val imgs {len(val_recs)}  "
              f"channels {D.NUM_CHANNELS}", flush=True)

    model = UNet(CFG["encoder"], work_size=CFG["work_size"]).to(device).to(memory_format=torch.channels_last)
    core = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    enc_params = [p for p in core.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": CFG["lr"] * CFG["encoder_lr_mult"]},
        {"params": dec_params, "lr": CFG["lr"]},
    ], weight_decay=CFG["weight_decay"])

    budget = TIME_BUDGET_MINUTES * 60
    train_seconds = 0.0
    best = -1.0
    best_metrics = None
    step = 0
    eval_every = CFG["eval_every_min"] * 60
    next_eval = eval_every

    def lr_scale():
        f = train_seconds / budget
        if f < CFG["warmup_frac"]:
            return f / CFG["warmup_frac"]
        return 0.5 * (1 + np.cos(np.pi * min(1.0, (f - CFG["warmup_frac"]) / (1 - CFG["warmup_frac"]))))

    base_lrs = [g["lr"] for g in opt.param_groups]
    chan_w = channel_weights(CFG["chamber_weight"], device)
    ema = None
    if CFG["ema"] and is_main:
        import copy
        ema = copy.deepcopy(core).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    model.train()
    done = False
    epoch = 0
    while not done:
        if ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            t0 = time.time()
            if CFG["gpu_aug"]:
                img, mask, view, valid, _ = batch
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                x, y = D.gpu_batch(img, mask, view, train=True)
                x = x.to(memory_format=torch.channels_last)
            else:
                x, y, valid, _ = batch
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            s = lr_scale()
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * s
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = masked_loss(logits.float(), y, valid, CFG["dice_weight"], CFG["bce_weight"],
                               chan_w, CFG["hier_weight"], CFG["boundary_alpha"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if ema is not None:
                d = min(CFG["ema"], (step + 1) / (step + 10))  # warm up the average
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), core.parameters()):
                        pe.lerp_(pm.detach(), 1 - d)
                    for be, bm in zip(ema.buffers(), core.buffers()):
                        be.copy_(bm)
            torch.cuda.synchronize()
            train_seconds += time.time() - t0
            step += 1

            # Eval/stop must be decided COLLECTIVELY: if one rank evaluated while the
            # other kept training, that rank's gradient all-reduce would wait forever.
            do_eval = do_stop = False
            if step % 50 == 0:
                if is_main:
                    print(f"step {step} t {train_seconds:.0f}s loss {loss.item():.4f} "
                          f"lr {opt.param_groups[1]['lr']:.2e}", flush=True)
                flags = torch.tensor(
                    [float(train_seconds >= next_eval or train_seconds >= budget),
                     float(train_seconds >= budget), train_seconds], device=device)
                if ddp:
                    import torch.distributed as dist
                    dist.broadcast(flags, 0)
                do_eval = bool(flags[0].item())
                do_stop = bool(flags[1].item())
                train_seconds = float(flags[2].item())   # keep the LR schedule in lockstep

            if do_eval:
                if is_main:
                    eval_model = ema if ema is not None else core
                    res = evaluate(eval_model, val_loader, device, val_recs)
                    m = res["metrics"]
                    print(f"[eval] t {train_seconds:.0f}s  mean_dice_chambers {res['mean_dice_chambers']:.4f} "
                          f"LA {m['dice_atrium_left']:.3f} RA {m['dice_atrium_right']:.3f} "
                          f"LV {m['dice_ventricle_left']:.3f} RV {m['dice_ventricle_right']:.3f} "
                          f"valves {m['mean_dice_valves']:.3f} border {m['border_attribution_acc']:.3f} "
                          f"ctr_mae {m['ctr_mae']:.4f}", flush=True)
                    torch.save({"model": eval_model.state_dict(), "cfg": CFG, "step": step},
                               os.path.join(rd, "checkpoints", "last.pth"))
                    if res["mean_dice_chambers"] > best:
                        best = res["mean_dice_chambers"]
                        best_metrics = m
                        torch.save({"model": eval_model.state_dict(), "cfg": CFG, "step": step,
                                    "metrics": m}, os.path.join(rd, "checkpoints", "best.pth"))
                    model.train()
                next_eval = train_seconds + eval_every
                if ddp:
                    import torch.distributed as dist
                    dist.barrier()

            if do_stop:
                done = True
                break

    if ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    if not is_main:
        return

    peak = torch.cuda.max_memory_allocated() / 1e6
    print("\n---")
    print(f"mean_dice_chambers: {best:.4f}\n")
    for k in ["dice_atrium_left", "dice_atrium_right", "dice_ventricle_left", "dice_ventricle_right",
              "dice_myocardium", "dice_heart", "mean_dice_vessels", "mean_dice_valves",
              "valve_centroid_err_px", "valve_detection_rate", "border_attribution_acc",
              "ctr_mae", "ctr_bias", "mean_dice_chambers_enlarged", "mean_dice_chambers_normal",
              "mean_dice_chambers_lv_dominant", "dice_lv_in_lv_dominant",
              "n_enlarged", "n_lv_dominant", "mean_dice_all"]:
        print(f"{k}: {best_metrics[k]:.4f}" if isinstance(best_metrics[k], float) else f"{k}: {best_metrics[k]}")
    print(f"\ntraining_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak:.1f}")
    with open(os.path.join(rd, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=1)


def main():
    t_start = time.time()
    rd = run_dir()
    os.makedirs(os.path.join(rd, "checkpoints"), exist_ok=True)
    world = torch.cuda.device_count() if CFG["ddp"] else 1
    if world > 1:
        torch.multiprocessing.spawn(worker, args=(world, rd, t_start), nprocs=world, join=True)
    else:
        worker(0, 1, rd, t_start)


if __name__ == "__main__":
    main()
