"""
train.py — PAXRay++ frontal CXR lung LOBE & ZONE segmentation (teaching tool).

- Multi-label (overlapping) segmentation: 14 sigmoid channels (5 lobes, 6 zones,
  3 lung-context) trained with BCE + soft-Dice. NOT argmax.
- MC-dropout head so inference can emit per-pixel confidence + uncertainty maps
  (the fissure/overlap regions where lobe assignment is genuinely ambiguous).
- Model selection: PRIMARY metric = mean Dice over the 11 lobe+zone channels
  (higher is better). Calibration (ECE/Brier) is logged only, never selected on.

Fixed evaluation contract: evaluate() returns
  {"mean_dice": float, "metrics": {...}} and must not change during the loop.
"""
import os
import sys
import math
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from prepare import TIME_BUDGET_MINUTES
import data as D

# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------
RUN_DIR = os.environ.get("RUN_DIR", "projects/20260708_13/run1")
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

SEED = 42
BATCH = int(os.environ.get("BATCH", "24"))
NUM_WORKERS = 8
LR = float(os.environ.get("LR", "5e-4"))   # run6: 5e-4 > 3e-4 (underconverged at 3e-4)
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
ENCODER = os.environ.get("ENCODER", "resnet34")
ARCH = os.environ.get("ARCH", "unet")       # unet | unetplusplus | manet | fpn
MC_DROPOUT_P = float(os.environ.get("MC_DROPOUT_P", "0.1"))   # run9: 0.1 > 0.2 (underconverged)
VAL_MAX = 500          # fixed deterministic val subset for in-loop selection (cheap+stable)
CALIB_BINS = 15
PRIMARY_METRIC = "mean_dice"
METRIC_DIRECTION = "higher"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Model: pretrained U-Net + MC-dropout before the segmentation head
# ---------------------------------------------------------------------------
class LobeZoneNet(nn.Module):
    def __init__(self, encoder=ENCODER, n_classes=D.N_CLASSES, drop_p=MC_DROPOUT_P, arch=ARCH):
        super().__init__()
        arch_map = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus,
                    "manet": smp.MAnet, "fpn": smp.FPN}
        self.net = arch_map[arch](encoder_name=encoder, encoder_weights="imagenet",
                                  in_channels=1, classes=n_classes, activation=None)
        # insert dropout right before the final 1x1 seg conv -> enables MC-dropout
        self.net.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=drop_p), self.net.segmentation_head
        )

    def forward(self, x):
        return self.net(x)   # (B, n_classes, H, W) logits


def enable_mc_dropout(model):
    """Keep BN in eval mode but re-activate dropout layers for MC sampling."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


# ---------------------------------------------------------------------------
# Loss: BCE + soft Dice over all channels
# ---------------------------------------------------------------------------
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    tp = (p * target).sum(dims)
    fp = (p * (1 - target)).sum(dims)
    fn = ((1 - p) * target).sum(dims)
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tv.mean()


LOSS = os.environ.get("LOSS", "bcedice")   # bcedice | bcetversky


def criterion(logits, target):
    if LOSS == "bcetversky":
        return F.binary_cross_entropy_with_logits(logits, target) + tversky_loss(logits, target)
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


# ---------------------------------------------------------------------------
# Evaluation (FIXED). fp32 metrics. Dataset-level Dice/IoU @0.5 per class,
# macro over the 11 lobe+zone channels = primary. Calibration inside lung.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    n = D.N_CLASSES
    inter = torch.zeros(n, dtype=torch.float64, device=DEVICE)   # for dice
    psum = torch.zeros(n, dtype=torch.float64, device=DEVICE)
    gsum = torch.zeros(n, dtype=torch.float64, device=DEVICE)
    union = torch.zeros(n, dtype=torch.float64, device=DEVICE)   # for iou
    # calibration accumulators (within-lung pixels, over primary channels)
    bin_conf = torch.zeros(CALIB_BINS, dtype=torch.float64, device=DEVICE)
    bin_acc = torch.zeros(CALIB_BINS, dtype=torch.float64, device=DEVICE)
    bin_cnt = torch.zeros(CALIB_BINS, dtype=torch.float64, device=DEVICE)
    brier_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    brier_cnt = torch.zeros((), dtype=torch.float64, device=DEVICE)
    lung_ch = D.CLASS_NAMES.index("lung")
    prim = torch.tensor(D.PRIMARY_IDX, device=DEVICE)

    for img, mask, _ in loader:
        img = img.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True).float()
        with torch.autocast("cuda", enabled=False):
            logits = model(img).float()
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        inter += (pred * mask).sum((0, 2, 3)).double()
        psum += pred.sum((0, 2, 3)).double()
        gsum += mask.sum((0, 2, 3)).double()
        union += ((pred + mask) >= 1).float().sum((0, 2, 3)).double()

        # calibration only where lung GT is present, primary channels only
        lung = mask[:, lung_ch:lung_ch + 1]                    # (B,1,H,W)
        pp = prob.index_select(1, prim)                        # (B,P,H,W)
        gg = mask.index_select(1, prim)
        m = lung.expand_as(pp) > 0.5
        pv = pp[m]; gv = gg[m]
        if pv.numel() > 0:
            brier_sum += ((pv - gv) ** 2).sum().double()
            brier_cnt += pv.numel()
            idx = torch.clamp((pv * CALIB_BINS).long(), 0, CALIB_BINS - 1)
            bin_cnt.index_add_(0, idx, torch.ones_like(pv, dtype=torch.float64))
            bin_conf.index_add_(0, idx, pv.double())
            bin_acc.index_add_(0, idx, gv.double())

    dice = (2 * inter + 1) / (psum + gsum + 1)
    iou = (inter + 1) / (union + 1)
    dice = dice.cpu().numpy(); iou = iou.cpu().numpy()

    prim_idx = D.PRIMARY_IDX
    mean_dice = float(np.mean([dice[i] for i in prim_idx]))
    mean_iou = float(np.mean([iou[i] for i in prim_idx]))
    lobe_dice = float(np.mean([dice[i] for i in D.LOBE_IDX]))
    zone_dice = float(np.mean([dice[i] for i in D.ZONE_IDX]))

    valid = bin_cnt > 0
    ece = float((((bin_acc / bin_cnt.clamp(min=1) - bin_conf / bin_cnt.clamp(min=1)).abs()
                  * bin_cnt)[valid].sum() / bin_cnt.sum().clamp(min=1)).item())
    brier = float((brier_sum / brier_cnt.clamp(min=1)).item())

    per_class = {D.CLASS_NAMES[i]: round(float(dice[i]), 4) for i in range(n)}
    return {
        "mean_dice": mean_dice,
        "metrics": {
            "mean_dice": round(mean_dice, 4),
            "mean_iou": round(mean_iou, 4),
            "lobe_dice": round(lobe_dice, 4),
            "zone_dice": round(zone_dice, 4),
            "ece_calibration": round(ece, 4),
            "brier": round(brier, 4),
            "per_class_dice": per_class,
        },
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    tr_ds, va_ds, te_ds, sp = D.build_datasets()
    # fixed deterministic val subset for cheap, stable in-loop selection
    if VAL_MAX and len(va_ds) > VAL_MAX:
        g = np.random.RandomState(0)
        sub = sorted(g.permutation(len(va_ds))[:VAL_MAX].tolist())
        va_ds = torch.utils.data.Subset(va_ds, sub)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                           pin_memory=True, drop_last=True, persistent_workers=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=True, persistent_workers=True)

    model = LobeZoneNet().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    def set_lr(frac):
        # cosine decay over the training-time budget (frac in [0,1]); 3% warmup
        if frac < 0.03:
            lr = LR * frac / 0.03
        else:
            p = (frac - 0.03) / 0.97
            lr = 0.5 * (1 + math.cos(math.pi * min(p, 1.0))) * LR
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    budget_s = TIME_BUDGET_MINUTES * 60
    train_seconds = 0.0
    t_total0 = time.time()
    best = -1e9 if METRIC_DIRECTION == "higher" else 1e9
    best_metrics = None

    print(f"train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} | budget={TIME_BUDGET_MINUTES}min")
    epoch = 0
    while train_seconds < budget_s:
        epoch += 1
        model.train()
        t_ep = time.time()
        running = 0.0; nb = 0
        for img, mask, _ in tr_loader:
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True).float()
            set_lr(min(train_seconds / budget_s, 1.0))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item(); nb += 1
            train_seconds = (time.time() - t_total0) - eval_time_acc[0]
            if train_seconds >= budget_s:
                break
        # ---- evaluate (time excluded from training budget) ----
        t_ev = time.time()
        res = evaluate(model, va_loader)
        eval_time_acc[0] += time.time() - t_ev
        md = res["mean_dice"]
        print(f"[epoch {epoch}] loss={running/max(nb,1):.4f} "
              f"train_s={train_seconds:.0f} {PRIMARY_METRIC}={md:.4f} "
              f"lobe={res['metrics']['lobe_dice']:.4f} zone={res['metrics']['zone_dice']:.4f} "
              f"ece={res['metrics']['ece_calibration']:.4f}", flush=True)

        sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"model": sd, "classes": D.CLASS_NAMES, "spec": D.TARGET_SPEC,
                    "encoder": ENCODER, "mc_dropout_p": MC_DROPOUT_P}, os.path.join(CKPT_DIR, "last.pth"))
        improved = md > best if METRIC_DIRECTION == "higher" else md < best
        if improved:
            best = md; best_metrics = res["metrics"]
            torch.save({"model": sd, "classes": D.CLASS_NAMES, "spec": D.TARGET_SPEC,
                        "encoder": ENCODER, "mc_dropout_p": MC_DROPOUT_P,
                        "val_metrics": best_metrics}, os.path.join(CKPT_DIR, "best.pth"))

    total_seconds = time.time() - t_total0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE == "cuda" else 0.0

    # ---- report (program.md output format) ----
    print("\n---")
    print(f"{PRIMARY_METRIC}:      {best:.4f}\n")
    print(f"metrics_summary:")
    for k, v in best_metrics.items():
        if k == "per_class_dice":
            continue
        print(f"{k}:            {v}")
    print(f"per_class_dice: {json.dumps(best_metrics['per_class_dice'])}")
    print(f"\ntraining_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")


eval_time_acc = [0.0]   # module-level so training-budget timer excludes eval

if __name__ == "__main__":
    main()
