"""
train.py — Frozen DINOv3 + conv decoder baseline for femoral-head segmentation.

Run:  uv run train.py
Outputs under projects/20260709_13/run{RUN}/: run.log (stdout), checkpoints/{best,last}.pth

EVALUATION CONTRACT (FIXED across the experiment loop):
  primary_metric = SegScore = 0.5*Dice + 0.5*NSD@tau   (higher is better)
  reported: Dice, IoU, Precision, Recall, NSD, HD95_norm, ASSD_norm (per head)
  aggregation: ALL (all heads), HARD (non normal/reference), and per-cohort means.
  Surface metrics computed at NATIVE mask resolution and normalized by the GT head
  equivalent diameter (no physical pixel spacing exists), tau = 0.05 * diameter.
"""
import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt, binary_erosion

import timm
from prepare import TIME_BUDGET_MINUTES
import data as D

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CFG = dict(
    run=5,
    seed=0,
    backbone="vit_large_patch16_dinov3_qkvb",
    weights="pretrained/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    embed_dim=1024,
    res=1024,
    feat_indices=[5, 11, 17, 23],   # multi-scale fusion
    decoder_ch=384,
    n_classes=2,                    # bilateral (ch0=image-left/patient-right, ch1=image-right/patient-left)
    dropout=0.1,
    boundary_w=5.0,                 # run5: upweight GT-margin pixels in BCE (target NSD/boundary)
    band_k=9,                       # boundary band width (px @ res)
    batch=6,
    lr=3e-4,
    wd=1e-4,
    val_every=5,            # epochs (tiny epochs -> avoid eval dominating wall-clock)
    eval_long=1024,         # cap metric computation at long-side 1024 (aspect preserved)
    tau_frac=0.05,          # NSD tolerance as fraction of head equiv-diameter
    thresh=0.5,
)
PROJECT = "projects/20260709_13"
DEVICE = "cuda:0"
BUDGET_S = TIME_BUDGET_MINUTES * 60


# --------------------------------------------------------------------------- #
# Backbone (frozen) with clean Meta->timm state_dict remap
# --------------------------------------------------------------------------- #
def remap_dinov3(sd, dim):
    out = {}
    for k, v in sd.items():
        if k == "mask_token" or k.endswith("bias_mask"):
            continue
        if k.endswith("attn.qkv.bias"):
            p = k[: -len("qkv.bias")]
            out[p + "q_bias"] = v[:dim].clone()
            out[p + "v_bias"] = v[2 * dim:].clone()
            continue
        nk = (k.replace(".ls1.gamma", ".gamma_1").replace(".ls2.gamma", ".gamma_2")
                .replace("storage_tokens", "reg_token"))
        out[nk] = v
    return out


def build_backbone():
    m = timm.create_model(CFG["backbone"], pretrained=False, num_classes=0, img_size=CFG["res"])
    sd = torch.load(CFG["weights"], map_location="cpu")
    missing, unexpected = m.load_state_dict(remap_dinov3(sd, CFG["embed_dim"]), strict=False)
    real_m = [k for k in missing if "rope" not in k]
    real_u = [k for k in unexpected if "rope" not in k]
    assert not real_m and not real_u, f"dirty load: missing={real_m[:5]} unexpected={real_u[:5]}"
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# Simple conv decoder: 1 ViT feature (H/16) -> full-res logits via 4x upsample.
# --------------------------------------------------------------------------- #
class ConvDecoder(nn.Module):
    def __init__(self, in_ch, ch, n_classes=1, dropout=0.1):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.GroupNorm(16, co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.GroupNorm(16, co), nn.GELU(),
            )
        self.proj = nn.Conv2d(in_ch, ch, 1)
        self.b1 = block(ch, ch)
        self.b2 = block(ch, ch // 2)
        self.b3 = block(ch // 2, ch // 4)
        self.b4 = block(ch // 4, ch // 4)
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(ch // 4, n_classes, 1)

    def forward(self, feat, out_hw):
        x = self.proj(feat)
        for b in (self.b1, self.b2, self.b3, self.b4):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = b(x)
        x = self.drop(x)
        x = self.head(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_backbone()
        n_feat = len(CFG["feat_indices"])
        self.decoder = ConvDecoder(CFG["embed_dim"] * n_feat, CFG["decoder_ch"],
                                   n_classes=CFG["n_classes"], dropout=CFG["dropout"])

    def features(self, x):
        with torch.no_grad():
            outs = self.backbone.forward_intermediates(
                x, indices=CFG["feat_indices"], norm=True,
                output_fmt="NCHW", intermediates_only=True)
        return torch.cat(outs, dim=1)  # multi-scale concat fusion

    def forward(self, x):
        feat = self.features(x)
        return self.decoder(feat, out_hw=x.shape[-2:])


# --------------------------------------------------------------------------- #
# Loss: Dice + BCE
# --------------------------------------------------------------------------- #
def _boundary_band(target, k):
    """Dilated GT boundary band, B,C,H,W in {0,1}. Emphasizes head-margin pixels."""
    pad = k // 2
    eroded = -F.max_pool2d(-target, k, 1, pad)
    boundary = (target - eroded).clamp(0, 1)
    return F.max_pool2d(boundary, k, 1, pad)


def dice_bce_loss(logits, target, presence):
    """Presence-masked (boundary-weighted BCE)+Dice over channels. B,C,H,W; presence B,C."""
    band = _boundary_band(target, CFG["band_k"])
    w = 1.0 + CFG["boundary_w"] * band
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w
    bce = bce_map.sum((2, 3)) / w.sum((2, 3))  # B,C  (normalized -> scale-stable)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((2, 3)) + 1
    den = p.sum((2, 3)) + target.sum((2, 3)) + 1
    dice = 1 - num / den  # B,C
    per = bce + dice
    return (per * presence).sum() / presence.sum().clamp(min=1)


# --------------------------------------------------------------------------- #
# METRICS (FIXED) — computed at native GT resolution
# --------------------------------------------------------------------------- #
def _surface(a, b):
    a = a.astype(bool); b = b.astype(bool)
    a_s = a ^ binary_erosion(a)
    b_s = b ^ binary_erosion(b)
    if a_s.sum() == 0 or b_s.sum() == 0:
        return None
    d_ab = distance_transform_edt(~b_s)[a_s]
    d_ba = distance_transform_edt(~a_s)[b_s]
    return d_ab, d_ba


def compute_metrics(pred, gt, tau_frac):
    """pred, gt: binary HxW (native). Returns dict of per-head metrics."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    ps, gs = pred.sum(), gt.sum()
    dice = 2 * inter / (ps + gs) if (ps + gs) > 0 else 1.0
    iou = inter / union if union > 0 else 1.0
    prec = inter / ps if ps > 0 else 0.0
    rec = inter / gs if gs > 0 else 0.0
    diam = 2 * math.sqrt(gs / math.pi) if gs > 0 else 1.0
    tau = tau_frac * diam
    if ps == 0:  # missed entirely
        nsd, hd95n, assdn = 0.0, 2.0, 2.0
    else:
        sd = _surface(pred, gt)
        if sd is None:
            nsd, hd95n, assdn = 0.0, 2.0, 2.0
        else:
            d_ab, d_ba = sd
            alld = np.concatenate([d_ab, d_ba])
            nsd = ((d_ab <= tau).sum() + (d_ba <= tau).sum()) / len(alld)
            hd95 = max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))
            hd95n = min(hd95 / diam, 2.0)
            assdn = min(alld.mean() / diam, 2.0)
    return dict(dice=float(dice), iou=float(iou), precision=float(prec),
                recall=float(rec), nsd=float(nsd), hd95=float(hd95n), assd=float(assdn),
                segscore=float(0.5 * dice + 0.5 * nsd))


def _eval_hw(H, W, long):
    s = long / max(H, W)
    return max(1, round(H * s)), max(1, round(W * s))


@torch.no_grad()
def evaluate(model, loader, tau_frac, thresh, eval_long):
    model.eval()
    rows = []
    for imgs, metas in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
        probs = torch.sigmoid(logits.float()).cpu().numpy()  # B,C,res,res
        for j, meta in enumerate(metas):
            gt = meta["gt"]
            H, W = gt.shape
            h, w = _eval_hw(H, W, eval_long)  # aspect-preserved, bounded cost
            import cv2 as _cv2
            gt_r = (_cv2.resize(gt, (w, h), interpolation=_cv2.INTER_NEAREST) > 0.5)
            # bilateral: pick channel by image side (right head->ch0, left head->ch1)
            ch = (0 if meta["side"] == "right" else 1) if probs.shape[1] == 2 else 0
            p = torch.from_numpy(probs[j, ch])[None, None]
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
            m = compute_metrics(p >= thresh, gt_r, tau_frac)
            m["cohort"] = meta["cohort"]
            rows.append(m)
    return aggregate(rows)


METRIC_KEYS = ["dice", "iou", "precision", "recall", "nsd", "hd95", "assd", "segscore"]


def aggregate(rows):
    def mean_over(subset):
        if not subset:
            return {k: float("nan") for k in METRIC_KEYS}
        return {k: float(np.mean([r[k] for r in subset])) for k in METRIC_KEYS}
    out = {"ALL": mean_over(rows),
           "HARD": mean_over([r for r in rows if r["cohort"] in D.HARD_COHORTS])}
    for c in D.COHORTS:
        out[c] = mean_over([r for r in rows if r["cohort"] == c])
    out["_n"] = len(rows)
    return out


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def format_report(agg, extra):
    lines = ["---", f"SegScore:      {agg['ALL']['segscore']:.4f}", "", "metrics_summary:"]
    for scope in ["ALL", "HARD"]:
        a = agg[scope]
        lines.append(f"[{scope}]  Dice {a['dice']:.4f}  IoU {a['iou']:.4f}  "
                     f"NSD {a['nsd']:.4f}  HD95n {a['hd95']:.4f}  ASSDn {a['assd']:.4f}  "
                     f"Prec {a['precision']:.4f}  Rec {a['recall']:.4f}  SegScore {a['segscore']:.4f}")
    for c in D.COHORTS:
        a = agg[c]
        lines.append(f"  {c:18s} Dice {a['dice']:.4f}  NSD {a['nsd']:.4f}  SegScore {a['segscore']:.4f}")
    for k, v in extra.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def main():
    CFG["seed"] = int(os.environ.get("SEED", CFG["seed"]))          # multi-seed final confirmation
    CFG["run"] = os.environ.get("RUN_TAG", CFG["run"])
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True
    run_dir = os.path.join(PROJECT, f"run{CFG['run']}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    tr_cases, va_cases, te_cases = D.make_splits(seed=CFG["seed"], val_frac=0.2)
    va = D.head_samples(va_cases)
    print(f"cases: train={len(tr_cases)} val={len(va_cases)} test={len(te_cases)} | "
          f"train_cases={len(tr_cases)} val_heads={len(va)}", flush=True)

    train_ds = D.CaseTrainDataset(tr_cases, CFG["res"], aug=D.DEFAULT_AUG, seed=CFG["seed"])
    val_ds = D.HeadEvalDataset(va, CFG["res"])
    train_ld = DataLoader(train_ds, batch_size=CFG["batch"], shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=CFG["batch"], shuffle=False, num_workers=6,
                        collate_fn=D.eval_collate, pin_memory=True)

    model = SegModel().to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG["lr"], weight_decay=CFG["wd"])
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    t0 = time.time()
    train_seconds = 0.0
    epoch = 0
    while True:
        model.train()
        ep_start = time.time()
        for imgs, masks, presence in train_ld:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            presence = presence.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = dice_bce_loss(logits, masks, presence)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        train_seconds += time.time() - ep_start
        epoch += 1

        budget_done = train_seconds >= BUDGET_S
        if epoch % CFG["val_every"] != 0 and not budget_done:
            print(f"epoch {epoch:3d} | loss {loss.detach().item():.4f} | t {train_seconds:.0f}s", flush=True)
            continue

        agg = evaluate(model, val_ld, CFG["tau_frac"], CFG["thresh"], CFG["eval_long"])
        seg = agg["ALL"]["segscore"]
        torch.save({"model": model.state_dict(), "cfg": CFG}, os.path.join(ckpt_dir, "last.pth"))
        improved = seg > best
        if improved:
            best = seg
            best_agg = agg
            torch.save({"model": model.state_dict(), "cfg": CFG, "agg": agg},
                       os.path.join(ckpt_dir, "best.pth"))
        print(f"epoch {epoch:3d} | loss {loss.detach().item():.4f} | val SegScore {seg:.4f} "
              f"Dice {agg['ALL']['dice']:.4f} HARD-Dice {agg['HARD']['dice']:.4f} "
              f"| best {best:.4f} | t {train_seconds:.0f}s{'  *' if improved else ''}", flush=True)

        if budget_done:
            break

    peak = torch.cuda.max_memory_allocated() / 1e6
    extra = {"training_seconds": f"{train_seconds:.1f}", "total_seconds": f"{time.time()-t0:.1f}",
             "peak_vram_mb": f"{peak:.1f}", "epochs": epoch}
    print(format_report(best_agg, extra), flush=True)


if __name__ == "__main__":
    main()
