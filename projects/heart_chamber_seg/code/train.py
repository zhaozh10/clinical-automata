"""Cardiac segmentation on PAXRay++ DRRs (frontal + lateral), aimed at real film.

Encoder is xraydino (DINOv2-recipe ViT-L/16 self-supervised on *real* radiographs,
native 512x512 grid).  The supervision here is CT-projected, so the encoder is the
main thing carrying real-film competence across the DRR domain gap; the decoder is
what we fit.

Model selection uses dice_heart (the cardiac silhouette) ONLY.  Chamber Dice is
logged but never selected on -- chambers are not observable on a plain film.
"""
import os
import re
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prepare import TIME_BUDGET_MINUTES
from data import (HeartDataset, make_splits, unpack_masks, augment,
                  HEART_NAMES, N_CH, PRIMARY_CH, CHAMBER_CH, IMG_SIZE)

XRAYDINO = "/path/to/your/datasets/pretrained/xraydino/xray_dino_vitl16_pretrained-ad31c2b0.pth"
OUT_DIR = os.environ.get("RUN_DIR", "projects/20260717_07/run1")

# ---- knobs (this run) ----------------------------------------------------
BATCH = int(os.environ.get("BATCH", 16))
LR = float(os.environ.get("LR", 3e-4))
WD = 0.01
UNFREEZE_LAST = int(os.environ.get("UNFREEZE_LAST", 24))  # trailing encoder blocks to finetune (0 = frozen); full ft = best
ENC_LR_MULT = float(os.environ.get("ENC_LR_MULT", 0.1))   # encoder pretrained on real film; adapt gently
TAPS = (5, 11, 17, 23)          # intermediate blocks fed to the decoder
EVAL_EVERY = 2000               # steps; ~6 evals inside the budget (~100s each)
SEED = 17


# ==========================================================================
# Model
# ==========================================================================
def build_encoder():
    import timm
    enc = timm.models.VisionTransformer(
        img_size=IMG_SIZE, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, init_values=1e-5, class_token=True, global_pool="",
        num_classes=0, dynamic_img_size=False,
    )
    raw = torch.load(XRAYDINO, map_location="cpu")
    # DINOv2 stores blocks chunked as blocks.<chunk>.<global_idx>.*; timm wants
    # blocks.<global_idx>.*  (the chunk axis is a sharding artifact, indices are global).
    sd = {}
    for k, v in raw.items():
        m = re.match(r"^blocks\.\d+\.(\d+)\.(.*)$", k)
        sd[f"blocks.{m.group(1)}.{m.group(2)}" if m else k] = v
    sd.pop("mask_token", None)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "head" not in k], f"missing enc weights: {missing[:6]}"
    assert not unexpected, f"unexpected: {unexpected[:6]}"
    print(f"[model] xraydino ViT-L/16 loaded ({sum(p.numel() for p in enc.parameters())/1e6:.0f}M)")
    return enc


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.f(x)


class HeartSeg(nn.Module):
    """Frozen/finetuned ViT encoder + progressive conv decoder, 32x32 -> 512x512."""

    def __init__(self, unfreeze_last=UNFREEZE_LAST, taps=TAPS):
        super().__init__()
        self.encoder = build_encoder()
        self.taps = taps
        self.depth = len(self.encoder.blocks)
        self.n_frozen = self.depth - unfreeze_last
        self.freeze = unfreeze_last == 0
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for blk in self.encoder.blocks[self.n_frozen:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        if not self.freeze:
            for p in self.encoder.norm.parameters():
                p.requires_grad_(True)
        d = 1024 * len(taps)
        self.proj = nn.Sequential(nn.Conv2d(d, 256, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.up = nn.ModuleList([ConvBlock(256, 128), ConvBlock(128, 96),
                                 ConvBlock(96, 64), ConvBlock(64, 48)])
        self.head = nn.Conv2d(48, N_CH, 1)

    def forward_features(self, x):
        """Frozen prefix runs under no_grad -- autograd would stop there anyway,
        so this just avoids storing activations we can never use."""
        e = self.encoder
        feats = []
        with torch.no_grad():
            x = e.patch_embed(x)
            x = e._pos_embed(x)
            x = e.patch_drop(x)
            x = e.norm_pre(x)
            for i in range(self.n_frozen):
                x = e.blocks[i](x)
                if i in self.taps:
                    feats.append(e.norm(x)[:, e.num_prefix_tokens:])
        for i in range(self.n_frozen, self.depth):
            x = e.blocks[i](x)
            if i in self.taps:
                feats.append(e.norm(x)[:, e.num_prefix_tokens:])
        return feats

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)                       # xraydino patch_embed is 3-channel
        feats = self.forward_features(x)
        B, n, _ = feats[0].shape
        g = int(n ** 0.5)
        f = torch.cat([t.transpose(1, 2).reshape(B, -1, g, g) for t in feats], 1).float()
        f = self.proj(f)
        for blk in self.up:
            f = blk(F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(f)


# ==========================================================================
# Loss
# ==========================================================================
def seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    dice = 1 - ((2 * inter + 1.0) / (p.sum(dims) + target.sum(dims) + 1.0))
    return bce + dice.mean()


# ==========================================================================
# EVALUATION -- FIXED CONTRACT. Do not change during the experiment loop.
#   primary: dice_heart  (channel PRIMARY_CH, per-image Dice @0.5, mean over
#            all val images of both views).  direction: higher.
#   Chamber metrics are logged only and MUST NOT drive selection.
# ==========================================================================
@torch.no_grad()
def evaluate(model, loader, device, views):
    from scipy.ndimage import distance_transform_edt
    model.eval()
    dice_per_ch = [[] for _ in range(N_CH)]
    iou_heart, dice_by_view = [], {"frontal": [], "lateral": []}
    hd95 = []

    for img, packed, idx in loader:
        img = img.to(device, non_blocking=True).float().unsqueeze(1) / 255.0
        tgt = unpack_masks(packed.to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img)
        pred = (torch.sigmoid(logits.float()) > 0.5).float()

        inter = (pred * tgt).sum((2, 3))
        psum, tsum = pred.sum((2, 3)), tgt.sum((2, 3))
        d = (2 * inter) / (psum + tsum).clamp(min=1e-6)
        present = tsum > 0                                   # skip channels absent from GT
        u = (psum + tsum - inter).clamp(min=1e-6)
        for c in range(N_CH):
            dice_per_ch[c] += d[present[:, c], c].tolist()
        hp = present[:, PRIMARY_CH]
        iou_heart += (inter[hp, PRIMARY_CH] / u[hp, PRIMARY_CH]).tolist()
        for bi, j in enumerate(idx.tolist()):
            if not present[bi, PRIMARY_CH]:
                continue
            dice_by_view[views[j]].append(d[bi, PRIMARY_CH].item())

        pn = pred[:, PRIMARY_CH].bool().cpu().numpy()
        tn = tgt[:, PRIMARY_CH].bool().cpu().numpy()
        for bi in range(pn.shape[0]):
            if not tn[bi].any():
                continue
            if not pn[bi].any():
                hd95.append(float(IMG_SIZE))
                continue
            dt_t = distance_transform_edt(~tn[bi])
            dt_p = distance_transform_edt(~pn[bi])
            bp = pn[bi] ^ _erode(pn[bi])
            bt = tn[bi] ^ _erode(tn[bi])
            dists = np.concatenate([dt_t[bp], dt_p[bt]])
            hd95.append(float(np.percentile(dists, 95)) if dists.size else float(IMG_SIZE))

    mean = lambda a: float(np.mean(a)) if len(a) else float("nan")
    metrics = {
        "dice_frontal": mean(dice_by_view["frontal"]),
        "dice_lateral": mean(dice_by_view["lateral"]),
        "iou_heart": mean(iou_heart),
        "hd95_heart": mean(hd95),
        "dice_chambers": mean([mean(dice_per_ch[c]) for c in CHAMBER_CH]),
    }
    for c, nm in enumerate(HEART_NAMES):
        metrics[f"dice_{nm}"] = mean(dice_per_ch[c])
    model.train()
    return {"dice_heart": mean(dice_per_ch[PRIMARY_CH]), "metrics": metrics}


def _erode(m):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m, np.ones((3, 3), bool), border_value=1)


# ==========================================================================
def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)

    arrays, train_idx, val_idx = make_splits()
    views = arrays[2]["views"]
    pats = arrays[2]["patients"]
    print(f"[data] train {len(train_idx)} imgs / {len(set(pats[i] for i in train_idx))} patients | "
          f"val {len(val_idx)} imgs / {len(set(pats[i] for i in val_idx))} patients")

    tl = DataLoader(HeartDataset(arrays, train_idx, train=True), batch_size=BATCH, shuffle=True,
                    num_workers=5, pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(HeartDataset(arrays, val_idx), batch_size=BATCH, shuffle=False,
                    num_workers=5, pin_memory=True, persistent_workers=True)

    model = HeartSeg().to(dev)
    model = model.to(memory_format=torch.channels_last)
    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    dec_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    params = enc_p + dec_p
    print(f"[model] trainable {sum(p.numel() for p in params)/1e6:.1f}M "
          f"(encoder {sum(p.numel() for p in enc_p)/1e6:.1f}M, decoder {sum(p.numel() for p in dec_p)/1e6:.1f}M)"
          f" / {sum(p.numel() for p in model.parameters())/1e6:.1f}M | unfrozen last {UNFREEZE_LAST} blocks")
    groups = [{"params": dec_p, "mult": 1.0}]
    if enc_p:
        groups.append({"params": enc_p, "mult": ENC_LR_MULT})
    opt = torch.optim.AdamW(groups, lr=LR, weight_decay=WD)

    budget = TIME_BUDGET_MINUTES * 60
    best, best_metrics, step = -1.0, None, 0
    t_train0 = time.time()
    eval_s = 0.0            # budget is training time; eval must not eat into it
    warmup = 200
    stop = False

    while not stop:
        for img, packed, _ in tl:
            elapsed = time.time() - t_train0 - eval_s
            if elapsed >= budget:
                stop = True
                break
            frac = elapsed / budget
            lr = LR * (step / warmup if step < warmup
                       else 0.5 * (1 + np.cos(np.pi * min(frac, 1.0))))
            for g in opt.param_groups:
                g["lr"] = lr * g["mult"]

            img = img.to(dev, non_blocking=True).float().unsqueeze(1) / 255.0
            tgt = unpack_masks(packed.to(dev, non_blocking=True))
            img, tgt = augment(img, tgt)
            img = img.to(memory_format=torch.channels_last)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = seg_loss(model(img).float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1

            if step % 100 == 0:
                print(f"[{elapsed:6.0f}s] step {step:6d} loss {loss.item():.4f} lr {lr:.2e}", flush=True)
            if step % EVAL_EVERY == 0:
                t_ev = time.time()
                r = evaluate(model, vl, dev, views)
                eval_s += time.time() - t_ev
                print(f"[eval] step {step} dice_heart {r['dice_heart']:.4f} "
                      f"(F {r['metrics']['dice_frontal']:.4f} / L {r['metrics']['dice_lateral']:.4f})", flush=True)
                torch.save({"model": model.state_dict(), "step": step}, f"{OUT_DIR}/checkpoints/last.pth")
                if r["dice_heart"] > best:
                    best, best_metrics = r["dice_heart"], r["metrics"]
                    torch.save({"model": model.state_dict(), "step": step, "metrics": r},
                               f"{OUT_DIR}/checkpoints/best.pth")

    train_s = time.time() - t_train0 - eval_s
    r = evaluate(model, vl, dev, views)
    torch.save({"model": model.state_dict(), "step": step}, f"{OUT_DIR}/checkpoints/last.pth")
    if r["dice_heart"] > best:
        best, best_metrics = r["dice_heart"], r["metrics"]
        torch.save({"model": model.state_dict(), "step": step, "metrics": r},
                   f"{OUT_DIR}/checkpoints/best.pth")

    peak = torch.cuda.max_memory_allocated() / 1e6
    print("\n---")
    print(f"dice_heart:      {best:.4f}\n")
    print("metrics:")
    for k, v in best_metrics.items():
        print(f"{k+':':24s}{v:.4f}")
    print(f"\ntraining_seconds: {train_s:.1f}")
    print(f"total_seconds:    {time.time()-t_start:.1f}")
    print(f"peak_vram_mb:     {peak:.1f}")
    with open(f"{OUT_DIR}/best_metrics.json", "w") as f:
        json.dump({"dice_heart": best, "metrics": best_metrics}, f, indent=1)


if __name__ == "__main__":
    main()
