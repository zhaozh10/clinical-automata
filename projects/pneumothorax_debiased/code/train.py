"""Pneumothorax classification on NIH ChestX-ray14 (512x512) — confounder-aware.

CLINICAL FRAMING
----------------
The chest drain / chest tube is a strong *confounder* for pneumothorax in this
dataset: many pneumothorax-positive images were taken after a drain was placed,
so a naive classifier can score well just by detecting drains ("drain present ->
pneumothorax").  Such a model fails exactly where it matters — an *undrained*
pneumothorax (which still needs to be caught) and a *resolved* pneumothorax that
still has a drain in place (which should not be flagged).

We therefore (a) keep model selection on a confounder-robust primary metric and
(b) report drain-stratified diagnostics every run.

PRIMARY METRIC (fixed for the whole experiment; do not change)
--------------------------------------------------------------
    auroc_drainfree : AUROC for pneumothorax computed on validation images that
                      have NO chest drain.  A model relying on the drain shortcut
                      cannot do well here; a model with genuine pneumothorax
                      features can.   metric_direction = "higher".

Run 1 = naive ERM baseline (ImageNet-pretrained backbone, class-balanced sampler,
plain BCE, no drain debiasing) — kept as the reference point.  Later runs add
debiasing (group-balanced sampling, adversarial drain unlearning, etc.).
"""

import os
import sys
import json
import time
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score
import timm
from timm.utils import ModelEmaV3

from prepare import TIME_BUDGET_MINUTES
import data as datamod

# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #
PROJECT_DIR = "projects/20260512_13"
RUN_NAME = "run29"
RUN_DIR = os.path.join(PROJECT_DIR, RUN_NAME)
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")

SEED = 42

# --- evaluation contract (FROZEN) ---
PRIMARY_METRIC = "auroc_drainfree"
METRIC_DIRECTION = "higher"          # larger is better
SPEC_TARGET = 0.90                   # operating point for reported sensitivity (drain-free subset)

# --- model / optim (tunable across runs) ---
# NOTE: this is a heavily shared 6-core node; nn.DataParallel's per-step Python
# threading is starved by CPU contention and runs ~4x SLOWER than a single GPU
# here, so we deliberately train on ONE GPU.  Images are served from a uint8
# memmap so the few dataloader workers can keep the GPU fed.
# run 15: bump to ConvNeXt-Small (~50M params vs Tiny's ~28M) — every recipe
# tweak on Tiny has stalled at ~0.85; the bottleneck may be backbone capacity for
# genuine pneumothorax features.  Small is ~1.6-1.8x slower so STEP_TIME_EST and
# EVAL_BATCH_SIZE are adjusted (cosine horizon ~ wall_budget / step_time).
BACKBONE = "convnext_small.fb_in22k_ft_in1k"
DROP_RATE = 0.2
# run 16: ConvNeXt-Small (run 15, the new best 0.8579) overfits hard by ep4
# (val 0.858 -> 0.847 -> 0.824); bump stochastic depth 0.3 -> 0.45 to regularize
# the bigger backbone and (hopefully) flatten that post-peak decline / raise the peak.
DROP_PATH_RATE = 0.45                # stochastic depth (strong regularizer for ConvNeXt)
FREEZE_STAGES = 2                    # freeze stem + first N ConvNeXt stages (cuts trainable capacity)
IMG_SIZE = datamod.IMG_SIZE          # 512
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 96
NUM_WORKERS = 4
# run 21 tried LR 6e-5 (slower, uniformly worse).  run 23 brackets the other side
# of run 18's 8e-5: try 1.0e-4.  If 1e-4 helps, the peak shifts earlier with more
# headroom; if it hurts, LR is at/near optimum at 8e-5 and we move on to other knobs.
LR = 1e-4
LR_MIN_FRAC = 0.05                   # cosine never decays below this fraction of peak LR
WEIGHT_DECAY = 0.05
WARMUP_FRAC = 0.05
STEP_TIME_EST = 0.42                 # s/step estimate (Small, mixed frozen/unfrozen) -> sets the cosine horizon
MAX_EPOCHS = 100                     # time budget will cut this off
EVAL_EVERY_STEPS = 99999             # >> steps/epoch -> validate per epoch only (eval ~40s; keep it cheap)
GRAD_CLIP = 1.0
POS_WEIGHT = 1.0
# --- runs 2-3: debias via group-balanced sampling so the chest-drain label is
# NON-predictive of pneumothorax inside a batch (drain ⊥ PTX); run 3 added
# regularization to curb overfitting (best so far: auroc_drainfree=0.822).
SAMPLER_KIND = "group_balanced"
POS_FRAC = 0.30                      # only used by label_balanced sampler
GB_DRAIN_POS_FRAC = 0.40             # P(drain+) per batch
GB_PTX_POS_FRAC = 0.30               # P(PTX+ | drain stratum) per batch  -> P(PTX|drain+)=P(PTX|drain-)
# --- run 4: also scrub chest-drain information from the *representation* via a
# domain-adversarial head (gradient-reversal): the backbone is trained so its
# features can predict pneumothorax but NOT the chest drain.
ADV_DRAIN = True
ADV_HID = 256                        # adversary MLP hidden width
# run 17: lambda_max 0.5 -> 0.8 helped BOTH auroc_drainfree (0.866->0.873) and the
# drain_bias_gap.  run 18: push further, lambda_max 0.8 -> 1.2 — does more
# de-confounding keep helping, or does it start fighting the PTX task too hard?
ADV_LAMBDA_MAX = 1.2                 # max gradient-reversal strength
ADV_WARMUP_FRAC = 0.3                # ramp lambda 0 -> max over this fraction of est_total_steps
# --- run 6: EMA of the weights — runs 3-5 peak around epoch 2 then oscillate
# (~0.82-0.83); a Polyak/EMA average over the trajectory tends to give a more
# stable, slightly better checkpoint.  Eval & checkpointing use the EMA weights.
USE_EMA = True
EMA_DECAY = 0.999                    # ~1000-step effective window (~0.8 epoch here)
# run 27: multi-checkpoint averaging on EMA snapshots (run 26 same idea but had a
# tuple-unpacking bug; this is the corrected re-run).  Save the EMA state at end
# of every epoch; after training, average the top-K snapshots (by val
# auroc_drainfree).  Use averaged weights if their val beats single-best.
CKPT_AVG_K = 3                       # number of top-by-val snapshots to average (0 = disable)
# run 29: horizontal-flip test-time augmentation, applied ONLY at the final
# val+test eval after ckpt-avg picks weights.  Per-epoch val (during training)
# stays plain so the FROZEN evaluate() is called identically to prior runs.
# Wrapping the model with a TTA forward changes only the model's predictions,
# not the metric.  If TTA val > no-TTA val, use TTA as the reported best.
TTA_FINAL = True
# --- run 9: "freeze more, then unfreeze" — keep stem + first 2 stages frozen
# early, then unfreeze them partway through with a small LR so the model can
# adapt low-level features to chest X-rays (a real ImageNet -> CXR domain gap).
LATE_UNFREEZE = True
UNFREEZE_AT_FRAC = 0.25              # unfreeze once global_step exceeds this fraction of est_total_steps
UNFREEZE_LR_MULT = 0.25             # LR for the newly-unfrozen layers = this * LR

TRAIN_WALL_BUDGET_S = TIME_BUDGET_MINUTES * 60   # wall-clock budget for the training phase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def threshold_at_specificity(y_true, y_score, target_spec: float) -> float:
    """Largest decision threshold whose specificity >= target_spec (on this set)."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    ok = spec >= target_spec
    if not ok.any():
        return float(np.max(y_score))
    # among thresholds satisfying the spec constraint, pick the one with highest sensitivity (tpr)
    idx = np.argmax(np.where(ok, tpr, -1.0))
    t = thr[idx]
    # roc_curve can return inf as the first threshold
    if not np.isfinite(t):
        t = float(np.max(y_score)) + 1e-6
    return float(t)


def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


# --------------------------------------------------------------------------- #
# Model  (+ optional domain-adversarial chest-drain head)
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)


class TTAHflipWrapper(nn.Module):
    """run 29: average the model's PTX logits over the original input and its horizontal flip.
    The wrapped model's predict-time behaviour improves; the metric definition is unchanged."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        l1 = self.model(x)
        l2 = self.model(torch.flip(x, dims=[-1]))
        return (l1 + l2) / 2.0


class PTXModel(nn.Module):
    """Pretrained backbone + binary pneumothorax head.

    If ``adv_drain`` is set, an auxiliary chest-drain classifier is attached to
    the pooled features through a gradient-reversal layer: the adversary head
    learns to predict the drain, while the backbone is pushed to make the drain
    *un*-predictable from its features (domain-adversarial training / DANN).
    """

    def __init__(self, backbone=BACKBONE, drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE,
                 freeze_stages=FREEZE_STAGES, adv_drain=ADV_DRAIN, adv_hid=ADV_HID):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0,
                                          drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                          global_pool="avg")
        self.feat_dim = self.backbone.num_features
        self.head = nn.Linear(self.feat_dim, 1)
        self.adv_drain = adv_drain
        if adv_drain:
            self.adv_head = nn.Sequential(
                nn.Linear(self.feat_dim, adv_hid), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(adv_hid, 1),
            )
        self.frozen_modules = []
        if freeze_stages > 0:
            self.frozen_modules.append(self.backbone.stem)
            for i in range(min(freeze_stages, len(self.backbone.stages))):
                self.frozen_modules.append(self.backbone.stages[i])
            for mod in self.frozen_modules:
                for p in mod.parameters():
                    p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for mod in self.frozen_modules:        # keep frozen submodules in eval mode
            mod.eval()
        return self

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)   # (B,) PTX logits

    def forward_adv(self, x, lambd: float):
        feat = self.backbone(x)
        ptx_logit = self.head(feat).squeeze(-1)
        drain_logit = self.adv_head(grad_reverse(feat, lambd)).squeeze(-1)
        return ptx_logit, drain_logit


# --------------------------------------------------------------------------- #
# Evaluation  (FROZEN — only the user may change this)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_probs(model, loader, n_rows: int) -> np.ndarray:
    model.eval()
    probs = np.full(n_rows, np.nan, dtype=np.float64)
    for batch in loader:
        x = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits = model(x)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        rows = batch["row"].numpy()
        probs[rows] = p
    assert not np.isnan(probs).any(), "some samples were not scored"
    return probs


def evaluate(model, loader, df: pd.DataFrame):
    """Return the evaluation-contract dict.

    primary_metric (auroc_drainfree) = AUROC for pneumothorax on the subset of
    `df` with no chest drain (drain==0); higher is better.
    """
    probs = predict_probs(model, loader, len(df))
    y = df["ptx"].values.astype(int)
    drain = df["drain"].values.astype(int)

    drainfree = drain == 0
    drainpos = drain == 1
    pn = y == 0                       # pneumothorax-negative

    auroc_drainfree = safe_auc(y[drainfree], probs[drainfree])    # <-- PRIMARY
    auroc_overall = safe_auc(y, probs)
    auroc_drainpos = safe_auc(y[drainpos], probs[drainpos])
    auprc_drainfree = safe_ap(y[drainfree], probs[drainfree])

    # operating point: threshold @ >= SPEC_TARGET specificity on the drain-free subset
    thr = threshold_at_specificity(y[drainfree], probs[drainfree], SPEC_TARGET)
    pred = (probs >= thr).astype(int)
    pos_df_mask = drainfree & (y == 1)
    sens_drainfree = float(pred[pos_df_mask].mean()) if pos_df_mask.any() else float("nan")
    f1 = float(f1_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")

    # interpretable shortcut-reliance diagnostic: how much a chest drain (on a
    # pneumothorax-NEGATIVE image) inflates the predicted PTX probability
    p_pn_drainpos = probs[pn & (drain == 1)]
    p_pn_drainneg = probs[pn & (drain == 0)]
    drain_bias_gap = (float(p_pn_drainpos.mean() - p_pn_drainneg.mean())
                      if len(p_pn_drainpos) and len(p_pn_drainneg) else float("nan"))

    metrics = {
        "auroc_drainfree": auroc_drainfree,
        "auroc_overall": auroc_overall,
        "auroc_drainpos": auroc_drainpos,
        "auprc_drainfree": auprc_drainfree,
        "sensitivity_drainfree@spec90": sens_drainfree,
        "drain_bias_gap": drain_bias_gap,
        "f1": f1,
        "threshold@spec90": thr,
    }
    return {PRIMARY_METRIC: auroc_drainfree, "metrics": metrics, "_probs": probs}


# --------------------------------------------------------------------------- #
# Checkpoint / prediction IO
# --------------------------------------------------------------------------- #
def save_ckpt(path, base_model, extra: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": base_model.state_dict(), **extra}, path)


def write_predictions_csv(path, df: pd.DataFrame, probs: np.ndarray, thr: float):
    out = pd.DataFrame({
        "image_id": df["image_id"].values,
        "patient_id": df["patient_id"].values,
        "split": df["split"].values,
        "ptx_label": df["ptx"].values.astype(int),
        "drain_label": df["drain"].values.astype(int),
        "drain_is_gt": df["drain_is_gt"].values.astype(int),
        "group": df["group"].map(datamod.GROUP_NAMES).values,
        "ptx_prob": np.round(probs, 6),
        "ptx_pred": (probs >= thr).astype(int),
        "threshold": thr,
    })
    out.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_lr_lambda(total_steps: int, warmup_steps: int, min_frac: float = 0.0):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_frac + (1.0 - min_frac) * cos
    return f


def adv_lambda_at(step: int, total_steps: int) -> float:
    """Ramp the gradient-reversal strength 0 -> ADV_LAMBDA_MAX over the first
    ADV_WARMUP_FRAC of training, then hold (DANN-style schedule)."""
    if not ADV_DRAIN:
        return 0.0
    p = min(1.0, step / max(1.0, ADV_WARMUP_FRAC * total_steps))
    return ADV_LAMBDA_MAX * (2.0 / (1.0 + math.exp(-5.0 * p)) - 1.0)


def main():
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    t_start = time.time()

    df = datamod.load_metadata()
    log("split summary:\n" + datamod.describe_splits(df))
    datamod.build_image_cache(df, workers=6)   # one-time; no-op if already built
    datamod.prewarm_cache(len(df))             # pull the ~29 GB image cache into RAM page-cache (startup)

    train_loader, val_loader, test_loader, tr_df, va_df, te_df = datamod.build_dataloaders(
        df, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=IMG_SIZE,
        sampler_kind=SAMPLER_KIND, pos_frac=POS_FRAC,
        gb_drain_pos_frac=GB_DRAIN_POS_FRAC, gb_ptx_pos_frac=GB_PTX_POS_FRAC,
        eval_batch_size=EVAL_BATCH_SIZE, use_cache=True,
    )
    steps_per_epoch = len(train_loader)
    # estimate how many optimizer steps fit in the wall budget so the cosine schedule
    # decays over (roughly) the whole run; a min-LR floor protects against under-estimation
    est_total_steps = max(steps_per_epoch, int(TRAIN_WALL_BUDGET_S / STEP_TIME_EST))
    warmup_steps = max(50, int(WARMUP_FRAC * steps_per_epoch * 4))
    log(f"steps/epoch={steps_per_epoch}  est_total_steps={est_total_steps}  warmup_steps={warmup_steps}  "
        f"adv_drain={ADV_DRAIN} adv_lambda_max={ADV_LAMBDA_MAX}")

    base_model = PTXModel().to(DEVICE, memory_format=torch.channels_last)
    model = base_model                                    # single GPU (see note above)
    n_all = sum(p.numel() for p in base_model.parameters())
    n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    log(f"model={BACKBONE} feat_dim={base_model.feat_dim} drop_path={DROP_PATH_RATE} freeze_stages={FREEZE_STAGES} "
        f"params={n_all/1e6:.1f}M trainable={n_trainable/1e6:.1f}M  GPUs_used=1/{torch.cuda.device_count()}")

    frozen_param_ids = {id(p) for mod in base_model.frozen_modules for p in mod.parameters()}
    trainable_params = [p for p in base_model.parameters() if id(p) not in frozen_param_ids]
    frozen_params = [p for mod in base_model.frozen_modules for p in mod.parameters()]
    base_cos = build_lr_lambda(est_total_steps, warmup_steps, min_frac=LR_MIN_FRAC)
    unfreeze_step = int(UNFREEZE_AT_FRAC * est_total_steps) if LATE_UNFREEZE else 10**9
    if LATE_UNFREEZE and frozen_params:
        # two param groups: g0 = always-trainable, g1 = frozen-then-unfrozen (LR 0 until unfreeze_step)
        optimizer = torch.optim.AdamW(
            [{"params": trainable_params, "lr": LR}, {"params": frozen_params, "lr": LR}],
            lr=LR, weight_decay=WEIGHT_DECAY)
        g1_cos = lambda s: 0.0 if s < unfreeze_step else UNFREEZE_LR_MULT * base_cos(s)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[base_cos, g1_cos])
        log(f"late-unfreeze: stem + {FREEZE_STAGES} stages unfreeze at step {unfreeze_step} "
            f"(~{UNFREEZE_AT_FRAC:.0%} of {est_total_steps}) with LR x{UNFREEZE_LR_MULT}")
    else:
        optimizer = torch.optim.AdamW(trainable_params or [p for p in base_model.parameters() if p.requires_grad],
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, base_cos)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ema = ModelEmaV3(base_model, decay=EMA_DECAY, use_warmup=True) if USE_EMA else None
    if ema is not None:
        log(f"EMA enabled (decay={EMA_DECAY}, warmup); eval/checkpoints use the EMA weights")
    unfrozen = not (LATE_UNFREEZE and frozen_params)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_metric = -math.inf if METRIC_DIRECTION == "higher" else math.inf
    best_epoch = -1
    best_step = -1
    history = []
    global_step = 0
    val_seconds = 0.0
    stop = False
    wall_start = time.time()

    def elapsed_training():
        return (time.time() - wall_start) - val_seconds

    def is_better(new, old):
        return new > old if METRIC_DIRECTION == "higher" else new < old

    def run_validation(epoch, tag=""):
        nonlocal best_metric, best_epoch, best_step, val_seconds
        v0 = time.time()
        eval_model = ema.module if ema is not None else model   # evaluate/checkpoint the EMA weights
        res = evaluate(eval_model, val_loader, va_df)
        m = res["metrics"]
        log(f"[val{tag}] epoch={epoch} step={global_step} {PRIMARY_METRIC}={res[PRIMARY_METRIC]:.4f} | "
            + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
        save_ckpt(os.path.join(CKPT_DIR, "last.pth"), eval_model,
                  {"epoch": epoch, "step": global_step, "val_metrics": m, "primary": res[PRIMARY_METRIC]})
        if CKPT_AVG_K > 0:
            save_ckpt(os.path.join(CKPT_DIR, f"ep{epoch}.pth"), eval_model,
                      {"epoch": epoch, "step": global_step, "val_metrics": m, "primary": res[PRIMARY_METRIC]})
        improved = is_better(res[PRIMARY_METRIC], best_metric) and not math.isnan(res[PRIMARY_METRIC])
        if improved:
            best_metric, best_epoch, best_step = res[PRIMARY_METRIC], epoch, global_step
            save_ckpt(os.path.join(CKPT_DIR, "best.pth"), eval_model,
                      {"epoch": epoch, "step": global_step, "val_metrics": m, "primary": res[PRIMARY_METRIC]})
            log(f"  -> new best {PRIMARY_METRIC}={best_metric:.4f} (epoch {epoch}, step {global_step})")
        history.append({"epoch": epoch, "step": global_step, "train_seconds": round(elapsed_training(), 1),
                        **{f"val_{k}": v for k, v in m.items()}})
        with open(os.path.join(RUN_DIR, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)
        val_seconds += time.time() - v0
        return res

    log(f"=== training starts: wall budget {TRAIN_WALL_BUDGET_S/60:.0f} min, eval every {EVAL_EVERY_STEPS} steps ===")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running, run_adv, run_dacc, n_run = 0.0, 0.0, 0.0, 0
        for it, batch in enumerate(train_loader):
            if not unfrozen and global_step >= unfreeze_step:
                for mod in base_model.frozen_modules:
                    for p in mod.parameters():
                        p.requires_grad_(True)
                    mod.train()
                base_model.frozen_modules = []
                unfrozen = True
                ntr = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6
                log(f"  >>> unfroze backbone stem + early stages at step {global_step} (trainable now {ntr:.1f}M)")
            x = batch["image"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            y = batch["ptx"].to(DEVICE, non_blocking=True)
            d = batch["drain"].to(DEVICE, non_blocking=True)
            lam = adv_lambda_at(global_step, est_total_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                if ADV_DRAIN:
                    logits, drain_logits = model.forward_adv(x, lam)
                    loss_ptx = loss_fn(logits, y)
                    loss_adv = F.binary_cross_entropy_with_logits(drain_logits, d)
                    loss = loss_ptx + loss_adv
                else:
                    logits = model(x)
                    loss = loss_ptx = loss_fn(logits, y)
                    loss_adv = torch.zeros((), device=x.device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update(model, step=global_step)
            running += loss_ptx.item(); run_adv += loss_adv.item(); n_run += 1
            if ADV_DRAIN:
                run_dacc += ((drain_logits.detach() > 0).float() == d).float().mean().item()
            if global_step % 200 == 0:
                log(f"  epoch {epoch} it {it+1}/{steps_per_epoch} step {global_step} "
                    f"loss_ptx={running/max(n_run,1):.4f} loss_adv={run_adv/max(n_run,1):.4f} "
                    f"adv_acc={run_dacc/max(n_run,1):.3f} lam={lam:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"train_s={elapsed_training():.0f}")
                running, run_adv, run_dacc, n_run = 0.0, 0.0, 0.0, 0
            if global_step % EVAL_EVERY_STEPS == 0:
                run_validation(epoch, tag="@step")
                model.train()
            if elapsed_training() >= TRAIN_WALL_BUDGET_S:
                stop = True
                break
        log(f"epoch {epoch} pass done (cum train {elapsed_training():.0f}s, step {global_step})")
        if global_step % EVAL_EVERY_STEPS != 0:        # avoid an immediate duplicate eval
            run_validation(epoch, tag="@epoch")
            model.train()
        if stop or elapsed_training() >= TRAIN_WALL_BUDGET_S:
            log("training wall budget reached — stopping.")
            break
    train_seconds = elapsed_training()

    # ----------------------------------------------------------------------- #
    # Final: report best validation metrics; then test best.pth (reporting only)
    # ----------------------------------------------------------------------- #
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE == "cuda" else 0.0

    best_path = os.path.join(CKPT_DIR, "best.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        base_model.load_state_dict(ckpt["model"])
        best_val = ckpt["val_metrics"]
        best_primary = ckpt["primary"]
        best_ep = ckpt.get("epoch", -1)
        best_st = ckpt.get("step", -1)
    else:  # never validated (shouldn't happen) — fall back to last state
        best_val = {k: v for k, v in (history[-1] if history else {}).items() if isinstance(v, (int, float))}
        best_primary = float("nan")
        best_ep, best_st = -1, -1

    # --- multi-checkpoint averaging on EMA snapshots (run 27+) ---
    avg_val = None
    avg_primary = float("nan")
    single_best_primary = best_primary
    single_best_val = dict(best_val)
    single_best_ep = best_ep
    if CKPT_AVG_K > 0:
        import glob
        snap_paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ep*.pth")))
        if len(snap_paths) >= 2:
            snaps = []
            for p in snap_paths:
                c = torch.load(p, map_location="cpu")
                snaps.append((c["primary"], c["epoch"], c["model"]))
            K = min(CKPT_AVG_K, len(snaps))
            snaps.sort(key=lambda x: (-x[0]) if not math.isnan(x[0]) else 1.0)   # higher primary first
            top = snaps[:K]
            log(f"ckpt-avg: averaging top-{K} EMA snapshots by val {PRIMARY_METRIC}: "
                + ", ".join(f"ep{e}={p:.4f}" for p, e, _ in top))
            top_sds = [s for _, _, s in top]
            avg_sd = {}
            for k in top_sds[0].keys():
                t0 = top_sds[0][k]
                if torch.is_tensor(t0) and t0.is_floating_point():
                    avg_sd[k] = sum(sd[k].float() for sd in top_sds) / K
                    avg_sd[k] = avg_sd[k].to(t0.dtype)
                else:
                    avg_sd[k] = t0
            base_model.load_state_dict(avg_sd)
            avg_res = evaluate(model, val_loader, va_df)
            avg_primary = avg_res[PRIMARY_METRIC]
            avg_val = avg_res["metrics"]
            log(f"ckpt-avg val {PRIMARY_METRIC}={avg_primary:.4f} | "
                + " ".join(f"{k}={v:.4f}" for k, v in avg_val.items()))
            if not math.isnan(avg_primary) and is_better(avg_primary, best_primary):
                log(f"  ckpt-avg > single-best ({avg_primary:.4f} > {best_primary:.4f}) — using averaged weights for TEST")
                best_primary = avg_primary
                best_val = avg_val
                best_ep = f"avg(top{K})"
            else:
                log(f"  ckpt-avg <= single-best ({avg_primary:.4f} vs {best_primary:.4f}) — reloading single-best for TEST")
                ckpt = torch.load(best_path, map_location=DEVICE)
                base_model.load_state_dict(ckpt["model"])

    # --- run 29: TTA-augmented final val eval on the chosen weights ---
    no_tta_primary = best_primary
    no_tta_val = dict(best_val)
    tta_primary = float("nan")
    tta_val = None
    if TTA_FINAL:
        tta_model = TTAHflipWrapper(model)
        tta_res = evaluate(tta_model, val_loader, va_df)
        tta_primary = tta_res[PRIMARY_METRIC]
        tta_val = tta_res["metrics"]
        log(f"TTA-hflip val {PRIMARY_METRIC}={tta_primary:.4f} (no-TTA was {no_tta_primary:.4f}) | "
            + " ".join(f"{k}={v:.4f}" for k, v in tta_val.items()))
        if not math.isnan(tta_primary) and is_better(tta_primary, best_primary):
            log(f"  TTA-hflip > chosen-no-TTA ({tta_primary:.4f} > {best_primary:.4f}) — reporting TTA val/test as best")
            best_primary = tta_primary
            best_val = tta_val
            best_ep = (best_ep + "+tta") if not isinstance(best_ep, int) else f"ep{best_ep}+tta"
        else:
            log(f"  TTA-hflip <= chosen-no-TTA ({tta_primary:.4f} vs {best_primary:.4f}) — keeping no-TTA")

    total_seconds = time.time() - t_start
    print("\n---")
    print(f"{PRIMARY_METRIC}:      {best_primary:.4f}   (best epoch {best_ep}, step {best_st})")
    print(f"\nmetrics_summary:")
    for k, v in best_val.items():
        print(f"{k}: {v:.4f}")
    print(f"\ntraining_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")

    # ---- TEST (reporting only; never used for selection / tuning) ----
    # run 29: if TTA won val, use TTA-wrapped model for test inference too
    use_tta_for_test = TTA_FINAL and not math.isnan(tta_primary) and is_better(tta_primary, no_tta_primary)
    test_model = TTAHflipWrapper(model) if use_tta_for_test else model
    log(f"running TEST evaluation with {'TTA-wrapped ' if use_tta_for_test else ''}best weights ...")
    test_res = evaluate(test_model, test_loader, te_df)
    tm = test_res["metrics"]
    write_predictions_csv(os.path.join(RUN_DIR, "test_predictions.csv"), te_df,
                          test_res["_probs"], tm["threshold@spec90"])
    # also save val predictions for the chosen model (handy for error analysis; not used for selection)
    val_res = evaluate(test_model, val_loader, va_df)
    write_predictions_csv(os.path.join(RUN_DIR, "val_predictions.csv"), va_df,
                          val_res["_probs"], val_res["metrics"]["threshold@spec90"])
    with open(os.path.join(RUN_DIR, "summary.json"), "w") as fh:
        json.dump({"best_epoch": best_ep, "best_step": best_st, "best_primary": best_primary,
                   "val_metrics": best_val, "test_metrics": tm,
                   "single_best_primary": single_best_primary, "single_best_val": single_best_val,
                   "single_best_epoch": single_best_ep, "ckpt_avg_primary": avg_primary, "ckpt_avg_val": avg_val,
                   "ckpt_avg_k": CKPT_AVG_K,
                   "no_tta_primary": no_tta_primary, "no_tta_val": no_tta_val,
                   "tta_primary": tta_primary, "tta_val": tta_val, "tta_used_for_test": use_tta_for_test,
                   "training_seconds": train_seconds,
                   "total_seconds": total_seconds, "peak_vram_mb": peak_vram_mb,
                   "backbone": BACKBONE, "sampler": SAMPLER_KIND, "lr": LR, "drop_rate": DROP_RATE,
                   "drop_path_rate": DROP_PATH_RATE, "freeze_stages": FREEZE_STAGES,
                   "gb_drain_pos_frac": GB_DRAIN_POS_FRAC, "gb_ptx_pos_frac": GB_PTX_POS_FRAC,
                   "adv_drain": ADV_DRAIN, "adv_lambda_max": ADV_LAMBDA_MAX, "adv_warmup_frac": ADV_WARMUP_FRAC,
                   "use_ema": USE_EMA, "ema_decay": EMA_DECAY,
                   "late_unfreeze": LATE_UNFREEZE, "unfreeze_at_frac": UNFREEZE_AT_FRAC,
                   "unfreeze_lr_mult": UNFREEZE_LR_MULT},
                  fh, indent=2)
    print("\n=== TEST (reporting only) ===")
    print(f"test_{PRIMARY_METRIC}: {tm['auroc_drainfree']:.4f}")
    for k, v in tm.items():
        print(f"test_{k}: {v:.4f}")
    log("done.")


if __name__ == "__main__":
    main()
