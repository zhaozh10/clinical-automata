"""
ISIC 2019 Skin Lesion Classification — Training & Evaluation.
Primary metric: mel_auc (one-vs-rest AUC for melanoma, higher is better).
Clinical priority: maximize melanoma sensitivity, minimize melanoma→nevus confusion.
"""

import os
import sys
import time
import json
import csv
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

import timm
from timm.data.mixup import Mixup

from prepare import TIME_BUDGET_MINUTES, DATA_DIR
from data import (
    build_dataloaders, get_class_weights, CLASSES, NUM_CLASSES, CLASS_TO_IDX,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "projects", "20260407_00")

def _next_run_iter():
    i = 1
    while os.path.exists(os.path.join(PROJECT_DIR, f"run{i}")):
        i += 1
    return i

RUN_ITER = _next_run_iter()
RUN_DIR = os.path.join(PROJECT_DIR, f"run{RUN_ITER}")
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 48
LR = 1.5e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
NUM_EPOCHS = 20
BACKBONE = "convnext_base.fb_in22k_ft_in1k"
LABEL_SMOOTHING = 0.05

# EMA decay for model averaging
EMA_DECAY = 0.999

NUM_GPUS = 1
MEL_IDX = CLASS_TO_IDX["MEL"]
NV_IDX = CLASS_TO_IDX["NV"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model():
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


# ---------------------------------------------------------------------------
# Loss — weighted CE with melanoma emphasis
# ---------------------------------------------------------------------------

def build_criterion(train_samples):
    weights = get_class_weights(train_samples)
    weights[MEL_IDX] *= 1.5
    return nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=LABEL_SMOOTHING)


# Soft target CE for mixup
def soft_target_ce(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


# ---------------------------------------------------------------------------
# Evaluation (FIXED — do not change during experiment loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []

    for imgs, labels, img_ids in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy().tolist())
        all_ids.extend(img_ids)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    known_mask = all_labels >= 0
    probs_known = all_probs[known_mask]
    labels_known = all_labels[known_mask]

    mel_binary = (labels_known == MEL_IDX).astype(int)
    mel_probs = probs_known[:, MEL_IDX]
    try:
        mel_auc = roc_auc_score(mel_binary, mel_probs)
    except ValueError:
        mel_auc = 0.0

    try:
        n_present = len(set(labels_known))
        if n_present >= 2:
            overall_auc = roc_auc_score(
                labels_known, probs_known, multi_class="ovr", average="macro",
                labels=list(range(NUM_CLASSES))
            )
        else:
            overall_auc = 0.0
    except ValueError:
        overall_auc = 0.0

    preds_known = probs_known.argmax(axis=1)
    mel_true_mask = labels_known == MEL_IDX
    n_mel = mel_true_mask.sum()

    mel_sensitivity = (preds_known[mel_true_mask] == MEL_IDX).sum() / n_mel if n_mel > 0 else 0.0
    non_mel_mask = ~mel_true_mask
    n_non_mel = non_mel_mask.sum()
    mel_specificity = (preds_known[non_mel_mask] != MEL_IDX).sum() / n_non_mel if n_non_mel > 0 else 0.0
    mel_to_nv_rate = (preds_known[mel_true_mask] == NV_IDX).sum() / n_mel if n_mel > 0 else 0.0
    macro_f1 = f1_score(labels_known, preds_known, average="macro", zero_division=0)

    results = {
        "mel_auc": float(mel_auc),
        "metrics": {
            "mel_auc": float(mel_auc),
            "mel_sensitivity": float(mel_sensitivity),
            "mel_specificity": float(mel_specificity),
            "mel_to_nv_rate": float(mel_to_nv_rate),
            "overall_auc": float(overall_auc),
            "macro_f1": float(macro_f1),
        },
    }
    per_sample = {
        "ids": all_ids,
        "labels": all_labels.tolist(),
        "probs": all_probs.tolist(),
        "preds": all_probs.argmax(axis=1).tolist(),
    }
    return results, per_sample


def save_test_predictions(per_sample, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image_id", "ground_truth", "predicted_class", "predicted_label"]
        header += [f"prob_{c}" for c in CLASSES]
        writer.writerow(header)
        for i, img_id in enumerate(per_sample["ids"]):
            gt = per_sample["labels"][i]
            pred_idx = per_sample["preds"][i]
            gt_name = CLASSES[gt] if 0 <= gt < NUM_CLASSES else "UNK"
            pred_name = CLASSES[pred_idx] if 0 <= pred_idx < NUM_CLASSES else "UNK"
            row = [img_id, gt_name, pred_name, pred_idx]
            row += [f"{p:.6f}" for p in per_sample["probs"][i]]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Training with CutMix + EMA
# ---------------------------------------------------------------------------

def train():
    print(f"=== ISIC 2019 Skin Lesion Classification ===")
    print(f"Run: {RUN_ITER} | Backbone: {BACKBONE}")
    print(f"Device: {DEVICE} | GPUs: {NUM_GPUS} | Batch: {BATCH_SIZE}")
    print(f"Time budget: {TIME_BUDGET_MINUTES} min")
    print(f"LR: {LR} | EMA: {EMA_DECAY} | CutMix + Mixup")
    print()

    # Data
    train_loader, val_loader, test_loader, train_samples, val_samples, test_samples = \
        build_dataloaders(batch_size=BATCH_SIZE)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # Model
    model = build_model()
    ema = ModelEMA(model, decay=EMA_DECAY)

    # CutMix/Mixup
    mixup_fn = Mixup(
        mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5,
        switch_prob=0.5, mode="batch", num_classes=NUM_CLASSES
    )

    criterion_hard = build_criterion(train_samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    scaler = GradScaler("cuda")
    best_mel_auc = 0.0
    best_epoch = -1
    start_time = time.time()

    print(f"Starting training...")
    for epoch in range(NUM_EPOCHS):
        elapsed = (time.time() - start_time) / 60.0
        if elapsed >= TIME_BUDGET_MINUTES:
            print(f"Time budget reached at epoch {epoch} ({elapsed:.1f} min)")
            break

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Apply CutMix/Mixup
            imgs_mixed, labels_mixed = mixup_fn(imgs, labels)

            with torch.amp.autocast("cuda"):
                logits = model(imgs_mixed)
                loss = soft_target_ce(logits, labels_mixed)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1
            if (time.time() - start_time) / 60.0 >= TIME_BUDGET_MINUTES:
                break

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate EMA model
        val_results, _ = evaluate(ema.ema, val_loader)
        mel_auc = val_results["mel_auc"]
        metrics = val_results["metrics"]

        elapsed = (time.time() - start_time) / 60.0
        print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | mel_auc={mel_auc:.4f} | "
              f"mel_sens={metrics['mel_sensitivity']:.4f} | mel→nv={metrics['mel_to_nv_rate']:.4f} | "
              f"ovr_auc={metrics['overall_auc']:.4f} | f1={metrics['macro_f1']:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}min")

        state = {
            "epoch": epoch,
            "model_state_dict": ema.state_dict(),
            "mel_auc": mel_auc,
            "metrics": metrics,
        }
        torch.save(state, os.path.join(CKPT_DIR, "last.pth"))
        if mel_auc > best_mel_auc:
            best_mel_auc = mel_auc
            best_epoch = epoch
            torch.save(state, os.path.join(CKPT_DIR, "best.pth"))
            print(f"  >> New best mel_auc: {best_mel_auc:.4f} at epoch {epoch+1}")

    # --- Final results ---
    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    best_ckpt = torch.load(os.path.join(CKPT_DIR, "best.pth"), weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_results, _ = evaluate(model, val_loader)
    metrics = val_results["metrics"]

    print("\nRunning test evaluation...")
    test_results, test_per_sample = evaluate(model, test_loader)
    test_metrics = test_results["metrics"]
    save_test_predictions(test_per_sample, os.path.join(RUN_DIR, "test_predictions.csv"))

    total_time = time.time() - start_time
    print()
    print("---")
    print(f"mel_auc:      {val_results['mel_auc']:.4f}")
    print()
    print("metrics_summary:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print()
    print("test_metrics_summary:")
    for k, v in test_metrics.items():
        print(f"  test_{k:16s}: {v:.4f}")
    print()
    print(f"training_seconds: {total_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"best_epoch:       {best_epoch + 1}")
    print("---")


if __name__ == "__main__":
    log_path = os.path.join(RUN_DIR, "run.log")
    import sys as _sys
    _log_file = open(log_path, "w")

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    _sys.stdout = _Tee(_sys.stdout, _log_file)
    _sys.stderr = _Tee(_sys.stderr, _log_file)

    train()
