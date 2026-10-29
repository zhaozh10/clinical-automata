"""
train.py — Binary chest drain classifier (present/absent) on NIH ChestX-ray14.
Backbone: timm pretrained ConvNeXt-Tiny.
Emphasis on high recall (sensitivity) as primary metric.
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import timm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

from prepare import TIME_BUDGET_MINUTES, MAX_LOOPS, EXT_DATA_DIR
from data import get_dataloaders

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = "projects/20260408_15"
RUN_TAG = "run6"
RUN_DIR = os.path.join(PROJECT_DIR, RUN_TAG)
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 3e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Focal loss parameters (gamma > 0 downweights easy negatives, helping recall)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.6  # weight for positive class


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=1)
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Evaluation (FIXED — do not change during experiment loop)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    all_fnames = []

    for imgs, labels, fnames in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        all_fnames.extend(fnames)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # AUC
    auc = roc_auc_score(all_labels, all_probs)

    # Find threshold that maximizes sensitivity while keeping specificity >= 0.5
    # (Youden-style but biased toward recall)
    best_sens = 0.0
    best_thresh = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (all_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Maximize sensitivity, but require minimum specificity of 0.4
        if sens > best_sens and spec >= 0.4:
            best_sens = sens
            best_thresh = t

    preds = (all_probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(all_labels, preds)

    return {
        "sensitivity": sensitivity,
        "metrics": {
            "auc": auc,
            "f1": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "threshold": best_thresh,
        },
        "probs": all_probs,
        "labels": all_labels,
        "fnames": all_fnames,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    print(f"Device: {DEVICE}")
    print(f"Time budget: {TIME_BUDGET_MINUTES} min")
    print(f"Run dir: {RUN_DIR}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Model
    model = build_model()
    criterion = FocalBCELoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    scaler = GradScaler()

    best_metric = -1.0
    best_auc = -1.0
    best_epoch = -1
    start_time = time.time()

    epoch = 0
    while True:
        elapsed = (time.time() - start_time) / 60.0
        if elapsed >= TIME_BUDGET_MINUTES:
            break

        epoch += 1
        model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, labels, _ in train_loader:
            # Check time budget mid-epoch
            if (time.time() - start_time) / 60.0 >= TIME_BUDGET_MINUTES:
                break

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).float()

            optimizer.zero_grad()
            with autocast():
                logits = model(imgs).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if n_batches == 0:
            break

        avg_loss = running_loss / n_batches
        elapsed = (time.time() - start_time) / 60.0

        # Validate
        val_result = evaluate(model, val_loader)
        val_sens = val_result["sensitivity"]
        val_metrics = val_result["metrics"]

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"val_sens={val_sens:.4f} | val_auc={val_metrics['auc']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f} | val_spec={val_metrics['specificity']:.4f} | "
              f"thresh={val_metrics['threshold']:.2f} | "
              f"elapsed={elapsed:.1f}min")

        # Save last
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last.pth"))

        # Save best (primary: sensitivity, tiebreaker: AUC)
        val_auc = val_metrics["auc"]
        if val_sens > best_metric or (val_sens == best_metric and val_auc > best_auc):
            best_metric = val_sens
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pth"))
            print(f"  -> New best sensitivity: {best_metric:.4f} (auc={val_auc:.4f}) at epoch {epoch}")

    total_time = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # ---------------------------------------------------------------------------
    # Load best and report val results
    # ---------------------------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best.pth"), map_location=DEVICE))
    val_result = evaluate(model, val_loader)

    print("\n---")
    print(f"sensitivity:      {val_result['sensitivity']:.4f}")
    print(f"\nmetrics_summary:")
    for k, v in val_result["metrics"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\ntraining_seconds: {total_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"best_epoch:       {best_epoch}")

    # ---------------------------------------------------------------------------
    # Test set inference
    # ---------------------------------------------------------------------------
    test_result = evaluate(model, test_loader)
    print(f"\ntest_sensitivity: {test_result['sensitivity']:.4f}")
    print("test_metrics_summary:")
    for k, v in test_result["metrics"].items():
        print(f"  {k}: {v:.4f}")

    # Save test predictions CSV
    test_preds_df = pd.DataFrame({
        "filename": test_result["fnames"],
        "label": test_result["labels"].astype(int),
        "prob": test_result["probs"],
        "pred": (test_result["probs"] >= test_result["metrics"]["threshold"]).astype(int),
    })
    test_preds_df.to_csv(os.path.join(RUN_DIR, "test_predictions.csv"), index=False)
    print(f"\nTest predictions saved to {RUN_DIR}/test_predictions.csv")


if __name__ == "__main__":
    train()
