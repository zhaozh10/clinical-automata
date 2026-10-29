"""
train.py — Binary MEL vs NV classifier.
Primary metric: mel_sens_at_spec80 (sensitivity at 80% specificity).
"""

import os
import sys
import time
import csv
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import timm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve

from prepare import TIME_BUDGET_MINUTES, MAX_LOOPS, DATA_DIR, EXT_DATA_DIR
from data import build_dataloaders, CLASSES, IMG_SIZE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = "projects/20260408_12"
RUN_INDEX = 20  # incremented each experiment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

# Hyperparameters
BACKBONE = "convnext_large.fb_in22k_ft_in1k"
BATCH_SIZE = 24
LR = 5e-5
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 1
LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999
FOCAL_GAMMA = 1.0  # focal loss gamma; 0 = standard CE
CUTMIX_ALPHA = 1.0
MIXUP_ALPHA = 0.3
DROP_PATH_RATE = 0.2

# Primary metric
PRIMARY_METRIC = "mel_sens_at_spec80"
METRIC_DIRECTION = "higher"  # higher is better


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------
def cutmix_data(images, labels, alpha=1.0):
    """Apply CutMix augmentation. Returns mixed images and (labels_a, labels_b, lam)."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    # Random box
    W, H = images.size(3), images.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)  # adjust lambda for actual box size
    return images, labels, labels[index], lam


def cutmix_criterion(criterion, logits, labels_a, labels_b, lam):
    """Compute CutMix loss as weighted combination."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=2,
                              drop_path_rate=DROP_PATH_RATE)
    model = model.to(DEVICE)
    return model


# ---------------------------------------------------------------------------
# Evaluation (FIXED — do not modify during experiment loop)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    """
    Compute all metrics for binary MEL(1) vs NV(0).
    Returns dict with primary_metric and all secondary metrics.
    Also returns per-sample predictions for test CSV.
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []

    for images, labels, img_ids in loader:
        images = images.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(MEL)
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        all_ids.extend(img_ids)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # AUC
    mel_auc = roc_auc_score(all_labels, all_probs)

    # ROC curve for threshold-based metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    specificity = 1 - fpr

    # Sensitivity at 80% specificity
    idx_spec80 = np.where(specificity >= 0.80)[0]
    if len(idx_spec80) > 0:
        best_idx = idx_spec80[np.argmin(specificity[idx_spec80])]  # closest to 80%
        mel_sens_at_spec80 = tpr[best_idx]
        threshold_at_spec80 = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        mel_sens_at_spec80 = 0.0
        threshold_at_spec80 = 0.5

    # Optimal threshold (Youden)
    youden = tpr + specificity - 1
    best_youden_idx = np.argmax(youden)
    optimal_threshold = thresholds[best_youden_idx] if best_youden_idx < len(thresholds) else 0.5

    # Binary predictions at optimal threshold
    preds_optimal = (all_probs >= optimal_threshold).astype(int)
    tp = np.sum((preds_optimal == 1) & (all_labels == 1))
    fn = np.sum((preds_optimal == 0) & (all_labels == 1))
    fp = np.sum((preds_optimal == 1) & (all_labels == 0))
    tn = np.sum((preds_optimal == 0) & (all_labels == 0))

    sensitivity = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    f2 = 5 * tp / max(5 * tp + 4 * fn + fp, 1)

    results = {
        PRIMARY_METRIC: float(mel_sens_at_spec80),
        "metrics": {
            "mel_auc": float(mel_auc),
            "sensitivity": float(sensitivity),
            "specificity": float(spec),
            "f1": float(f1),
            "f2": float(f2),
            "precision": float(precision),
            "threshold_at_spec80": float(threshold_at_spec80),
            "optimal_threshold": float(optimal_threshold),
        },
    }

    sample_preds = {
        "ids": all_ids,
        "labels": all_labels,
        "probs": all_probs,
    }

    return results, sample_preds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    run_dir = os.path.join(PROJECT_DIR, f"run{RUN_INDEX}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"=== Run {RUN_INDEX} | {BACKBONE} | BS={BATCH_SIZE} | LR={LR} ===")
    print(f"GPUs: {NUM_GPUS} | Device: {DEVICE}")
    print(f"Time budget: {TIME_BUDGET_MINUTES} min")

    # Data
    train_loader, val_loader, test_loader, info = build_dataloaders(batch_size=BATCH_SIZE)
    print(f"Classes: {CLASSES}")

    # Model
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {BACKBONE}, {total_params:.1f}M params")

    # Class weights for loss
    class_counts = info["class_counts_train"]
    total = sum(class_counts)
    class_weights = torch.tensor([total / (2 * c) for c in class_counts], dtype=torch.float32).to(DEVICE)

    # Loss
    if FOCAL_GAMMA > 0:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # EMA
    ema = EMAModel(model, decay=EMA_DECAY)

    # LR scheduler with warmup
    steps_per_epoch = len(train_loader)
    total_epochs = 200  # will be cut by time budget
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_epochs * steps_per_epoch - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = GradScaler('cuda')

    # Training loop
    best_metric = -1e9 if METRIC_DIRECTION == "higher" else 1e9
    best_epoch = -1
    global_step = 0
    start_time = time.time()

    for epoch in range(total_epochs):
        # Time check
        elapsed_min = (time.time() - start_time) / 60
        if elapsed_min >= TIME_BUDGET_MINUTES:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels, _ in train_loader:
            # Time check mid-epoch
            if (time.time() - start_time) / 60 >= TIME_BUDGET_MINUTES:
                break

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # CutMix or Mixup with 50% probability
            r = np.random.random()
            use_mix = False
            if r < 0.25 and CUTMIX_ALPHA > 0:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                use_mix = True
            elif r < 0.5 and MIXUP_ALPHA > 0:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                index = torch.randperm(images.size(0), device=images.device)
                images = lam * images + (1 - lam) * images[index]
                labels_a, labels_b = labels, labels[index]
                use_mix = True

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(images)
                if use_mix:
                    loss = cutmix_criterion(criterion, logits, labels_a, labels_b, lam)
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ema.update(model)
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        if n_batches == 0:
            break

        avg_loss = epoch_loss / n_batches

        # Save last checkpoint
        raw_model = model.module if hasattr(model, 'module') else model
        torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "last.pth"))

        # Evaluate with EMA weights
        ema.apply_shadow(model)
        val_results, _ = evaluate(model, val_loader)
        ema.restore(model)

        current_metric = val_results[PRIMARY_METRIC]
        is_better = (current_metric > best_metric) if METRIC_DIRECTION == "higher" else (current_metric < best_metric)

        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            # Save best with EMA weights
            ema.apply_shadow(model)
            raw_model = model.module if hasattr(model, 'module') else model
            torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
            ema.restore(model)

        elapsed_min = (time.time() - start_time) / 60
        lr_now = optimizer.param_groups[0]['lr']
        m = val_results["metrics"]
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
              f"{PRIMARY_METRIC}={current_metric:.4f} | auc={m['mel_auc']:.4f} | "
              f"sens={m['sensitivity']:.4f} | spec={m['specificity']:.4f} | "
              f"best={best_metric:.4f}@{best_epoch} | {elapsed_min:.1f}min")

    training_seconds = time.time() - start_time

    # ---------------------------------------------------------------------------
    # Final evaluation on best checkpoint with EMA
    # ---------------------------------------------------------------------------
    print("\n--- Loading best checkpoint for final evaluation ---")
    best_path = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(best_path):
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))

    model.eval()
    val_results, val_preds = evaluate_tta(model, val_loader)

    total_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # Print final results
    m = val_results["metrics"]
    print(f"\n---")
    print(f"{PRIMARY_METRIC}:      {val_results[PRIMARY_METRIC]:.4f}")
    print(f"\nmetrics_summary:")
    for k, v in m.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"\ntraining_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")

    return model, val_results, test_loader, run_dir


# ---------------------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_tta(model, loader, device=DEVICE):
    """Evaluate with 4x TTA (original, hflip, vflip, hflip+vflip)."""
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []

    for images, labels, img_ids in loader:
        images = images.to(device, non_blocking=True)
        augs = [images, torch.flip(images, [3]), torch.flip(images, [2]),
                torch.flip(images, [2, 3])]
        probs_sum = torch.zeros(images.size(0), 2, device=device)
        for aug in augs:
            with autocast('cuda'):
                logits = model(aug)
            probs_sum += F.softmax(logits, dim=1)
        avg_probs = (probs_sum / len(augs))[:, 1].cpu().numpy()
        all_probs.extend(avg_probs)
        all_labels.extend(labels.numpy())
        all_ids.extend(img_ids)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    mel_auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    specificity = 1 - fpr
    idx_spec80 = np.where(specificity >= 0.80)[0]
    if len(idx_spec80) > 0:
        best_idx = idx_spec80[np.argmin(specificity[idx_spec80])]
        mel_sens_at_spec80 = tpr[best_idx]
        threshold_at_spec80 = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        mel_sens_at_spec80 = 0.0
        threshold_at_spec80 = 0.5
    youden = tpr + specificity - 1
    best_youden_idx = np.argmax(youden)
    optimal_threshold = thresholds[best_youden_idx] if best_youden_idx < len(thresholds) else 0.5
    preds_optimal = (all_probs >= optimal_threshold).astype(int)
    tp = np.sum((preds_optimal == 1) & (all_labels == 1))
    fn = np.sum((preds_optimal == 0) & (all_labels == 1))
    fp = np.sum((preds_optimal == 1) & (all_labels == 0))
    tn = np.sum((preds_optimal == 0) & (all_labels == 0))
    results = {
        PRIMARY_METRIC: float(mel_sens_at_spec80),
        "metrics": {
            "mel_auc": float(mel_auc), "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "f2": float(5 * tp / max(5 * tp + 4 * fn + fp, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "threshold_at_spec80": float(threshold_at_spec80),
            "optimal_threshold": float(optimal_threshold),
        },
    }
    return results, {"ids": all_ids, "labels": all_labels, "probs": all_probs}


def run_test(model, test_loader, run_dir):
    """Run inference on test set with TTA, save predictions CSV."""
    test_results, test_preds = evaluate_tta(model, test_loader)

    # Save sample-wise predictions
    csv_path = os.path.join(run_dir, "test_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "ground_truth", "predicted_label", "mel_probability"])
        for img_id, label, prob in zip(test_preds["ids"], test_preds["labels"], test_preds["probs"]):
            pred_label = 1 if prob >= test_results["metrics"]["threshold_at_spec80"] else 0
            writer.writerow([img_id, int(label), pred_label, f"{prob:.6f}"])

    m = test_results["metrics"]
    print(f"\n--- Test Results ---")
    print(f"test_{PRIMARY_METRIC}: {test_results[PRIMARY_METRIC]:.4f}")
    for k, v in m.items():
        print(f"  test_{k}: {v:.4f}")

    return test_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, val_results, test_loader, run_dir = train()
    # Baseline run: always test
    test_results = run_test(model, test_loader, run_dir)
