"""
train.py — YOLOv8 fracture detection on GRAZPEDWRI-DX.

YOLOv8m pretrained on COCO, fine-tuned for single-class fracture detection.
4x GPU DDP training, time-budgeted, mAP50 as primary metric.
"""

import os
import sys
import time
import shutil
import glob as globmod
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import TIME_BUDGET_MINUTES, EXT_DATA_DIR
from data import prepare_yolo_dataset, generate_pseudo_labels, YAML_PATH, YOLO_DIR

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(REPO_ROOT, "projects", "20260406_18")
PSEUDO_MODEL = os.path.join(PROJECT_DIR, "run3", "checkpoints", "best.pt")


def find_next_run_idx():
    # Allow override via env var to avoid mkdir race condition
    env_idx = os.environ.get("RUN_IDX")
    if env_idx:
        return int(env_idx)
    p = Path(PROJECT_DIR)
    if not p.exists():
        return 1
    indices = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("run"):
            try:
                indices.append(int(d.name[3:]))
            except ValueError:
                pass
    return max(indices) + 1 if indices else 1


# ---------------------------------------------------------------------------
# Evaluation (FIXED — do not modify across iterations)
# ---------------------------------------------------------------------------
def evaluate(model, data_yaml, split="val"):
    """Compute detection metrics. Returns dict with mAP50 and sub-metrics."""
    results = model.val(data=data_yaml, split=split, verbose=False)

    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mAP50": map50,
        "metrics": {
            "mAP50-95": map50_95,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        },
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    from ultralytics import YOLO

    # ---- Pseudo-label generation + data preparation ----
    pseudo = generate_pseudo_labels(PSEUDO_MODEL)
    yaml_path, n_train, n_val, n_test = prepare_yolo_dataset(
        force=True, pseudo_labels=pseudo)

    # ---- Run directory (absolute paths for DDP compatibility) ----
    run_idx = find_next_run_idx()
    run_dir = os.path.join(PROJECT_DIR, f"run{run_idx}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Project:      {PROJECT_DIR}")
    print(f"Run:          run{run_idx}")
    print(f"Data:         train={n_train}  val={n_val}  test={n_test}")
    print(f"Time budget:  {TIME_BUDGET_MINUTES} min")
    print(f"Device:       4x GPU (DDP)")
    print()

    # ---- Model ----
    model = YOLO("yolov8m.pt")

    # ---- Train ----
    # DDP note: callbacks don't run in worker processes, so we control
    # training duration via epoch count. Measured: ~26s/epoch with 4xGPU batch64.
    # With pseudo-labels: ~7500 train images. Measured: 42s/epoch (run13, 90ep in 63min).
    # Extended budget: 70 min = 4200s. Minus ~5min pseudo → ~3900s. 3900/42 ≈ 93ep. Use 110
    # and accept slight overshoot (run13 measured 3800s for 90ep, so 110ep ≈ 4640s ≈ 77min).
    max_epochs = 110

    print(f"Starting training ({max_epochs} epochs)...")
    t0 = time.time()

    model.train(
        data=yaml_path,
        epochs=max_epochs,
        imgsz=640,
        batch=64,
        device=[0, 1, 2, 3],
        project=run_dir,
        name="yolo",
        patience=0,                # no early stopping; epoch count controls
        save=True,
        workers=8,
        exist_ok=True,
        verbose=True,
        # --- training schedule ---
        label_smoothing=0.05,      # regularize against noisy pseudo-labels
        # --- medical-appropriate augmentation ---
        hsv_h=0.0,                 # no hue shift (grayscale)
        hsv_s=0.1,
        hsv_v=0.3,                 # stronger brightness jitter
        degrees=15.0,              # slightly more rotation
        translate=0.15,
        scale=0.4,                 # wider scale range
        fliplr=0.5,
        flipud=0.0,               # no vertical flip for wrist X-rays
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    training_seconds = time.time() - t0

    # ---- Locate best weights (DDP may nest under runs/detect/) ----
    best_candidates = [
        os.path.join(run_dir, "yolo", "weights", "best.pt"),
        *globmod.glob(os.path.join(REPO_ROOT, "runs", "**", "best.pt"), recursive=True),
    ]
    best_path = None
    for p in best_candidates:
        if os.path.exists(p):
            best_path = p
            break

    if best_path is None:
        print("ERROR: Could not find best.pt after training")
        return None, 0.0

    # Copy to our checkpoint dir
    shutil.copy2(best_path, os.path.join(ckpt_dir, "best.pt"))
    # Also copy last if available
    last_path = best_path.replace("best.pt", "last.pt")
    if os.path.exists(last_path):
        shutil.copy2(last_path, os.path.join(ckpt_dir, "last.pt"))

    # ---- Final validation ----
    best_model = YOLO(os.path.join(ckpt_dir, "best.pt"), task="detect")
    val_metrics = evaluate(best_model, yaml_path, "val")

    total_seconds = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    # ---- Print results ----
    print("\n---")
    print(f"mAP50:      {val_metrics['mAP50']:.4f}")
    print()
    print("metrics_summary:")
    for k, v in val_metrics["metrics"].items():
        print(f"{k}:    {v:.4f}")
    print()
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")

    # ---- Cleanup stray runs/ directory created by DDP ----
    stray = os.path.join(REPO_ROOT, "runs")
    if os.path.exists(stray):
        shutil.rmtree(stray, ignore_errors=True)

    return val_metrics, peak_vram_mb


if __name__ == "__main__":
    train()
