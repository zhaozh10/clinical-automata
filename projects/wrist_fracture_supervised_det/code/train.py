"""Wrist fracture detection — Faster R-CNN baseline."""

import os
import sys
import time
import json
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision

from prepare import TIME_BUDGET_MINUTES, DATA_DIR
from data import get_dataloaders

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = "projects/20260405_15"
ITERATION = 29
RUN_DIR = os.path.join(PROJECT_DIR, f"run{ITERATION}")
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")

NUM_CLASSES = 2          # background + fracture
BATCH_SIZE = 8           # per GPU
LR = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4

PRIMARY_METRIC = "mAP@50"
METRIC_DIRECTION = "higher"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_model():
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=640,
        max_size=1024,
        trainable_backbone_layers=3,  # freeze layers 0-1, train layers 2-4+FPN+head
    )
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    return model


# ---------------------------------------------------------------------------
# Evaluation  (FIXED — do not change during experiment loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Compute mAP@50 (primary), mAP@50:95, recall, precision, F1."""
    model.eval()
    preds, gts = [], []

    for imgs, tgts in loader:
        imgs = [x.to(device) for x in imgs]
        with torch.amp.autocast("cuda"):
            outs = model(imgs)
        for i, o in enumerate(outs):
            preds.append({k: v.cpu() for k, v in o.items()})
            gts.append({"boxes": tgts[i]["boxes"], "labels": tgts[i]["labels"]})

    # mAP @ IoU=0.5  (sync_on_compute=False — we evaluate full val on rank 0)
    m50 = MeanAveragePrecision(iou_thresholds=[0.5], sync_on_compute=False)
    m50.update(preds, gts)
    r50 = m50.compute()

    # mAP @ IoU=0.5:0.95
    m5095 = MeanAveragePrecision(sync_on_compute=False)
    m5095.update(preds, gts)
    r5095 = m5095.compute()

    # Precision / recall / F1 at score≥0.5, IoU≥0.5
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        keep = p["scores"] >= 0.5
        pb, gb = p["boxes"][keep], g["boxes"]
        if len(gb) == 0:
            fp += len(pb)
            continue
        if len(pb) == 0:
            fn += len(gb)
            continue
        iou = torchvision.ops.box_iou(pb, gb)
        matched = set()
        for j in range(len(pb)):
            val, idx = iou[j].max(0)
            if val >= 0.5 and idx.item() not in matched:
                tp += 1
                matched.add(idx.item())
            else:
                fp += 1
        fn += len(gb) - len(matched)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    return {
        PRIMARY_METRIC: r50["map"].item(),
        "metrics": {
            "mAP@50:95": r5095["map"].item(),
            "fracture_recall@50": rec,
            "fracture_precision@50": prec,
            "f1@50": f1,
        },
    }


# ---------------------------------------------------------------------------
# Test inference — annotated images for clinical review
# ---------------------------------------------------------------------------

def save_test_predictions(model, loader, device, out_dir):
    """Draw fracture boxes on test images and save them + a JSON summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    all_preds = {}

    for imgs, tgts in loader:
        imgs_dev = [x.to(device) for x in imgs]
        with torch.amp.autocast("cuda"):
            outs = model(imgs_dev)
        for i, o in enumerate(outs):
            fs = tgts[i]["filestem"]
            boxes = o["boxes"].cpu()
            scores = o["scores"].cpu()
            keep = scores >= 0.3          # low threshold — assistive, flag more
            boxes, scores = boxes[keep], scores[keep]

            img_np = imgs[i][0].numpy()   # single-channel view
            fig, ax = plt.subplots(1, figsize=(6, 10))
            ax.imshow(img_np, cmap="gray")
            for b, s in zip(boxes, scores):
                x1, y1, x2, y2 = b.tolist()
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="red", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"{s:.2f}", color="red",
                        fontsize=9, fontweight="bold")
            ax.set_axis_off()
            fig.savefig(os.path.join(out_dir, f"{fs}.png"),
                        bbox_inches="tight", dpi=120)
            plt.close(fig)

            all_preds[fs] = [
                {"box": b.tolist(), "score": round(s.item(), 4)}
                for b, s in zip(boxes, scores)
            ]

    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump(all_preds, f, indent=2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_worker(rank, world_size, port):
    distributed = world_size > 1
    if distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)

    # Data — val/test loaders are non-distributed (rank 0 only)
    train_ld, val_ld, test_ld, sampler = get_dataloaders(
        BATCH_SIZE, NUM_WORKERS, distributed, world_size, rank,
    )

    model = build_model().to(device)
    if distributed:
        model = DDP(model, device_ids=[rank])
    raw = model.module if distributed else model

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=2,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=LR * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[2],
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val = -1.0
    best_results = None
    budget = TIME_BUDGET_MINUTES * 60
    t0 = time.time()
    epoch = 0

    # ---- training loop ----
    while True:
        if time.time() - t0 >= budget:
            break
        epoch += 1
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        loss_sum, n = 0.0, 0
        for imgs, tgts in train_ld:
            if time.time() - t0 >= budget:
                break
            imgs = [x.to(device) for x in imgs]
            tgts_dev = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in tgts
            ]

            with torch.amp.autocast("cuda"):
                loss_dict = model(imgs, tgts_dev)
                loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            n += 1

        scheduler.step()

        # ---- evaluate (rank 0 only) ----
        if rank == 0:
            res = evaluate(raw, val_ld, device)
            v = res[PRIMARY_METRIC]
            elapsed_min = (time.time() - t0) / 60
            print(
                f"Epoch {epoch:3d} | loss {loss_sum / max(n, 1):.4f} | "
                f"{PRIMARY_METRIC} {v:.4f} | "
                f"recall {res['metrics']['fracture_recall@50']:.4f} | "
                f"prec {res['metrics']['fracture_precision@50']:.4f} | "
                f"f1 {res['metrics']['f1@50']:.4f} | "
                f"{elapsed_min:.1f}min",
                flush=True,
            )

            # checkpoints
            torch.save(
                {"model": raw.state_dict(), "epoch": epoch, "metric": v},
                os.path.join(CKPT_DIR, "last.pth"),
            )
            if v > best_val:
                best_val = v
                best_results = res
                torch.save(
                    {"model": raw.state_dict(), "epoch": epoch, "metric": v},
                    os.path.join(CKPT_DIR, "best.pth"),
                )

        if distributed:
            dist.barrier()

    # ---- final report (rank 0) ----
    if rank == 0:
        total = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

        # reload best checkpoint
        ckpt = torch.load(os.path.join(CKPT_DIR, "best.pth"),
                          map_location=device, weights_only=False)
        raw.load_state_dict(ckpt["model"])

        print("\n---")
        print(f"{PRIMARY_METRIC}:      {best_val:.4f}")
        print("\nmetrics_summary:")
        if best_results:
            for k, v in best_results["metrics"].items():
                print(f"{k}:  {v:.4f}")
        print(f"\ntraining_seconds: {budget:.1f}")
        print(f"total_seconds:    {total:.1f}")
        print(f"peak_vram_mb:     {peak_mb:.1f}")

        # test evaluation
        test_res = evaluate(raw, test_ld, device)
        print(f"\n--- TEST ---")
        print(f"test_{PRIMARY_METRIC}:  {test_res[PRIMARY_METRIC]:.4f}")
        for k, v in test_res["metrics"].items():
            print(f"test_{k}:  {v:.4f}")

    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    world_size = torch.cuda.device_count()
    port = find_free_port()
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, port),
                 nprocs=world_size, join=True)
    else:
        train_worker(0, 1, None)


if __name__ == "__main__":
    main()
