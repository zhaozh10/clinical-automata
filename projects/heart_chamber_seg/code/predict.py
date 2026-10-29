"""Run the cardiac segmentation model on one or more chest X-ray images.

Usage:
    uv run python projects/20260717_07_heart_chamber_seg_xraydino/predict.py IMAGE.png [IMAGE2.png ...]
    uv run python projects/20260717_07_heart_chamber_seg_xraydino/predict.py --ckpt <path> --outdir <dir> IMAGE.png

For each input it writes:
    <name>_overlay.png  -- cardiac silhouette (red) + chamber contours over the film
    <name>_heart.png    -- the binary cardiac-silhouette mask (the trustworthy output)
and prints the predicted heart area and the cardiothoracic ratio (CTR) estimate.

Notes:
  * The cardiac SILHOUETTE (heart) is the reliable output. The chamber contours are
    an atlas prior, NOT a measurement -- see USAGE_MANUAL.md section 4.
  * Input is read as grayscale and resized to 512x512. Frontal (PA/AP) films are
    what the real-film validation covers; lateral is unverified on real film.
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data import HEART_NAMES, PRIMARY_CH, CHAMBER_CH, IMG_SIZE   # noqa: E402
from train import HeartSeg                                        # noqa: E402

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "best_run8.pth")
CHAMBER_COLORS = [(255, 80, 80), (80, 160, 255), (80, 255, 255), (80, 255, 120), (200, 120, 255)]


def load_model(ckpt, device):
    model = HeartSeg().to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model


@torch.no_grad()
def predict(model, img_gray, device):
    """img_gray: HxW uint8. Returns (7, 512, 512) float probs and the resized 512 image."""
    if img_gray.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    else:
        img = img_gray
    x = torch.from_numpy(img).to(device).float()[None, None] / 255.0
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        prob = torch.sigmoid(model(x).float())[0]
    return prob.cpu().numpy(), img


def ctr(heart_mask):
    """Cardiothoracic-ratio proxy: heart max width / image width (no thorax mask here)."""
    xs = np.where(heart_mask.any(0))[0]
    return (xs.max() - xs.min() + 1) / heart_mask.shape[1] if xs.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--outdir", default="projects/20260717_07_heart_chamber_seg_xraydino/predictions")
    ap.add_argument("--thr", type=float, default=0.5)   # 0.5 is calibrated; see manual
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    for path in args.images:
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"[skip] cannot read {path}")
            continue
        prob, img512 = predict(model, img_gray, device)
        pred = prob > args.thr
        heart = pred[PRIMARY_CH]

        base = cv2.cvtColor(img512, cv2.COLOR_GRAY2BGR)
        base[heart] = (0.6 * base[heart] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
        for k, c in enumerate(CHAMBER_CH):
            m = pred[c].astype(np.uint8)
            if m.sum():
                cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(base, cont, -1, CHAMBER_COLORS[k % len(CHAMBER_COLORS)], 1)

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.outdir, f"{stem}_overlay.png"), base)
        cv2.imwrite(os.path.join(args.outdir, f"{stem}_heart.png"), heart.astype(np.uint8) * 255)
        conf = float(prob[PRIMARY_CH][heart].mean()) if heart.any() else 0.0
        print(f"{os.path.basename(path)}: heart area {int(heart.sum())} px "
              f"({heart.mean()*100:.1f}% of frame), CTR~{ctr(heart):.2f}, "
              f"mean conf {conf:.3f} -> {args.outdir}/{stem}_overlay.png")


if __name__ == "__main__":
    main()
