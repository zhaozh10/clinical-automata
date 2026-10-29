"""
Paediatric wrist-fracture evidence-map detector -- inference / finalisation.

Usage
  uv run projects/20260910_09_wrist_fracture_det/predict.py --test          # sealed test set + metrics
  uv run projects/20260910_09_wrist_fracture_det/predict.py --images DIR    # new radiographs (PNG)

Outputs per radiograph: a fracture probability, the evidence map overlaid on
the film, and a box around the marked region -- all in ORIGINAL pixel
coordinates of the input image.
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import train as T
import data as D
from prepare import EXT_DATA_DIR

PROJECT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(PROJECT, "run12", "checkpoints", "best.pth")
VAL_THRESHOLD = 0.0614          # operating point chosen on VALIDATION only
OVERLAY_DIR = os.path.join(EXT_DATA_DIR, "gpwd_sep10_test_overlays")
dev = torch.device("cuda:0")


def load_model(ckpt=CKPT):
    m = T.EvidenceNet().to(dev)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model"])
    m.eval()
    return m


def to_original(box, rec):
    """letterbox pixel coords -> original radiograph pixel coords"""
    sc = rec["scale"]; px, py = rec["pad"]; w, h = rec["orig"]
    x1 = float(np.clip((box[0] - px) / sc, 0, w - 1))
    y1 = float(np.clip((box[1] - py) / sc, 0, h - 1))
    x2 = float(np.clip((box[2] - px) / sc, 0, w - 1))
    y2 = float(np.clip((box[3] - py) / sc, 0, h - 1))
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def draw_overlay(stem, rec, prob, boxes, scores, heat, out_path, gt=None):
    """Radiograph + evidence map + marked fracture, at original resolution."""
    img = cv2.imread(os.path.join(D.IMG_DIR, stem + ".png"), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 3:
        img = img[..., 0]
    img8 = (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    h, w = img8.shape
    rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    # evidence map: stride-8 grid -> letterbox -> strip padding -> original size
    hm = cv2.resize(heat, (D.IMG_W, D.IMG_H), interpolation=cv2.INTER_CUBIC)
    sc = rec["scale"]; px, py = rec["pad"]
    nw, nh = int(round(w * sc)), int(round(h * sc))
    hm = hm[py:py + nh, px:px + nw]
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)
    colour = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (np.clip(hm / max(hm.max(), 1e-6), 0, 1) ** 1.5)[..., None] * 0.45
    rgb = (rgb * (1 - alpha) + colour * alpha).astype(np.uint8)

    if gt:
        for g in gt:
            b = [int(round(v)) for v in to_original(g, rec)]
            cv2.rectangle(rgb, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    for b, s in zip(boxes, scores):
        if s < VAL_THRESHOLD:
            continue
        bb = [int(round(v)) for v in b]
        cv2.rectangle(rgb, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        cv2.putText(rgb, f"{s:.2f}", (bb[0], max(bb[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    verdict = "FRACTURE" if prob >= VAL_THRESHOLD else "no fracture"
    cv2.putText(rgb, f"{verdict}  p={prob:.3f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if gt is not None:
        cv2.putText(rgb, "green = radiologist   red = model", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, rgb)
    return True


@torch.no_grad()
def collect_maps(model, recs, batch=24):
    """second pass: evidence maps for the overlays"""
    dl = DataLoader(D.WristDataset(recs, False), batch_size=batch, shuffle=False,
                    num_workers=10, collate_fn=D.collate, pin_memory=True)
    out = {}
    for b in dl:
        x = b["img"].to(dev).float()
        hl, lt, rb = model(x)
        p = torch.sigmoid(hl)[:, 0].float().cpu().numpy()
        for k, gi in enumerate(b["idx"].tolist()):
            out[recs[gi]["stem"]] = p[k]
    return out


def run_test():
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    gallery = os.path.join(PROJECT, "example_overlays")
    os.makedirs(gallery, exist_ok=True)
    model = load_model()
    recs = D.load_test_index()
    print(f"test films: {len(recs)}  fracture-positive: {sum(r['label'] for r in recs)}", flush=True)

    dl = DataLoader(D.WristDataset(recs, False), batch_size=32, shuffle=False,
                    num_workers=10, collate_fn=D.collate, pin_memory=True)
    res = T.evaluate(model, dl, recs, collect=True)          # LOCKED metric function
    model.eval()
    per = {p["stem"]: p for p in res["per_sample"]}

    # patient ids for the CSV
    meta = {}
    with open(os.path.join(D.ROOT, "dataset.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            meta[r["filestem"]] = r
    maps = collect_maps(model, recs)

    rows, doctor = [], []
    n_over = 0
    for i, rec in enumerate(recs):
        stem = rec["stem"]
        p = per[stem]
        prob = p["score"]
        order = np.argsort(-np.array(p["scores"]))
        boxes = [to_original(p["boxes"][j], rec) for j in order]
        scores = [round(float(p["scores"][j]), 4) for j in order]
        kept = [(b, s) for b, s in zip(boxes, scores) if s >= VAL_THRESHOLD][:5]
        gt_o = [to_original(g, rec) for g in rec["boxes"]]

        # per-film localisation score: best overlap between a kept mark and a
        # true fracture, measured in the same space the metric uses
        best_iou = 0.0
        if rec["boxes"] and kept:
            kept_lb = np.array([p["boxes"][j] for j in order
                                if p["scores"][j] >= VAL_THRESHOLD][:5], np.float32)
            m = T.iou_mat(kept_lb, np.array(rec["boxes"], np.float32).reshape(-1, 4))
            best_iou = float(m.max()) if m.size else 0.0

        ov = os.path.join(OVERLAY_DIR, stem + ".png")
        if draw_overlay(stem, rec, prob, [b for b, _ in kept], [s for _, s in kept],
                        maps[stem], ov, gt=rec["boxes"]):
            n_over += 1
        if i < 24:
            draw_overlay(stem, rec, prob, [b for b, _ in kept], [s for _, s in kept],
                         maps[stem], os.path.join(gallery, stem + ".png"), gt=rec["boxes"])

        md = meta.get(stem, {})
        rows.append({
            "sample_id": stem,
            "patient_id": md.get("patient_id", stem.split("_")[0]),
            "age_years": md.get("age", ""),
            "sex": md.get("gender", ""),
            "projection": md.get("projection", ""),
            "label": rec["label"],
            "prediction": int(prob >= VAL_THRESHOLD),
            "confidence": round(float(prob), 4),
            "n_true_fractures": len(rec["boxes"]),
            "n_predicted_marks": len(kept),
            "top_box_xyxy": ";".join(str(v) for v in kept[0][0]) if kept else "",
            "top_box_confidence": kept[0][1] if kept else "",
            "best_overlap_with_true_fracture": round(best_iou, 4),
            "overlay_png": ov,
        })
        doctor.append({
            "film": stem, "patient_id": md.get("patient_id", ""),
            "fracture_probability": round(float(prob), 4),
            "call": "fracture" if prob >= VAL_THRESHOLD else "no fracture",
            "marked_regions": [{"box_xyxy_original_px": b, "confidence": s} for b, s in kept],
            "radiologist_fracture_boxes_xyxy_original_px": gt_o,
            "overlay": ov,
        })

    with open(os.path.join(PROJECT, "test_predictions.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with open(os.path.join(PROJECT, "test_report.json"), "w") as f:
        json.dump({"films": doctor}, f, indent=1)

    # operating point fixed on VALIDATION (the locked metric retunes on whatever
    # set it is given, which would be test-set tuning)
    y = np.array([r["label"] for r in recs])
    s = np.array([per[r["stem"]]["score"] for r in recs])
    pr = s >= VAL_THRESHOLD
    tp = int((pr & (y == 1)).sum()); fn = int((~pr & (y == 1)).sum())
    tn = int((~pr & (y == 0)).sum()); fp = int((pr & (y == 0)).sum())
    fixed = {"threshold": VAL_THRESHOLD,
             "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
             "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
             "f1": 2 * tp / max(2 * tp + fp + fn, 1),
             "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    out = {k: v for k, v in res.items() if k != "per_sample"}
    out["at_validation_threshold"] = fixed
    out["n_overlays"] = n_over
    with open(os.path.join(PROJECT, "test_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


def run_images(folder):
    model = load_model()
    stems = sorted(x[:-4] for x in os.listdir(folder) if x.lower().endswith(".png"))
    os.makedirs(os.path.join(folder, "predictions"), exist_ok=True)
    res = []
    for stem in stems:
        im = cv2.imread(os.path.join(folder, stem + ".png"), cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = im[..., 0]
        h, w = im.shape
        im8 = (im >> 8).astype(np.uint8) if im.dtype == np.uint16 else im.astype(np.uint8)
        sc, nw, nh, px, py = D._letterbox_params(w, h)
        canvas = np.zeros((D.IMG_H, D.IMG_W), np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(im8, (nw, nh))
        x = ((canvas.astype(np.float32) / 255.0 - D.MEAN) / D.STD)[None, None]
        with torch.no_grad():
            hl, lt, rb = model(torch.from_numpy(x).to(dev))
            prob = float(torch.sigmoid(T.pool_logit(hl[:, 0])).item())
            boxes, scores, _, _ = T.decode(hl, lt, rb, K=5)
        rec = {"scale": sc, "pad": (px, py), "orig": (w, h)}
        bs = boxes[0].cpu().numpy(); ss = scores[0].cpu().numpy()
        kept = [(to_original(bs[j], rec), float(ss[j])) for j in range(len(ss)) if ss[j] >= VAL_THRESHOLD]
        print(f"{stem}: p(fracture)={prob:.3f} -> {'FRACTURE' if prob>=VAL_THRESHOLD else 'no fracture'}, "
              f"{len(kept)} marked region(s)")
        res.append({"film": stem, "fracture_probability": round(prob, 4),
                    "marked_regions": [{"box_xyxy_original_px": b, "confidence": round(s, 4)}
                                       for b, s in kept]})
    with open(os.path.join(folder, "predictions", "report.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--images")
    a = ap.parse_args()
    if a.test:
        run_test()
    elif a.images:
        run_images(a.images)
    else:
        ap.error("give --test or --images DIR")
