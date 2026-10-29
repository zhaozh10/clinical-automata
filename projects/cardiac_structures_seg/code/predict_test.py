"""Test-set inference for the cardiac teaching model.

Writes, for every test radiograph:
  * the segmentation masks (packed .npz, one bit-plane per structure)  -> EXT_DATA_DIR
  * a per-sample CSV row with Dice per structure, valve localisation error,
    border composition and CTR (for reporting and post-hoc error analysis)
  * optional teaching figures for a subset

Usage:
    uv run predict_test.py --checkpoint projects/.../best.pth [--figures 12]
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

import data as D
import train as T
import visualize as V

OUT_DIR = os.path.join(D.EXT_DATA_DIR, "cardiac_teaching_predictions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--project", default="projects/20260723_07")
    ap.add_argument("--figures", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    device = "cuda"
    model = V.load_model(args.checkpoint, device)

    recs = D.load_records("test")
    if args.limit:
        recs = recs[:args.limit]
    ds = D.PaxrayCardiacDataset(recs, train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                         num_workers=6, pin_memory=True)

    # aggregate metrics with the same (fixed) evaluation function used for model selection
    t0 = time.time()
    res = T.evaluate(model, loader, device, recs)
    print("=== test set (n=%d) ===" % len(recs))
    for k, v in res["metrics"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"eval seconds: {time.time() - t0:.1f}")

    # per-sample predictions + masks
    rows = []
    K = len(T.BORDER_IDX)
    with torch.no_grad():
        for x, y, valid, idxs in loader:
            x = x.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prob = torch.sigmoid(model(x).float())
            pred = (prob > 0.5).to(torch.uint8).cpu().numpy()
            gt = y.numpy().astype(np.uint8)
            conf = prob.cpu().numpy()
            for b in range(pred.shape[0]):
                r = recs[int(idxs[b])]
                p, g = pred[b], gt[b]
                np.savez_compressed(
                    os.path.join(args.out, "masks", r["file_name"].replace(".png", ".npz")),
                    masks=np.packbits(p, axis=-1), channels=np.array(D.CHANNEL_NAMES),
                    shape=np.array(p.shape))
                row = {"file_name": r["file_name"], "patient": r["patient"], "view": r["view"]}
                for i, n in enumerate(D.CHANNEL_NAMES):
                    if r["view"] == "lateral" and i in D.LUNG_IDX:
                        row[f"dice_{n}"] = ""
                        row[f"conf_{n}"] = ""
                        continue
                    inter = float((p[i] & g[i]).sum())
                    row[f"dice_{n}"] = round(2 * inter / (p[i].sum() + g[i].sum() + 1e-6), 4) \
                        if g[i].sum() > 0 else ""
                    row[f"conf_{n}"] = round(float(conf[b, i][p[i] > 0].mean()), 4) if p[i].any() else 0.0
                    row[f"area_px_{n}"] = int(p[i].sum())
                for name, _, _ in D.VALVE_CHANNELS:
                    c = D.IDX[name]
                    cg, cp = _cent(g[c]), _cent(p[c])
                    row[f"err_{name}_px"] = round(float(np.linalg.norm(cg - cp)), 2) \
                        if cg is not None and cp is not None else ""
                    row[f"{name}_x"] = round(float(cp[1]), 1) if cp is not None else ""
                    row[f"{name}_y"] = round(float(cp[0]), 1) if cp is not None else ""
                if r["view"] == "frontal":
                    mm_p, mm_g = V.measurements(p), V.measurements(g)
                    row["ctr_pred"] = round(mm_p.get("ctr", float("nan")), 4)
                    row["ctr_gt"] = round(mm_g.get("ctr", float("nan")), 4)
                    row["lv_area_cm2_pred"] = round(mm_p.get("area_ventricle_left_cm2", 0), 1)
                    row["lv_frac_pred"] = round(mm_p.get("frac_ventricle_left", 0), 4)
                    row["enlarged_pred"] = int(mm_p.get("ctr", 0) >= T.ENLARGED_CTR)
                    row["enlarged_gt"] = int(mm_g.get("ctr", 0) >= T.ENLARGED_CTR)
                    lab_p, _ = V.border_segments(p)
                    lab_g, _ = V.border_segments(g)
                    m = lab_g >= 0
                    row["border_acc"] = round(float(((lab_p == lab_g) & m).sum() / max(1, m.sum())), 4)
                rows.append(row)

    cols = sorted({k for r in rows for k in r})
    cols = ["file_name", "patient", "view"] + [c for c in cols if c not in ("file_name", "patient", "view")]
    csv_path = os.path.join(args.project, "test_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("per-sample predictions:", csv_path)
    print("masks:", os.path.join(args.out, "masks"))

    if args.figures:
        fig_dir = os.path.join(args.project, "teaching_examples")
        os.makedirs(fig_dir, exist_ok=True)
        frontal = [r for r in rows if r["view"] == "frontal" and r.get("ctr_gt") == r.get("ctr_gt")]
        frontal.sort(key=lambda r: r["ctr_gt"])
        n = args.figures
        picks = [frontal[int(i * (len(frontal) - 1) / max(1, n - 1))] for i in range(n)]
        for r in picks:
            for fn in (r["file_name"], r["file_name"].replace("_frontal", "_lateral")):
                rec = next((x for x in recs if x["file_name"] == fn), None)
                if rec is None:
                    continue
                img = D.read_image(fn)
                _, masks = V.predict(model, img, rec["view"], device)
                out = os.path.join(fig_dir, fn.replace(".png", "_Modell.png"))
                V.figure_for_image(img, masks, rec["view"], f"{fn} ({rec['view']}, Modell)", out)
                print(out)


def _cent(m):
    idx = np.nonzero(m)
    if idx[0].size == 0:
        return None
    return np.array([idx[0].mean(), idx[1].mean()])


if __name__ == "__main__":
    main()
