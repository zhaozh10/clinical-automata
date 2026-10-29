"""Teaching visualisations for cardiac chamber segmentation on chest radiographs.

Produces, for one radiograph (frontal and/or lateral):

  1. Superimposition panel  - how strongly the cardiac structures project on top of
     each other; a projection radiograph is a summation image, not a slice.
  2. Chamber panel          - each chamber separately, ordered anterior -> posterior.
  3. Valve-plane panel      - the four valve planes as derived contact bands.
  4. Border panel           - which structure forms which part of the cardiac silhouette.
  5. Measurement panel      - CTR and chamber morphometry, with LV-enlargement read-out.

Usage:
    uv run visualize.py --checkpoint <best.pth> --image <name>_frontal.png --out <dir>
    uv run visualize.py --checkpoint <best.pth> --demo 6 --out <dir>       # curated examples
    uv run visualize.py --checkpoint <best.pth> --image <...> --gt        # use ground truth
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

import data as D

# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------
COLORS = {
    "ventricle_right": (0.30, 0.60, 1.00),
    "ventricle_left": (0.89, 0.34, 0.18),
    "atrium_right": (0.09, 0.75, 0.73),
    "atrium_left": (0.71, 0.48, 0.92),
    "myocardium": (0.75, 0.75, 0.78),
    "aorta_ascending": (0.95, 0.76, 0.31),
    "aorta_arch": (0.95, 0.63, 0.24),
    "aorta_descending": (0.79, 0.47, 0.18),
    "pulmonary_artery": (0.17, 0.66, 0.29),
    "pulmonary_trunk": (0.10, 0.80, 0.45),
    "vena_cava_inferior": (0.48, 0.58, 0.68),
    "trachea": (0.95, 0.95, 0.95),
    "heart": (1.00, 1.00, 1.00),
    "valve_mitral": (0.89, 0.34, 0.18),
    "valve_tricuspid": (0.09, 0.75, 0.73),
    "valve_aortic": (0.95, 0.76, 0.31),
    "valve_pulmonary": (0.17, 0.66, 0.29),
}

DE = {
    "ventricle_right": "rechter Ventrikel",
    "ventricle_left": "linker Ventrikel",
    "atrium_right": "rechter Vorhof",
    "atrium_left": "linker Vorhof",
    "myocardium": "Myokard",
    "aorta_ascending": "Aorta ascendens",
    "aorta_arch": "Aortenbogen",
    "aorta_descending": "Aorta descendens",
    "pulmonary_artery": "Pulmonalgefäße (gesamt)",
    "pulmonary_trunk": "Truncus pulmonalis / zentrale PA",
    "vena_cava_inferior": "V. cava inferior",
    "trachea": "Trachea",
    "heart": "Herz gesamt",
    "valve_mitral": "Mitralklappe",
    "valve_tricuspid": "Trikuspidalklappe",
    "valve_aortic": "Aortenklappe",
    "valve_pulmonary": "Pulmonalklappe",
}

# anterior -> posterior on the frontal projection (what lies on top of what)
DEPTH_ORDER = ["ventricle_right", "ventricle_left", "atrium_right", "atrium_left"]

BORDER_CANDIDATES = ["aorta_ascending", "aorta_arch", "aorta_descending", "pulmonary_trunk",
                     "atrium_right", "atrium_left", "ventricle_left", "ventricle_right",
                     "vena_cava_inferior"]

PX_PER_CM = D.IMG_SIZE / 40.0  # PAXRay renders a ~40 cm field of view at 512 px
REF_STATS_PATH = "projects/20260723_07/reference_stats.json"
_REF = None


def reference_stats():
    """Cohort reference percentiles, computed once on the TRAINING ground truth."""
    global _REF
    if _REF is None:
        import json
        with open(REF_STATS_PATH) as f:
            _REF = json.load(f)
    return _REF


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_model(checkpoint, device="cuda"):
    """checkpoint may be a single .pth or a comma-separated list -> ensemble."""
    import train as T
    paths = [p for p in str(checkpoint).split(",") if p]
    if len(paths) > 1:
        return T.Ensemble(paths).to(device).eval()
    ck = torch.load(paths[0], map_location="cpu")
    cfg = ck.get("cfg", T.CFG)
    model = T.UNet(cfg.get("encoder", "convnext_small"), work_size=cfg.get("work_size", 0))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, img, view, device="cuda", threshold=0.5, tta=False):
    """tta defaults to off so the rendered masks match the reported validation metrics."""
    x = np.stack([img.astype(np.float32) / 255.0,
                  np.full(img.shape, 1.0 if view == "lateral" else 0.0, np.float32)])
    xt = torch.from_numpy(x)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prob = torch.sigmoid(model(xt).float())
        if tta:  # small-scale TTA: mild zoom, averaged back
            xs = torch.nn.functional.interpolate(xt, scale_factor=1.15, mode="bilinear")
            p2 = torch.sigmoid(model(xs).float())
            p2 = torch.nn.functional.interpolate(p2, size=prob.shape[-2:], mode="bilinear")
            prob = 0.5 * (prob + p2)
    prob = prob[0].float().cpu().numpy()
    return prob, (prob > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Derived teaching quantities
# ---------------------------------------------------------------------------
def overlap_table(masks):
    """Pairwise projection overlap of the chambers, in % of the smaller structure."""
    rows = []
    names = DEPTH_ORDER
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = masks[D.IDX[a]], masks[D.IDX[b]]
            inter = int((ma & mb).sum())
            small = max(1, min(int(ma.sum()), int(mb.sum())))
            rows.append((a, b, inter, 100.0 * inter / small))
    return rows


def border_candidates(view):
    """On the lateral view the descending aorta and IVC lie behind/below the heart and
    would claim the posterior contour, which is formed by LA and LV."""
    if view == "lateral":
        return [n for n in BORDER_CANDIDATES if n not in ("aorta_descending", "vena_cava_inferior")]
    return BORDER_CANDIDATES


def border_segments(masks, view="frontal"):
    """Return per-row border-forming structure for right and left cardiac border."""
    import train as T
    cand = masks[[D.IDX[n] for n in border_candidates(view)]]
    core = (masks[D.IDX["aorta_ascending"]] | masks[D.IDX["aorta_arch"]] | masks[D.IDX["heart"]])
    rows = np.nonzero(core.any(1))[0]
    rm = np.zeros(masks.shape[1], bool)
    if rows.size:
        rm[rows[0]:rows[-1] + 1] = True
    return T._border_labels(cand, rm), rm  # (2, H) index into BORDER_CANDIDATES, -1 = none


def measurements(masks):
    """CTR and chamber morphometry."""
    heart = masks[D.IDX["heart"]]
    lungs = masks[D.IDX["lung_right"]] | masks[D.IDX["lung_left"]]
    cols_h = np.nonzero(heart.any(0))[0]
    cols_t = np.nonzero(lungs.any(0))[0]
    out = {}
    if cols_h.size and cols_t.size:
        w_h = cols_h[-1] - cols_h[0] + 1
        w_t = cols_t[-1] - cols_t[0] + 1
        out["ctr"] = float(w_h / w_t)
        out["heart_width_cm"] = float(w_h / PX_PER_CM)
        out["thorax_width_cm"] = float(w_t / PX_PER_CM)
        out["heart_cols"] = (int(cols_h[0]), int(cols_h[-1]))
        out["thorax_cols"] = (int(cols_t[0]), int(cols_t[-1]))
    for n in DEPTH_ORDER:
        out[f"area_{n}_cm2"] = float(masks[D.IDX[n]].sum() / PX_PER_CM ** 2)
    tot = sum(out[f"area_{n}_cm2"] for n in DEPTH_ORDER) + 1e-6
    for n in DEPTH_ORDER:
        out[f"frac_{n}"] = out[f"area_{n}_cm2"] / tot
    # left-ventricular lateral extent beyond the mid-clavicular reference
    lv = masks[D.IDX["ventricle_left"]]
    if lv.sum() and cols_t.size:
        mid = 0.5 * (cols_t[0] + cols_t[-1])
        out["lv_left_extent_frac"] = float((np.nonzero(lv.any(0))[0][-1] - mid) / (0.5 * (cols_t[-1] - cols_t[0])))
    return out


def _contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cs


def _base(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, color="w")
    for s in ax.spines.values():
        s.set_color("0.3")


def _outline(ax, mask, color, lw=1.6, fill=0.0, label=None):
    if mask.sum() == 0:
        return
    if fill > 0:
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., :3] = color
        rgba[..., 3] = mask * fill
        ax.imshow(rgba, interpolation="nearest")
    for c in _contour(mask):
        ax.plot(c[:, 0, 0], c[:, 0, 1], color=color, lw=lw, label=label)
        label = None


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_superimposition(ax, img, masks):
    """How many cardiac structures project onto the same point."""
    _base(ax, img, "Summation: Überlagerung der Herzhöhlen")
    stack = np.stack([masks[D.IDX[n]] for n in DEPTH_ORDER]).astype(np.int32)
    count = stack.sum(0)
    cmap = plt.get_cmap("magma")
    rgba = cmap(np.clip(count / 4.0, 0, 1))
    rgba[..., 3] = np.where(count > 0, 0.55, 0.0)
    ax.imshow(rgba, interpolation="nearest")
    for n in DEPTH_ORDER:
        _outline(ax, masks[D.IDX[n]], COLORS[n], lw=1.2)
    n_over = int((count >= 2).sum())
    n_any = max(1, int((count >= 1).sum()))
    ax.text(0.02, 0.02, f"{100 * n_over / n_any:.0f} % der Herzfläche = überlagerte Höhlen",
            transform=ax.transAxes, color="w", fontsize=8,
            bbox=dict(fc="black", alpha=0.5, ec="none", pad=2))


def panel_chambers(ax, img, masks, view):
    _base(ax, img, "Herzhöhlen (anterior → posterior)")
    for n in DEPTH_ORDER:
        _outline(ax, masks[D.IDX[n]], COLORS[n], lw=1.8, fill=0.22, label=DE[n])
    for n in ["aorta_ascending", "aorta_arch", "aorta_descending", "pulmonary_trunk",
              "vena_cava_inferior"]:
        _outline(ax, masks[D.IDX[n]], COLORS[n], lw=1.1, fill=0.10, label=DE[n])
    ax.legend(loc="lower right", fontsize=6, framealpha=0.6, facecolor="black",
              labelcolor="w", edgecolor="0.4")


def panel_valves(ax, img, masks):
    _base(ax, img, "Klappenebenen (Kontaktzonen)")
    _outline(ax, masks[D.IDX["heart"]], (1, 1, 1), lw=0.8)
    for n, a, b in D.VALVE_CHANNELS:
        m = masks[D.IDX[n]]
        if m.sum() == 0:
            continue
        _outline(ax, m, COLORS[n], lw=1.4, fill=0.45, label=DE[n])
        ys, xs = np.nonzero(m)
        ax.plot(xs.mean(), ys.mean(), marker="+", color=COLORS[n], ms=10, mew=2)
    ax.legend(loc="lower right", fontsize=6, framealpha=0.6, facecolor="black",
              labelcolor="w", edgecolor="0.4")


def panel_border(ax, img, masks, view="frontal"):
    _base(ax, img, "Randbildende Strukturen der Herzsilhouette" if view == "frontal"
          else "Vordere und hintere Herzkontur (seitlich)")
    names = border_candidates(view)
    lab, rm = border_segments(masks, view)
    cand = masks[[D.IDX[n] for n in names]]
    sil = cand.any(0) & rm[:, None]
    H, W = sil.shape
    first = np.argmax(sil, axis=1)
    last = W - 1 - np.argmax(sil[:, ::-1], axis=1)
    seen = set()
    for side, xs in ((0, first), (1, last)):
        for y in range(H):
            k = lab[side, y]
            if k < 0:
                continue
            name = names[k]
            ax.plot(xs[y], y, marker="s", ms=1.6, color=COLORS[name])
            seen.add(name)
    handles = [mpatches.Patch(color=COLORS[n], label=DE[n]) for n in names if n in seen]
    ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.6,
              facecolor="black", labelcolor="w", edgecolor="0.4")


def panel_measure(ax, img, masks):
    _base(ax, img, "Messung: Herz-Thorax-Quotient")
    m = measurements(masks)
    _outline(ax, masks[D.IDX["heart"]], (1, 1, 1), lw=1.2)
    _outline(ax, masks[D.IDX["ventricle_left"]], COLORS["ventricle_left"], lw=1.6, fill=0.20)
    if "heart_cols" in m:
        h0, h1 = m["heart_cols"]; t0, t1 = m["thorax_cols"]
        y_h, y_t = 0.62 * D.IMG_SIZE, 0.80 * D.IMG_SIZE
        ax.annotate("", xy=(h0, y_h), xytext=(h1, y_h),
                    arrowprops=dict(arrowstyle="<->", color="#ff5555", lw=1.5))
        ax.annotate("", xy=(t0, y_t), xytext=(t1, y_t),
                    arrowprops=dict(arrowstyle="<->", color="#55aaff", lw=1.5))
        ref = reference_stats()
        txt = (f"CTR {m['ctr']:.2f}  (Herz {m['heart_width_cm']:.1f} cm / "
               f"Thorax {m['thorax_width_cm']:.1f} cm) – Kohorte: Median "
               f"{ref['ctr']['p50']:.2f}, p90 {ref['ctr']['p90']:.2f}\n"
               f"LV-Fläche {m['area_ventricle_left_cm2']:.0f} cm²  "
               f"({100 * m['frac_ventricle_left']:.0f} % der Höhlenfläche, "
               f"Median {100 * ref['frac_ventricle_left']['p50']:.0f} %)")
        if m["ctr"] >= ref["ctr"]["p90"]:
            txt += "\n⚠ Herzsilhouette ≥ p90 der Kohorte – deutlich verbreitert"
        elif m["ctr"] >= ref["ctr"]["p50"] + 0.04:
            txt += "\n• Herzsilhouette oberhalb des Kohortenmedians"
        if (m.get("frac_ventricle_left", 0) >= ref["frac_ventricle_left"]["p90"]
                and m.get("lv_left_extent_frac", 0) >= ref["lv_left_extent_frac"]["p50"]):
            txt += "\n⚠ LV-Anteil ≥ p90 und nach lateral ausladend – LV-betonte Vergrößerung"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, color="w", fontsize=7,
                bbox=dict(fc="black", alpha=0.55, ec="none", pad=3))
    return m


def figure_for_image(img, masks, view, title, out_path, atlas=None):
    if view == "lateral":
        # PAXRay renders laterals with the patient facing right; radiological display
        # convention is facing left, so mirror image and masks together for teaching.
        img = np.ascontiguousarray(img[:, ::-1])
        masks = np.ascontiguousarray(masks[:, :, ::-1])
        title += "  – Anzeige: Patient blickt nach links"
    panels = [panel_superimposition, panel_chambers, panel_valves, panel_border]
    if atlas:
        panels.append(panel_atlas)
    if view == "frontal":
        panels.append(panel_measure)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), facecolor="#111111")
    meas = None
    for ax, fn in zip(np.atleast_1d(axes), panels):
        if fn is panel_chambers:
            fn(ax, img, masks, view)
        elif fn is panel_atlas:
            fn(ax, img, masks, atlas, view)
        elif fn is panel_border:
            fn(ax, img, masks, view)
        elif fn is panel_measure:
            meas = fn(ax, img, masks)
        else:
            fn(ax, img, masks)
    fig.suptitle(title, color="w", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return meas


def valve_atlas(n=300, split="train_val", view="frontal", seed=3):
    """Typical valve positions in a heart-normalised coordinate frame.

    Every valve centroid is expressed relative to the bounding box of the cardiac
    silhouette, so hearts of different size and position become comparable.
    """
    recs = [r for r in D.load_records(split) if r["view"] == view]
    rng = np.random.RandomState(seed)
    sel = [recs[i] for i in rng.permutation(len(recs))[:n]]
    pts = {name: [] for name, _, _ in D.VALVE_CHANNELS}
    for r in sel:
        m = D.load_mask(r)
        h = m[D.IDX["heart"]]
        ys, xs = np.nonzero(h)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        H, W = max(1, y1 - y0), max(1, x1 - x0)
        for name, _, _ in D.VALVE_CHANNELS:
            v = m[D.IDX[name]]
            if v.sum() == 0:
                continue
            vy, vx = np.nonzero(v)
            pts[name].append(((vx.mean() - x0) / W, (vy.mean() - y0) / H))
    return {k: np.array(v) for k, v in pts.items() if len(v)}


def panel_atlas(ax, img, masks, atlas, view):
    """Overlay the population valve atlas on one case (atlas is built in the same
    heart-normalised frame, so it transfers to the mirrored lateral display too)."""
    if view == "lateral":
        atlas = {k: np.stack([1 - v[:, 0], v[:, 1]], 1) for k, v in atlas.items()}
    _base(ax, img, f"Typische Klappenlage ({'p.a.' if view == 'frontal' else 'seitlich'})")
    _outline(ax, masks[D.IDX["heart"]], (1, 1, 1), lw=1.2)
    ys, xs = np.nonzero(masks[D.IDX["heart"]])
    if ys.size == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    H, W = max(1, y1 - y0), max(1, x1 - x0)
    for name, _, _ in D.VALVE_CHANNELS:
        if name not in atlas:
            continue
        p = atlas[name]
        px, py = x0 + p[:, 0] * W, y0 + p[:, 1] * H
        ax.scatter(px, py, s=3, color=COLORS[name], alpha=0.25, edgecolors="none")
        mx, my = px.mean(), py.mean()
        sx, sy = px.std(), py.std()
        ax.add_patch(mpatches.Ellipse((mx, my), 2 * sx, 2 * sy, fill=False,
                                      edgecolor=COLORS[name], lw=1.6, label=DE[name]))
        ax.plot(mx, my, "+", color=COLORS[name], ms=9, mew=2)
    ax.legend(loc="lower right", fontsize=6, framealpha=0.6, facecolor="black",
              labelcolor="w", edgecolor="0.4")
    ax.text(0.02, 0.02, f"n = {len(next(iter(atlas.values())))} Fälle,\nEllipse = ±1 SD",
            transform=ax.transAxes, color="w", fontsize=7,
            bbox=dict(fc="black", alpha=0.5, ec="none", pad=2))


def overlap_text(masks):
    lines = []
    for a, b, inter, pct in overlap_table(masks):
        if inter > 0:
            lines.append(f"{DE[a]} ∩ {DE[b]}: {pct:.0f} % (der kleineren Struktur)")
    return lines


# ---------------------------------------------------------------------------
_ATLAS = {}


def _atlas_cache(n, view):
    if view not in _ATLAS:
        _ATLAS[view] = valve_atlas(n=n, view=view)
    return _ATLAS[view]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--image", default=None, help="file name, e.g. 0001a_frontal.png")
    ap.add_argument("--split", default="test")
    ap.add_argument("--demo", type=int, default=0, help="render N demo patients (frontal+lateral)")
    ap.add_argument("--gt", action="store_true", help="render ground-truth masks instead of predictions")
    ap.add_argument("--real", default=None,
                    help="glob of external radiographs (real CXR); rendered as frontal views")
    ap.add_argument("--view", default="frontal", choices=["frontal", "lateral"])
    ap.add_argument("--atlas", type=int, default=0,
                    help="add a population valve-atlas panel built from N training cases")
    ap.add_argument("--out", default="projects/20260723_07/teaching")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.real:
        import glob
        model = load_model(args.checkpoint)
        for path in sorted(glob.glob(args.real)):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (D.IMG_SIZE, D.IMG_SIZE):
                img = cv2.resize(img, (D.IMG_SIZE, D.IMG_SIZE), interpolation=cv2.INTER_AREA)
            _, masks = predict(model, img, args.view)
            name = os.path.basename(path)
            out = os.path.join(args.out, name.replace(".png", f"_{args.view}_Modell.png"))
            figure_for_image(img, masks, args.view, f"{name} ({args.view}, Modell)", out)
            print(out)
        return

    recs = {r["file_name"]: r for r in D.load_records(args.split)}
    model = None if args.gt else load_model(args.checkpoint)

    if args.image:
        targets = [args.image]
    else:
        # demo selection: span the range of heart sizes (normal -> enlarged)
        frontal = [r for r in recs.values() if r["view"] == "frontal"]
        frontal = sorted(frontal, key=lambda r: r["file_name"])[:400]
        ctrs = []
        for r in frontal:
            m = D.build_mask(r)
            c = measurements(m).get("ctr")
            if c:
                ctrs.append((c, r["file_name"]))
        ctrs.sort()
        n = max(1, args.demo)
        picks = [ctrs[int(i * (len(ctrs) - 1) / max(1, n - 1))] for i in range(n)]
        targets = [f for _, f in picks]

    for fn in targets:
        for view_fn in [fn, fn.replace("_frontal", "_lateral")]:
            if view_fn not in recs:
                continue
            r = recs[view_fn]
            img = D.read_image(view_fn)
            if args.gt:
                masks = D.build_mask(r)
            else:
                _, masks = predict(model, img, r["view"])
            tag = "GT" if args.gt else "Modell"
            out = os.path.join(args.out, view_fn.replace(".png", f"_{tag}.png"))
            atlas = _atlas_cache(args.atlas, r["view"]) if args.atlas else None
            meas = figure_for_image(img, masks, r["view"],
                                    f"{view_fn}  ({r['view']}, {tag})", out, atlas)
            print(out)
            if r["view"] == "frontal" and meas:
                print(f"   CTR {meas['ctr']:.2f} | " + " | ".join(overlap_text(masks)))


if __name__ == "__main__":
    main()
