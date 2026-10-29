# Lung Lobe & Zone Segmentation — Teaching Tool for Frontal Chest X-ray

A teaching aid that, on a **frontal** chest radiograph, shows the **5 anatomical
lung lobes** and the **6 radiographic lung zones** as *separate, switchable*
overlays, and displays the model's **confidence and uncertainty** rather than
drawing every boundary as if it were certain.

The core teaching point it is built to convey: **lobes ≠ zones**, and because
lobes overlap in a 2-D projection, their boundaries on a plain film are
*inherently ambiguous* — which is exactly what the uncertainty map makes visible.

---

## What the model produces

For one frontal CXR it outputs 14 overlapping probability maps:

- **5 anatomical lobes** — `RUL, RML, RLL` (right upper/middle/lower), `LUL, LLL` (left upper/lower)
- **6 radiographic zones** — `R_upper, R_mid, R_base`, `L_upper, L_mid, L_base`
  (apical merged into upper, per request)
- **3 lung-context maps** — whole lung, right lung, left lung

It is a **multi-label** model (each structure is an independent map, not a single
"which-region-is-this-pixel" label), so a pixel can belong to a lobe *and* a zone
*and* the lung at once — the honest representation of a projection where these
structures overlap.

Two confidence signals are shown:
1. **Confidence** — the predicted probability, rendered as *graded transparency*
   (opaque = confident, faint = unsure). Boundaries are never drawn as hard lines.
2. **Uncertainty** — repeated stochastic ("MC-dropout") passes; bright areas are
   where the model disagrees with itself, i.e. the fissure / lobe-overlap regions.

---

## Held-out test performance (1,110 frontal films, never used in training)

| Metric | Value |
|---|---|
| **Mean Dice (lobes + zones)** | **0.932** |
| Lobe Dice | 0.939 |
| Zone Dice | 0.926 |
| Mean IoU | 0.872 |
| Calibration error (ECE) | 0.016 |
| Brier score | 0.021 |

Per-structure Dice (test): RUL 0.954 · RML 0.915 · RLL 0.933 · LUL 0.954 ·
LLL 0.937 · R_upper 0.919 · R_mid 0.934 · R_base 0.922 · L_upper 0.927 ·
L_mid 0.937 · L_base 0.918.

The **right middle lobe (0.915)** is the weakest — expected, as it is thin and the
most overlapped lobe in frontal projection. Validation Dice was 0.9325 vs test
0.9317, i.e. it generalizes without overfitting. The low ECE means the displayed
confidence values are trustworthy.

---

## How to use it

Predict on a single frontal CXR and render overlays:

```bash
# lobe overlay + uncertainty
uv run predict.py path/to/frontal.png --group lobe  --alpha 0.55 --out lobes.png
# same image, zone overlay (switch the concept)
uv run predict.py path/to/frontal.png --group zone  --alpha 0.55 --out zones.png
```

- `--group lobe|zone` — **switch between anatomical lobes and radiographic zones**.
- `--alpha 0.0–1.0` — **adjustable transparency** of the overlay.
- `--mc N` — number of MC-dropout passes for the uncertainty map (default 15).

Each run writes a 3-panel image: original film | colored overlay (transparency =
confidence) | MC-dropout uncertainty heatmap.

---

## Files in this project

| File | Contents |
|---|---|
| `test_predictions.csv` | One row per test film: per-structure Dice, GT/pred pixel counts, mean confidence, mean uncertainty, and `mean_uncertainty` per case. For statistics / error analysis. |
| `test_gallery/` | 12 rendered 4-panel examples spanning low→high model uncertainty. |
| `results.tsv` | Full experiment log (baseline → best), kept/discarded runs. |
| `run13/checkpoints/best.pth` | The best model (the one used for all outputs). |
| `../predict.py` | Single-image teaching predictor (above). |
| `../lobezone_infer.py` | Shared inference + overlay rendering. |
| `../infer.py` | Reproduces the test metrics, CSV, prob-maps, and gallery. |

Per-sample probability + uncertainty maps for all 1,110 test films (compact `.npz`,
for an interactive UI to render with adjustable transparency) are in
`/path/to/your/datasets/lobezone_outputs/test_prob_maps/`.

---

## Important limitations (please teach these alongside the tool)

- **The lobe "ground truth" is a CT-derived projection, not a film-traceable
  boundary.** On a real plain frontal film a radiologist cannot draw exact lobar
  fissures; the labels here come from CT projected into the X-ray plane. The tool
  is therefore best used to teach *where lobes project* and *why plain-film lobe
  localization is uncertain* — **not** as a claim that these are definite
  boundaries. The uncertainty overlay is the honest core of the lesson.
- **Frontal films only.** Laterals were excluded (lobe/zone teaching is done on the
  frontal view here).
- **Zones are radiographic descriptors** (horizontal bands), not anatomy — that
  contrast with the anatomical lobes is the intended teaching payload.
- **Educational / assistive only.** This is not a diagnostic device and makes no
  clinical decision.

---

## Reproduce

```bash
uv run train.py     # trains best recipe (resnet34 U-Net, LR 5e-4, dropout 0.1,
                    #  light aug, cosine LR) for the fixed time budget
uv run infer.py     # test metrics + CSV + prob-maps + gallery
```
