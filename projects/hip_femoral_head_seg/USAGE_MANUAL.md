# Femoral-Head Segmentation — Pediatric Hip AP Radiographs

**Clinical task.** Delineate the left and right femoral heads on pediatric hip AP
X-rays (assistive segmentation). Both the region overlap and the boundary accuracy
matter, because downstream measurements (head coverage, sphericity/Perthes deformity,
dysplasia assessment) are derived from the mask and its margin.

## Result (headline)

Evaluated on the **held-out 27-case test set** (54 femoral heads), never used for
model selection. Metrics match the prior baseline's definitions
(ALL = mean over all heads; HARD = all cohorts except normal/reference).

| cohort | prior baseline (Dice) | **this model, Dice (mean±std, 3 seeds)** | 3-seed ensemble Dice |
|---|---|---|---|
| **ALL** | 0.894 | **0.928 ± 0.002** | **0.931** |
| **HARD** | 0.853 | **0.906 ± 0.004** | **0.910** |
| reference | 0.938 | 0.951 ± 0.002 | 0.954 |
| mat_removed | 0.914 | 0.929 ± 0.001 | 0.931 |
| metal | 0.882 | 0.925 ± 0.005 | 0.929 |
| morbus_perthes | 0.778 | 0.832 ± 0.009 | 0.840 |
| normal | 0.940 | 0.952 ± 0.005 | 0.955 |
| severe_dysplasia | 0.799 | 0.901 ± 0.012 | 0.906 |

The model improves **every cohort**, with the largest gains on the clinically hard
subgroups (dysplasia +0.10, metal +0.04, Perthes +0.05 Dice). Boundary quality is
strong: test NSD (surface Dice) 0.854 ALL / 0.802 HARD; normalized HD95 0.14 ALL.
Std across seeds is far smaller than the improvement → the win is robust, not noise.

## Method (winning recipe)

- **Backbone:** frozen DINOv3 ViT-L/16 (`pretrained/dinov3/dinov3_vitl16_pretrain_lvd1689m`),
  loaded via a corrected Meta→timm key remap (layerscale, register tokens, q/v-bias split).
- **Formulation (the key idea):** **bilateral 2-channel** — one full image in, two masks
  out (left head, right head). This removes the fundamental ambiguity of the per-head
  setup (two near-identical heads, one label) and lets the model use the contralateral
  hip as an internal reference. This single change moved Dice 0.62 → 0.92.
- **Decoder:** multi-scale fusion of 4 DINOv3 layers (5/11/17/23) + progressive-upsample
  conv decoder, input resolution **1024**.
- **Loss:** boundary-weighted BCE (up-weights GT-margin pixels) + Dice, presence-masked
  per channel (unilateral cases don't penalize the missing side).
- **Augmentation:** rotation/scale/translation, symmetry-aware horizontal flip (mirrors
  image **and** swaps the two head channels), brightness/contrast/gamma.
- **Selection:** primary metric **SegScore = 0.5·Dice + 0.5·NSD** on a case-level,
  cohort-stratified 20% validation split (test never touched).

Rejected after testing (kept the model simple / avoided overfitting): Tversky loss
(traded precision for recall), cohort-balanced sampling (starved easy cohorts),
high-res image-stem decoder (spurious edges near metal), flip-TTA (already flip-invariant).

## Files

- `run5/checkpoints/best.pth` — best single model (seed 0). `run_final_s1`, `run_final_s2` — seeds 1, 2.
- `test_predictions.csv` — per-head test metrics for the 3-seed ensemble (case, side, cohort, Dice, IoU, precision, recall, NSD, HD95n, ASSDn, SegScore).
- `test_gallery.png` — qualitative overlays (GT green, prediction red/yellow) across cohorts.
- `results.tsv` — full experiment log (runs 1–7 + final).
- `../../data.py`, `../../train.py` — data pipeline and model/training/eval code.

## Reproduce

```bash
uv run train.py                                  # train seed 0 (30-min budget) -> run5 config
SEED=1 RUN_TAG=_final_s1 uv run train.py          # additional seeds
```

Inference: build `SegModel`, load a `best.pth`, feed a 1024×1024 ImageNet-normalized
image (grayscale→3ch); output channel 0 = head in image-left half, channel 1 = image-right
half; sigmoid > 0.5. Averaging the 3 seeds' probabilities gives the ensemble (best).

## Limitations

- **Scope:** femoral heads only (what is annotated). Acetabulum/whole-pelvis structures
  are not modeled; migration percentage / CE-angle would need those labels.
- **No physical pixel spacing** in the source JPEGs, so boundary metrics (HD95, NSD) are
  normalized by head equivalent-diameter rather than reported in mm.
- **Small data**, especially morbus_perthes (6 training studies); its per-case metric is
  the noisiest and the most data-limited cohort.
- Numbers are for this fixed 30-min/run compute budget and this dataset split.
