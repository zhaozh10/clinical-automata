# MEL vs NV Classifier — Data Distribution & Usage Guide

## 1. Data Distribution

### Source
- **Dataset:** ISIC 2019 (International Skin Imaging Collaboration)
- **Task:** Binary classification — Melanoma (MEL) vs Melanocytic Nevus (NV)
- **Image modality:** Dermoscopy (polarized light, close-up skin imaging)

### Split Strategy
- **Patient-level stratified split** using `lesion_id` (prevents data leakage from multiple images of the same lesion)
- Samples without `lesion_id` are each treated as unique patients
- Train/Val: 80/20 split of ISIC_2019_Training set
- Test: ISIC_2019_Test set (held out, never touched during training or model selection)
- Random seed: 42

### Sample Counts

| Split | Total | MEL (pos) | NV (neg) | MEL Ratio | Purpose |
|---|---|---|---|---|---|
| **Train** | 13,869 | 3,623 | 10,246 | 26.1% | Model training |
| **Val** | 3,528 | 899 | 2,629 | 25.5% | Model selection, threshold tuning |
| **Test** | 3,822 | 1,327 | 2,495 | 34.7% | Final reporting only (never used for training decisions) |
| **Total** | 21,219 | 5,849 | 15,370 | 27.6% | — |

### Class Imbalance Handling
- **Weighted loss:** class weights inversely proportional to frequency (`w_NV ≈ 0.68`, `w_MEL ≈ 1.91`)
- **Weighted sampler:** each class contributes equally per batch via `WeightedRandomSampler`
- **Focal loss** (gamma=1.0): emphasizes hard examples
- **No synthetic oversampling** (e.g., SMOTE) — preserves natural image distribution

---

## 2. Model Usage Guide

### 2.1 Environment

**Requirements:**
- Python 3.10+
- NVIDIA GPU with >= 20 GB VRAM (model uses ~16 GB at inference)
- PyTorch 2.9.1 + CUDA 12.8
- `timm` for the ConvNeXt backbone
- `uv` for dependency management

**Install:**
```bash
cd ~/project_s1/automata
uv sync
```

### 2.2 Model Specification

| Attribute | Value |
|---|---|
| Backbone | `convnext_large.fb_in22k_ft_in1k` (196M params) |
| Input size | 384 x 384 RGB |
| Normalization | ImageNet mean/std |
| Output | 2 logits — `[P(NV), P(MEL)]` after softmax |
| Best checkpoint | `projects/20260408_12_melanoma_nv_dermo/run20/checkpoints/best.pth` |

### 2.3 Loading the Model

```python
import torch
import timm

CKPT = "projects/20260408_12_melanoma_nv_dermo/run20/checkpoints/best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "convnext_large.fb_in22k_ft_in1k",
    pretrained=False,
    num_classes=2,
    drop_path_rate=0.0,  # disable drop_path at inference
)
state_dict = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval().to(DEVICE)
```

### 2.4 Inference on a Single Image

```python
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image_path, model, device="cuda"):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    p_nv, p_mel = probs[0].item(), probs[1].item()
    return {"P(NV)": p_nv, "P(MEL)": p_mel}
```

### 2.5 Inference with TTA (Recommended — matches reported metrics)

```python
def predict_tta(image_path, model, device="cuda"):
    """4x TTA: original + hflip + vflip + hflip+vflip. Adds ~0.006 to AUC for free."""
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    augs = [
        x,
        torch.flip(x, [3]),          # hflip
        torch.flip(x, [2]),          # vflip
        torch.flip(x, [2, 3]),       # hflip+vflip
    ]
    probs_sum = torch.zeros(1, 2, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for aug in augs:
            probs_sum += F.softmax(model(aug), dim=1)
    probs = (probs_sum / len(augs))[0]
    return {"P(NV)": probs[0].item(), "P(MEL)": probs[1].item()}
```

### 2.6 Decision Thresholds

The raw probability `P(MEL)` alone is not a decision — you must apply a threshold. Three operating points are calibrated on the validation set:

| Threshold | Operating Point | Use Case |
|---|---|---|
| **0.6121** | **80% specificity (primary operating point)** | **Recommended default — matches the primary metric `mel_sens_at_spec80`** |
| 0.7478 | Youden's J (max sensitivity + specificity - 1) | Balanced sensitivity/specificity |
| 0.5 | Argmax of softmax | Naive default — not recommended |

**Sensitivity-first usage (recommended, matches the task's risk preference):**

```python
THRESHOLD_SPEC80 = 0.6121  # calibrated on val for 80% specificity
def classify_melanoma(image_path, model, device="cuda"):
    pred = predict_tta(image_path, model, device)
    p_mel = pred["P(MEL)"]
    flag_for_biopsy = p_mel >= THRESHOLD_SPEC80
    return {
        "P(MEL)": p_mel,
        "flag_suspicious": flag_for_biopsy,
        "threshold_used": THRESHOLD_SPEC80,
        "operating_point": "80% specificity (high sensitivity mode)",
    }
```

### 2.7 Expected Performance (Test Set, n=3822)

With 4x TTA at `threshold=0.6121` (or via `mel_sens_at_spec80` operating point):

| Metric | Estimate | 95% Bootstrap CI |
|---|---|---|
| mel_sens_at_spec80 | **0.9103** | **[0.8903, 0.9254]** |
| AUC | 0.9424 | [0.9347, 0.9497] |
| Sensitivity (Youden) | 0.8417 | [0.8105, 0.8850] |
| Specificity (Youden) | 0.8978 | [0.8523, 0.9274] |

**At the recommended 80%-specificity operating point: catches ~91% of melanomas with ~20% false alarm rate on nevi.**

### 2.8 Batch Inference on Many Images

```python
from torch.utils.data import Dataset, DataLoader

class ImageListDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.paths[i]

def predict_batch(image_paths, model, device="cuda", batch_size=32, tta=True):
    ds = ImageListDataset(image_paths, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    results = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for x, paths in loader:
            x = x.to(device, non_blocking=True)
            if tta:
                augs = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
                probs_sum = torch.zeros(x.size(0), 2, device=device)
                for aug in augs:
                    probs_sum += F.softmax(model(aug), dim=1)
                probs = probs_sum / len(augs)
            else:
                probs = F.softmax(model(x), dim=1)
            for j, p in enumerate(paths):
                results.append({"path": p,
                                "P(MEL)": probs[j, 1].item(),
                                "flag": probs[j, 1].item() >= THRESHOLD_SPEC80})
    return results
```

---

## 3. Expected Usage Scenarios

The model is designed as a **sensitivity-first assistive screening tool** for confirmed pigmented melanocytic lesions. Below are the intended deployment contexts, each mapped to where the model fits in the workflow, what its output means, and what it should NOT be used for.

### Scenario A: Dermatology Clinic Triage (Primary Use Case)

**Context:** A dermatologist performs dermoscopy on a suspicious pigmented lesion and is uncertain whether to biopsy or schedule follow-up.

- **Input:** Dermoscopic image (polarized/contact dermoscopy) of the lesion
- **Model output:** `P(MEL) >= 0.6121` -> flag as "biopsy-warranted"
- **Downstream action:**
  - Flagged lesion -> excisional biopsy or short-interval (3-month) follow-up
  - Not flagged -> routine re-evaluation at standard interval
- **Why it fits:** The 80%-specificity operating point keeps unnecessary biopsies manageable (~20% of benign nevi flagged) while catching ~91% of melanomas. The miss rate (~9%) should be offset by the clinician's own judgment — the model is a **second read**, not a replacement.

### Scenario B: Teledermatology / Remote Consultation

**Context:** A general practitioner or nurse captures a dermoscopic image and submits it for review; a remote dermatologist is queued to read cases.

- **Input:** Dermoscopic image uploaded via teledermatology platform
- **Model role:** **Prioritizes the reading queue** — cases with high `P(MEL)` are surfaced to the top so suspected melanomas get reviewed first
- **Downstream action:** Dermatologist reviews the image as usual; model output is visible as a risk score but does not override human judgment
- **Why it fits:** The model's high sensitivity means melanoma cases rarely sink to the bottom of the queue, reducing time-to-review for the most urgent cases.

### Scenario C: Resident / Trainee Decision Support

**Context:** A dermatology resident or primary-care physician-in-training is learning to interpret dermoscopy.

- **Model role:** Provides a second opinion after the trainee makes their own call; agreement/disagreement triggers discussion with the attending
- **Use pattern:** Trainee records their own assessment first (to avoid automation bias), then reveals the model score
- **Why it fits:** The model is a calibration tool during training, not a shortcut.

### Scenario D: Longitudinal Mole Monitoring

**Context:** A patient with multiple atypical nevi is followed over time; the dermatologist wants to track which lesions change most suspiciously.

- **Model role:** Applied to each follow-up dermoscopic image; a **rise in `P(MEL)` over time** is more informative than the absolute value at any one visit
- **Downstream action:** Lesions with an upward trajectory in the risk score get prioritized for biopsy at the next visit
- **Why it fits:** Relative changes over time on the same lesion are robust to calibration issues.

---

### Scenarios Where This Model Should NOT Be Used

- **General skin-cancer screening of any lesion:** the model was trained ONLY on MEL vs NV. It has never seen BCC, SCC, AK, BKL, DF, or VASC and will produce meaningless outputs on them. Use only after a dermatologist has clinically narrowed the differential to "melanocytic — melanoma or nevus."
- **Clinical (non-dermoscopic) photographs:** input must be dermoscopy. Cellphone photos of moles will degrade performance significantly.
- **Pediatric Spitz nevi and other atypical benign variants:** ISIC 2019 is adult-skewed; extrapolation to pediatric dermatology has not been validated.
- **Primary autonomous diagnosis:** the model is assistive, not diagnostic. Final diagnosis requires histopathology.
- **Consumer-facing mole-check apps:** the 20% false-positive rate on nevi would cause alarm and overuse of medical services if deployed direct-to-consumer.
- **Calibrated probability reporting:** `P(MEL)` is a ranking score. Do not report "this mole has a 73% chance of being melanoma" to a patient — the number is not calibrated to a true posterior.

---

### Operating-Point Selection Based on Use Case

| Use case | Threshold | Sensitivity | Specificity | Rationale |
|---|---|---|---|---|
| **Sensitivity-first screening (default)** | **0.6121** | **~91%** | **80%** | Primary recommended mode — aligns with "don't miss melanoma" |
| Balanced triage | 0.7478 (Youden) | ~84% | ~90% | When resource constraints force fewer false positives |
| Conservative / research | Custom (sweep ROC) | — | — | Select operating point from published ROC on val set |

### Technical Deployment Notes

- **Input resolution:** 384x384 RGB; lower resolutions will degrade performance
- **Inference latency:** ~50ms/image with TTA on a single L40S GPU; ~15ms without TTA
- **Recommended:** always enable 4x-TTA at inference — zero training cost, consistent gain
- **Patient-level guarantee:** all reported metrics are on held-out test data; no lesion in test appeared in training
