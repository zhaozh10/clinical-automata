"""Assemble the ChestX-ray14 auxiliary pool.

Two uses, both about false alarms rather than about detection:
  * extra hard negatives for training -- normal apices, plus films whose only
    abnormality is somewhere else, plus lines/tubes/devices;
  * an external "does it stay quiet on films read as normal" measurement.

Any film whose perceptual signature matches a SIIM film is dropped: the two
datasets are built from the same source archive, so without this the SIIM
validation and test films would leak in through the back door.
"""
import os, numpy as np, pandas as pd, cv2
from multiprocessing import Pool

NIH_ROOT = "/path/to/your/datasets/Chest_Xray/preprocessed_512/NIH_ChestX-ray14"
NIH_IMG = os.path.join(NIH_ROOT, "CXR14", "preprocessed")
EXT = "/path/to/your/external_data/ptxsecond_v1"
SIM_T = 0.80          # conservative: drop anything even close to a SIIM film
S = 512
N_TRAIN_NEG = 20000
N_EXT_NORMAL = 4000

def load(a):
    i, p = a
    im = cv2.imread(os.path.join(NIH_IMG, p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        im = np.zeros((S, S), np.uint8)
    if im.shape != (S, S):
        im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
    return i, im

if __name__ == "__main__":
    m = pd.read_csv(os.path.join(NIH_ROOT, "final_cxr14_master_list.csv"))
    m2 = pd.read_csv(os.path.join(EXT, "siim_nih_match_v2.csv"))   # high-pass signature
    m1 = pd.read_csv(os.path.join(EXT, "siim_nih_match.csv"))       # coarse signature
    leaked_pat = (set(m2.loc[m2.sim > SIM_T, "nih_patient_id"].unique())
                  | set(m1.loc[m1.sim > 0.95, "nih_patient_id"].unique()))
    leaked_img = set(m2.loc[m2.sim > SIM_T, "nih_image_id"].unique())
    print(f"leaked films {len(leaked_img)}  leaked patients {len(leaked_pat)}", flush=True)
    # drop at PATIENT level: a different film of the same patient is still the same
    # chest and would leak
    clean = m[~m.patient_id.isin(leaked_pat)].copy()
    print("clean films", len(clean), "of", len(m), flush=True)

    rng = np.random.RandomState(1337)
    neg = clean[(clean.pneumothorax == 0) & (clean.split.isin(["train", "valid"]))]
    normal = neg[neg.no_finding == 1]
    abnormal = neg[neg.no_finding == 0]          # abnormal elsewhere, apex still clean
    n_ab = min(len(abnormal), N_TRAIN_NEG // 2)
    n_no = min(len(normal), N_TRAIN_NEG - n_ab)
    pool = pd.concat([normal.sample(n_no, random_state=1337),
                      abnormal.sample(n_ab, random_state=1337)])
    pool["role"] = "train_neg"

    ext = clean[(clean.split == "test") & (clean.no_finding == 1)]
    ext = ext.sample(min(len(ext), N_EXT_NORMAL), random_state=7).copy()
    ext["role"] = "ext_normal"

    keep = pd.concat([pool, ext], ignore_index=True)
    keep["idx"] = np.arange(len(keep))
    print(keep.role.value_counts().to_dict(), flush=True)

    arr = np.lib.format.open_memmap(os.path.join(EXT, "nih_img_512.npy"), mode="w+",
                                    dtype=np.uint8, shape=(len(keep), S, S))
    with Pool(12) as p:
        for k, (i, im) in enumerate(p.imap_unordered(load, list(enumerate(keep.img_rel_path)),
                                                     chunksize=32)):
            arr[i] = im
            if k % 4000 == 0:
                print(k, flush=True)
    arr.flush()
    keep.to_csv(os.path.join(EXT, "nih_index.csv"), index=False)
    print("done", len(keep), flush=True)
