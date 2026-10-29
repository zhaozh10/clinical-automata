"""Perceptual-hash matching between SIIM-ACR pneumothorax films and NIH ChestX-ray14.

SIIM-ACR is derived from the same source DICOMs as ChestX-ray14, so any NIH film we
use as an extra negative may in fact be a SIIM validation/test film. This script
builds a 32x32 normalized signature for every film in both datasets and reports the
best NIH match for each SIIM film, so downstream code can drop the overlap.
"""
import os, glob, numpy as np, pandas as pd, cv2, torch
from multiprocessing import Pool

EXT = "/path/to/your/external_data/ptxsecond_v1"
SIIM = "/path/to/your/datasets/siim-acr-pneumothorax"
NIH_ROOT = "/path/to/your/datasets/Chest_Xray/preprocessed_512/NIH_ChestX-ray14"
NIH_IMG = os.path.join(NIH_ROOT, "CXR14", "preprocessed")
S = 64

def sig(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return np.zeros(S * S, np.float32)
    im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
    # high-pass: every chest film has the same low-frequency silhouette, so the
    # coarse image is not discriminative. The residual texture is.
    im = im - cv2.GaussianBlur(im, (0, 0), 4.0)
    im -= im.mean()
    n = np.linalg.norm(im) + 1e-6
    return (im / n).ravel()

def run(paths, nproc=12):
    with Pool(nproc) as p:
        out = p.map(sig, paths, chunksize=64)
    return np.stack(out)

if __name__ == "__main__":
    os.makedirs(EXT, exist_ok=True)
    siim_files = sorted(os.listdir(os.path.join(SIIM, "png_images")))
    siim_paths = [os.path.join(SIIM, "png_images", f) for f in siim_files]
    print("siim", len(siim_paths), flush=True)

    master = pd.read_csv(os.path.join(NIH_ROOT, "final_cxr14_master_list.csv"))
    nih_paths = [os.path.join(NIH_IMG, p) for p in master.img_rel_path]
    print("nih", len(nih_paths), flush=True)

    A = run(siim_paths); print("siim sigs done", A.shape, flush=True)
    B = run(nih_paths);  print("nih sigs done", B.shape, flush=True)

    dev = "cuda:0"
    At = torch.from_numpy(A).to(dev)
    Bt = torch.from_numpy(B).to(dev)
    best_sim = torch.full((At.shape[0],), -2.0, device=dev)
    best_idx = torch.zeros(At.shape[0], dtype=torch.long, device=dev)
    for i in range(0, Bt.shape[0], 4096):
        chunk = Bt[i:i + 4096]
        s = At @ chunk.T
        v, j = s.max(1)
        upd = v > best_sim
        best_sim[upd] = v[upd]
        best_idx[upd] = j[upd] + i
    bs = best_sim.cpu().numpy(); bi = best_idx.cpu().numpy()

    df = pd.DataFrame({
        "new_filename": siim_files,
        "nih_image_id": master.image_id.values[bi],
        "nih_patient_id": master.patient_id.values[bi],
        "sim": bs,
    })
    df.to_csv(os.path.join(EXT, "siim_nih_match_v2.csv"), index=False)
    np.save(os.path.join(EXT, "nih_sig64.npy"), B)
    for t in [0.999, 0.995, 0.99, 0.98, 0.95, 0.9]:
        print(f"sim>{t}: {(bs > t).sum()} / {len(bs)}", flush=True)
    print(df.sort_values("sim", ascending=False).head(10).to_string())
    print(df.sim.describe())
