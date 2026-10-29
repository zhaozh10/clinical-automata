"""Cache SIIM-ACR films + pneumothorax masks as flat memmaps at native 1024.

Decoding 12k 1024x1024 PNGs per epoch would starve the GPU; caching once lets
the dataloader pick any working resolution cheaply. The original dataset files
are only read, never modified.
"""
import os, numpy as np, pandas as pd, cv2
from multiprocessing import Pool

SIIM = "/path/to/your/datasets/siim-acr-pneumothorax"
EXT = "/path/to/your/external_data/ptxsecond_v1"
R = 1024

def load(i_fn):
    i, fn = i_fn
    img = cv2.imread(os.path.join(SIIM, "png_images", fn), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(SIIM, "png_masks", fn), cv2.IMREAD_GRAYSCALE)
    if img.shape != (R, R):
        img = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
    if msk.shape != (R, R):
        msk = cv2.resize(msk, (R, R), interpolation=cv2.INTER_NEAREST)
    return i, img, np.packbits(msk > 127)

if __name__ == "__main__":
    os.makedirs(EXT, exist_ok=True)
    tr = pd.read_csv(os.path.join(SIIM, "stage_1_train_images.csv")); tr["siim_split"] = "train"
    te = pd.read_csv(os.path.join(SIIM, "stage_1_test_images.csv")); te["siim_split"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    drain = pd.read_csv(os.path.join(SIIM, "chest_drain_predictions.csv"))
    df = df.merge(drain[["new_filename", "label", "probability"]]
                  .rename(columns={"label": "drain", "probability": "drain_prob"}),
                  on="new_filename", how="left")
    df["idx"] = np.arange(len(df))
    n = len(df)
    im = np.lib.format.open_memmap(os.path.join(EXT, "siim_img_1024.npy"), mode="w+",
                                   dtype=np.uint8, shape=(n, R, R))
    mk = np.lib.format.open_memmap(os.path.join(EXT, "siim_msk_1024.npy"), mode="w+",
                                   dtype=np.uint8, shape=(n, R * R // 8))
    with Pool(12) as p:
        for k, (i, img, msk) in enumerate(p.imap_unordered(load, list(enumerate(df.new_filename)),
                                                           chunksize=16)):
            im[i] = img; mk[i] = msk
            if k % 2000 == 0:
                print(k, flush=True)
    im.flush(); mk.flush()
    df.to_csv(os.path.join(EXT, "siim_index.csv"), index=False)
    print("done", n, df.groupby("siim_split").has_pneumo.agg(["size", "sum"]), flush=True)
