"""Downscaled copy of the film cache. The 1024 cache stays for high-resolution
experiments; this one removes the per-sample resize from the dataloader so the
GPU is not left waiting."""
import numpy as np, os, cv2
EXT = "/path/to/your/external_data/ptxsecond_v1"
S = 512
img = np.load(os.path.join(EXT, "siim_img_1024.npy"), mmap_mode="r")
msk = np.load(os.path.join(EXT, "siim_msk_1024.npy"), mmap_mode="r")
n = len(img)
oi = np.lib.format.open_memmap(os.path.join(EXT, "siim_img_512.npy"), mode="w+",
                               dtype=np.uint8, shape=(n, S, S))
om = np.lib.format.open_memmap(os.path.join(EXT, "siim_msk_512.npy"), mode="w+",
                               dtype=np.uint8, shape=(n, S * S // 8))
for i in range(n):
    oi[i] = cv2.resize(np.asarray(img[i]), (S, S), interpolation=cv2.INTER_AREA)
    m = np.unpackbits(np.asarray(msk[i]))[:1024 * 1024].reshape(1024, 1024)
    om[i] = np.packbits(cv2.resize(m * 255, (S, S), interpolation=cv2.INTER_AREA) > 127)
    if i % 2000 == 0:
        print(i, flush=True)
oi.flush(); om.flush(); print("done", n)
