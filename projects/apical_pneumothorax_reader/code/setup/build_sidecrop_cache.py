"""Pre-cut the per-side apical crops.

Cutting them on the fly meant every sample pulled a full 1 MB film off disk and
resized it twice, which left the GPU waiting. Cutting once turns each sample into
a 256 KB read, so the time budget is spent training instead of loading.
Row order is film*2 + side (0 = right apex, 1 = left apex, mirrored to match).
"""
import os, sys, numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, os.getcwd())
import data as D

EXT = D.EXT
R, SRC, MARGIN = 512, 1024, 0.35
img = msk = zon = None
NIH = len(sys.argv) > 1 and sys.argv[1] == "nih"

def init():
    global img, msk, zon
    if NIH:
        img = np.load(os.path.join(EXT, "nih_img_512.npy"), mmap_mode="r")
        zon = np.load(os.path.join(EXT, "nih_zones_512.npy"), mmap_mode="r")
    else:
        img = np.load(os.path.join(EXT, f"siim_img_{SRC}.npy"), mmap_mode="r")
        msk = np.load(os.path.join(EXT, f"siim_msk_{SRC}.npy"), mmap_mode="r")
        zon = np.load(os.path.join(EXT, "siim_zones_512.npy"), mmap_mode="r")

def one(k):
    z_all = np.unpackbits(np.asarray(zon[k]))[: 7 * 512 * 512].reshape(7, 512, 512)
    f_full = np.asarray(img[k])
    src = 512 if NIH else SRC
    sf = src // 512
    m_full = (None if NIH else
              np.unpackbits(np.asarray(msk[k]))[:src * src].reshape(src, src))
    out = []
    for s in range(2):
        y0, x0, sd = D.side_box(z_all[2 + s], 512, MARGIN)
        im = cv2.resize(f_full[y0 * sf:(y0 + sd) * sf, x0 * sf:(x0 + sd) * sf],
                        (R, R), interpolation=cv2.INTER_LINEAR)
        zz = cv2.resize(z_all[2 + s][y0:y0 + sd, x0:x0 + sd], (R, R),
                        interpolation=cv2.INTER_NEAREST) > 0
        if m_full is None:
            mm = np.zeros((R, R), bool)
        else:
            mm = cv2.resize(m_full[y0 * sf:(y0 + sd) * sf, x0 * sf:(x0 + sd) * sf] * 255,
                            (R, R), interpolation=cv2.INTER_AREA) > 127
        if s == 1:
            im, mm, zz = im[:, ::-1], mm[:, ::-1], zz[:, ::-1]
        out.append((np.ascontiguousarray(im), np.packbits(mm), np.packbits(zz)))
    return k, out

if __name__ == "__main__":
    tag = "nih" if NIH else "siim"
    ref = "nih_img_512.npy" if NIH else "siim_img_1024.npy"
    n = len(np.load(os.path.join(EXT, ref), mmap_mode="r"))
    oi = np.lib.format.open_memmap(os.path.join(EXT, f"{tag}_apexcrop_img.npy"), mode="w+",
                                   dtype=np.uint8, shape=(n * 2, R, R))
    om = np.lib.format.open_memmap(os.path.join(EXT, f"{tag}_apexcrop_msk.npy"), mode="w+",
                                   dtype=np.uint8, shape=(n * 2, R * R // 8))
    oz = np.lib.format.open_memmap(os.path.join(EXT, f"{tag}_apexcrop_zon.npy"), mode="w+",
                                   dtype=np.uint8, shape=(n * 2, R * R // 8))
    with Pool(12, initializer=init) as p:
        for c, (k, out) in enumerate(p.imap_unordered(one, range(n), chunksize=8)):
            for s in range(2):
                oi[2 * k + s], om[2 * k + s], oz[2 * k + s] = out[s]
            if c % 2000 == 0:
                print(c, flush=True)
    oi.flush(); om.flush(); oz.flush()
    print("done", tag, n * 2, flush=True)
