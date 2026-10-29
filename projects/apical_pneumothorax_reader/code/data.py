"""Data layer for the apical-zone pneumothorax second-reader.

Unit of the clinical task is a *zone*, not a film: each frontal film contributes a
left and a right apical zone. Pixel-level pneumothorax masks (SIIM-ACR) intersected
with an anatomical apical-zone mask (predicted once by the frozen PAXRay++ locator)
give a genuine per-side label, which is what makes region-level PPV/recall reportable.
"""
import os, numpy as np, pandas as pd, cv2, torch
from torch.utils.data import Dataset, DataLoader

EXT = "/path/to/your/external_data/ptxsecond_v1"
ZONE_NAMES = ["right_lung", "left_lung", "right_apex", "left_apex",
              "right_base", "left_base", "heart"]
APEX_CH = [2, 3]                      # right apex, left apex
CARRY_CH = [2, 3, 4, 5]               # apices + lung bases (bases = exploratory only)
ZS = 512                              # resolution the zone masks were stored at

# ---- frozen evaluation definition -------------------------------------------------
POS_FRAC = 0.01      # zone is positive when pneumothorax covers >=1% of the zone
IGN_FRAC = 1e-9      # any non-zero overlap below POS_FRAC is ignored, not counted as negative
VAL_FRAC = 0.20
SPLIT_SEED = 1337
# -----------------------------------------------------------------------------------


def load_index():
    """Film-level table with split assignment and per-zone pneumothorax coverage."""
    df = pd.read_csv(os.path.join(EXT, "siim_index.csv"))
    st = pd.read_csv(os.path.join(EXT, "siim_zone_stats.csv"))
    df = df.merge(st, on="idx", how="left")
    df["drain"] = df["drain"].fillna(0).astype(int)

    df["split"] = "test"
    tr = df.index[df.siim_split == "train"].to_numpy()
    # stratify on (pneumothorax, drain) so the drain-free positives -- the ones the
    # primary metric is computed on -- are spread evenly over train and val
    strat = (df.loc[tr, "has_pneumo"].astype(str) + "_" + df.loc[tr, "drain"].astype(str)).to_numpy()
    rng = np.random.RandomState(SPLIT_SEED)
    val_ids = []
    for s in np.unique(strat):
        ids = tr[strat == s]
        ids = ids[rng.permutation(len(ids))]
        val_ids.append(ids[: int(round(VAL_FRAC * len(ids)))])
    val_ids = np.concatenate(val_ids)
    df.loc[tr, "split"] = "train"
    df.loc[val_ids, "split"] = "val"

    for side, nm in zip(["r", "l"], ["right_apex", "left_apex"]):
        f = df[nm + "_frac"].fillna(0.0).to_numpy()
        y = np.zeros(len(df), np.int64)
        y[f >= POS_FRAC] = 1
        y[(f > IGN_FRAC) & (f < POS_FRAC)] = -1      # ignored
        df["apex_" + side] = y
    # a pneumothorax that already carries a chest drain is a known, treated finding:
    # alerting on it is not the clinical task, so it is dropped from the metric
    df["eval_ok"] = ~((df.has_pneumo == 1) & (df.drain == 1))
    return df


def zone_masks(zrows, idx, ch=CARRY_CH, out=None):
    """Unpack the stored anatomical zone masks for one film."""
    z = np.unpackbits(np.asarray(zrows[idx]))[: 7 * ZS * ZS].reshape(7, ZS, ZS)
    z = z[ch].astype(np.float32)
    if out is not None and out != ZS:
        z = np.stack([cv2.resize(c, (out, out), interpolation=cv2.INTER_NEAREST) for c in z])
    return z


def apex_crop_box(z, S, margin=0.35):
    """Square crop around both apical zones, with margin, from a (K,S,S) zone stack."""
    u = (z[0] + z[1]) > 0.5
    ys, xs = np.nonzero(u)
    if len(ys) < 10:
        return 0, 0, S, S
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    side = int(max(h, w) * (1 + 2 * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    # apical disease sits at the very top of the film, so bias the box upward
    cy = int(cy - 0.10 * side)
    side = int(min(side, S))
    y0 = int(np.clip(cy - side // 2, 0, S - side))
    x0 = int(np.clip(cx - side // 2, 0, S - side))
    return y0, x0, side, side


class SiimZoneDS(Dataset):
    def __init__(self, df, res=512, train=True, aug=None, crop=None):
        self.df = df.reset_index(drop=True)
        self.res = res
        self.train = train
        self.aug = aug or {}
        self.crop = crop
        self.img = self.msk = self.zon = None

    def _open(self):
        if self.img is None:
            # a pre-downscaled copy exists for <=512 work; using it keeps the
            # dataloader from becoming the bottleneck (identical pixels either way)
            self.src = 512 if self.res <= 512 else 1024
            self.img = np.load(os.path.join(EXT, f"siim_img_{self.src}.npy"), mmap_mode="r")
            self.msk = np.load(os.path.join(EXT, f"siim_msk_{self.src}.npy"), mmap_mode="r")
            self.zon = np.load(os.path.join(EXT, "siim_zones_512.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        self._open()
        r = self.df.iloc[i]
        k = int(r["idx"])
        R = self.res
        Sc = self.src
        img = np.asarray(self.img[k])
        m = np.unpackbits(np.asarray(self.msk[k]))[:Sc * Sc].reshape(Sc, Sc)
        if R != Sc:
            img = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
            m = (cv2.resize(m * 255, (R, R), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
        z = zone_masks(self.zon, k, out=R)
        if self.crop == "apex":
            y0, x0, hh, ww = apex_crop_box(z, R)
            img = cv2.resize(img[y0:y0 + hh, x0:x0 + ww], (R, R), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m[y0:y0 + hh, x0:x0 + ww] * 255, (R, R),
                           interpolation=cv2.INTER_NEAREST) > 127
            m = m.astype(np.uint8)
            z = np.stack([cv2.resize(c[y0:y0 + hh, x0:x0 + ww], (R, R),
                                     interpolation=cv2.INTER_NEAREST) for c in z])
        img = img.astype(np.float32) / 255.0
        m = m.astype(np.float32)

        if self.train:
            a = self.aug
            if a.get("affine", True):
                ang = np.random.uniform(-a.get("rot", 10), a.get("rot", 10))
                sc = np.random.uniform(*a.get("scale", (0.90, 1.12)))
                t = np.random.uniform(-a.get("shift", 0.05), a.get("shift", 0.05), 2) * R
                M = cv2.getRotationMatrix2D((R / 2, R / 2), ang, sc); M[:, 2] += t
                img = cv2.warpAffine(img, M, (R, R), flags=cv2.INTER_LINEAR)
                m = cv2.warpAffine(m, M, (R, R), flags=cv2.INTER_NEAREST)
                z = np.stack([cv2.warpAffine(c, M, (R, R), flags=cv2.INTER_NEAREST) for c in z])
            g = a.get("gamma", (0.75, 1.35))
            img = np.clip(img, 0, 1) ** np.random.uniform(*g)
            b = a.get("bright", 0.10); c = a.get("contrast", 0.15)
            img = np.clip(img * np.random.uniform(1 - c, 1 + c)
                          + np.random.uniform(-b, b), 0, 1)
            if np.random.rand() < a.get("p_blur", 0.15):
                img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.4, 1.2))
            if np.random.rand() < a.get("p_noise", 0.15):
                img = np.clip(img + np.random.randn(R, R).astype(np.float32) * 0.02, 0, 1)

        img = (img - 0.5) / 0.25
        return (torch.from_numpy(img[None].astype(np.float32)),
                torch.from_numpy(m[None].astype(np.float32)),
                torch.from_numpy(z.astype(np.float32)),
                torch.tensor([r["apex_r"], r["apex_l"]], dtype=torch.long),
                torch.tensor(int(r["idx"])))


def build_loaders(res=512, bs=16, workers=8, aug=None, train_df=None, val_df=None):
    df = load_index()
    tr = train_df if train_df is not None else df[df.split == "train"]
    va = val_df if val_df is not None else df[(df.split == "val") & df.eval_ok]
    dl_tr = DataLoader(SiimZoneDS(tr, res, True, aug), batch_size=bs, shuffle=True,
                       num_workers=workers, pin_memory=True, drop_last=True,
                       persistent_workers=workers > 0)
    dl_va = DataLoader(SiimZoneDS(va, res, False), batch_size=max(4, bs), shuffle=False,
                       num_workers=max(2, workers // 2), pin_memory=True)
    return df, tr, va, dl_tr, dl_va


if __name__ == "__main__":
    df = load_index()
    print(df.groupby("split").size())
    for sp in ["train", "val", "test"]:
        d = df[df.split == sp]
        de = d[d.eval_ok]
        pos = ((de.apex_r == 1).sum() + (de.apex_l == 1).sum())
        ign = ((de.apex_r == -1).sum() + (de.apex_l == -1).sum())
        neg = ((de.apex_r == 0).sum() + (de.apex_l == 0).sum())
        print(f"{sp:5s} films {len(d):5d} eval_films {len(de):5d} "
              f"apex zones +{pos} -{neg} ign{ign} prev {pos/max(pos+neg,1):.4f} "
              f"| ptx films {int(d.has_pneumo.sum())} drainfree_ptx {int(((d.has_pneumo==1)&(d.drain==0)).sum())}")


class NihNegDS(Dataset):
    """ChestX-ray14 films with no pneumothorax: extra normal anatomy, devices and
    abnormalities-elsewhere for the model to learn to stay quiet on. De-duplicated
    against SIIM at patient level before selection."""
    def __init__(self, res=512, train=True, aug=None, role="train_neg", n=None,
                 seed=0, crop=None):
        d = pd.read_csv(os.path.join(EXT, "nih_index.csv"))
        d = d[d.role == role].reset_index(drop=True)
        if n is not None and n < len(d):
            d = d.sample(n, random_state=seed).reset_index(drop=True)
        self.df = d
        self.res, self.train, self.aug, self.crop = res, train, aug or {}, crop
        self.img = self.zon = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self.img is None:
            self.img = np.load(os.path.join(EXT, "nih_img_512.npy"), mmap_mode="r")
            self.zon = np.load(os.path.join(EXT, "nih_zones_512.npy"), mmap_mode="r")
        k = int(self.df.iloc[i]["idx"]); R = self.res
        img = np.asarray(self.img[k])
        if R != 512:
            img = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
        z = zone_masks(self.zon, k, out=R)
        m = np.zeros((R, R), np.float32)
        if self.crop == "apex":
            y0, x0, hh, ww = apex_crop_box(z, R)
            img = cv2.resize(img[y0:y0 + hh, x0:x0 + ww], (R, R), interpolation=cv2.INTER_LINEAR)
            z = np.stack([cv2.resize(c[y0:y0 + hh, x0:x0 + ww], (R, R),
                                     interpolation=cv2.INTER_NEAREST) for c in z])
        img = img.astype(np.float32) / 255.0
        if self.train:
            a = self.aug
            if a.get("affine", True):
                ang = np.random.uniform(-a.get("rot", 10), a.get("rot", 10))
                sc = np.random.uniform(*a.get("scale", (0.90, 1.12)))
                t = np.random.uniform(-a.get("shift", 0.05), a.get("shift", 0.05), 2) * R
                M = cv2.getRotationMatrix2D((R / 2, R / 2), ang, sc); M[:, 2] += t
                img = cv2.warpAffine(img, M, (R, R), flags=cv2.INTER_LINEAR)
                z = np.stack([cv2.warpAffine(c, M, (R, R), flags=cv2.INTER_NEAREST) for c in z])
            img = np.clip(img, 0, 1) ** np.random.uniform(*a.get("gamma", (0.75, 1.35)))
            b = a.get("bright", 0.10); c = a.get("contrast", 0.15)
            img = np.clip(img * np.random.uniform(1 - c, 1 + c)
                          + np.random.uniform(-b, b), 0, 1)
        img = (img - 0.5) / 0.25
        return (torch.from_numpy(img[None].astype(np.float32)),
                torch.from_numpy(m[None]),
                torch.from_numpy(z.astype(np.float32)),
                torch.tensor([0, 0], dtype=torch.long),
                torch.tensor(-1 - k))


# ----------------------------------------------------------------------------------
# Per-side apical crop
#
# Both apices together span nearly the full width of the film, so cropping "the
# apical region" barely magnifies anything. Cropping ONE apex at a time gives about
# a two-fold magnification of the exact area the clinician asked to be re-checked,
# and lets the film be examined the way a second reader examines it: one blind spot
# at a time. Left crops are mirrored so every crop presents the same handedness,
# which doubles the effective number of training examples per side.
# ----------------------------------------------------------------------------------

def side_box(zside, S, margin=0.35, up=0.10):
    ys, xs = np.nonzero(zside > 0.5)
    if len(ys) < 10:
        return 0, 0, S
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
    side = int(np.clip(side, 32, S))
    cy = int((y0 + y1) // 2 - up * side)
    cx = int((x0 + x1) // 2)
    return (int(np.clip(cy - side // 2, 0, S - side)),
            int(np.clip(cx - side // 2, 0, S - side)), side)


class ApexSideDS(Dataset):
    """One sample = one apex of one film, read from the pre-cut crop cache."""
    CS = 512

    def __init__(self, df, res=512, train=True, aug=None, source="siim", src_res=1024,
                 margin=0.35, gpu_aug=False):
        self.df = df.reset_index(drop=True)
        self.res, self.train, self.aug = res, train, aug or {}
        self.source = source
        self.gpu_aug = gpu_aug
        self.img = self.msk = self.zon = None

    def __len__(self):
        return 2 * len(self.df)

    def _open(self):
        if self.img is not None:
            return
        t = self.source
        self.img = np.load(os.path.join(EXT, f"{t}_apexcrop_img.npy"), mmap_mode="r")
        self.msk = np.load(os.path.join(EXT, f"{t}_apexcrop_msk.npy"), mmap_mode="r")
        self.zon = np.load(os.path.join(EXT, f"{t}_apexcrop_zon.npy"), mmap_mode="r")

    def __getitem__(self, i):
        self._open()
        f, s = i // 2, i % 2
        r = self.df.iloc[f]
        k = 2 * int(r["idx"]) + s
        R, C = self.res, self.CS
        lab = 0 if self.source == "nih" else int(r["apex_r"] if s == 0 else r["apex_l"])
        if self.gpu_aug:
            # hand the raw crop over untouched; warping and intensity jitter happen on
            # the GPU in one batched call, which keeps the 12 CPU cores from capping
            # training throughput
            trip = np.stack([np.asarray(self.img[k]),
                             np.unpackbits(np.asarray(self.msk[k]))[:C * C].reshape(C, C) * 255,
                             np.unpackbits(np.asarray(self.zon[k]))[:C * C].reshape(C, C) * 255])
            return (torch.from_numpy(trip),
                    torch.tensor([lab], dtype=torch.long),
                    torch.tensor(int(r["idx"]) if self.source == "siim" else -1 - int(r["idx"])))
        img = np.asarray(self.img[k]).astype(np.float32) / 255.
        m = np.unpackbits(np.asarray(self.msk[k]))[:C * C].reshape(C, C).astype(np.float32)
        zc = np.unpackbits(np.asarray(self.zon[k]))[:C * C].reshape(C, C).astype(np.float32)
        if R != C:
            img = cv2.resize(img, (R, R), interpolation=cv2.INTER_LINEAR)
            m = (cv2.resize(m, (R, R), interpolation=cv2.INTER_NEAREST))
            zc = cv2.resize(zc, (R, R), interpolation=cv2.INTER_NEAREST)

        if self.train:
            a = self.aug
            if a.get("affine", True):
                ang = np.random.uniform(-a.get("rot", 10), a.get("rot", 10))
                sc = np.random.uniform(*a.get("scale", (0.90, 1.12)))
                t = np.random.uniform(-a.get("shift", 0.05), a.get("shift", 0.05), 2) * R
                M = cv2.getRotationMatrix2D((R / 2, R / 2), ang, sc); M[:, 2] += t
                img = cv2.warpAffine(img, M, (R, R), flags=cv2.INTER_LINEAR)
                m = cv2.warpAffine(m, M, (R, R), flags=cv2.INTER_NEAREST)
                zc = cv2.warpAffine(zc, M, (R, R), flags=cv2.INTER_NEAREST)
            img = np.clip(img, 0, 1) ** np.random.uniform(*a.get("gamma", (0.75, 1.35)))
            b = a.get("bright", 0.10); c = a.get("contrast", 0.15)
            img = np.clip(img * np.random.uniform(1 - c, 1 + c)
                          + np.random.uniform(-b, b), 0, 1)
            if np.random.rand() < a.get("p_noise", 0.15):
                img = np.clip(img + np.random.randn(R, R).astype(np.float32) * 0.02, 0, 1)
        img = (img - 0.5) / 0.25
        fid = int(r["idx"]) if self.source == "siim" else -1 - int(r["idx"])
        return (torch.from_numpy(img[None].astype(np.float32)),
                torch.from_numpy(m[None].astype(np.float32)),
                torch.from_numpy(zc[None].astype(np.float32)),
                torch.tensor([lab], dtype=torch.long),
                torch.tensor(fid))
