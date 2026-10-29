"""Run the frozen anatomy locator over every cached film and store its zone masks.

Also records, for each SIIM film, how much of each anatomical zone the pneumothorax
mask actually covers -- this fraction is what turns a pixel-level annotation into a
genuine per-side region label.
"""
import os, json, sys, numpy as np, pandas as pd, torch, cv2
import segmentation_models_pytorch as smp

EXT = "/path/to/your/external_data/ptxsecond_v1"
NCH, S = 7, 512

def load_model(dev="cuda:0"):
    ck = torch.load(os.path.join(EXT, "anatomy_unet.pth"), map_location="cpu")
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=NCH)
    m.load_state_dict(ck["sd"]); m.eval().to(dev)
    print("anatomy val dice", [round(d, 3) for d in ck["dice"]], flush=True)
    return m

@torch.no_grad()
def infer(model, imgs, dev="cuda:0", bs=32):
    """imgs: (n,H,W) uint8 memmap at any resolution -> packed (n, NCH*S*S/8)."""
    n = len(imgs)
    out = np.zeros((n, NCH * S * S // 8), np.uint8)
    for i in range(0, n, bs):
        batch = []
        for j in range(i, min(i + bs, n)):
            a = np.asarray(imgs[j])
            if a.shape != (S, S):
                a = cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA)
            batch.append(a)
        x = torch.from_numpy(np.stack(batch)).float().div_(255.).sub_(0.5).div_(0.25)
        x = x[:, None].to(dev).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", torch.bfloat16):
            p = torch.sigmoid(model(x).float()) > 0.5
        p = p.cpu().numpy()
        for k in range(p.shape[0]):
            out[i + k] = np.packbits(p[k])
        if i % 3200 == 0:
            print(i, "/", n, flush=True)
    return out

def unpack(row):
    return np.unpackbits(row)[: NCH * S * S].reshape(NCH, S, S).astype(bool)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "siim"
    model = load_model()
    if which == "siim":
        imgs = np.load(os.path.join(EXT, "siim_img_1024.npy"), mmap_mode="r")
        msks = np.load(os.path.join(EXT, "siim_msk_1024.npy"), mmap_mode="r")
        z = infer(model, imgs)
        np.save(os.path.join(EXT, "siim_zones_512.npy"), z)
        df = pd.read_csv(os.path.join(EXT, "siim_index.csv"))
        names = ["right_lung", "left_lung", "right_apex", "left_apex",
                 "right_base", "left_base", "heart"]
        recs = []
        for i in range(len(df)):
            zz = unpack(z[i])
            m1024 = np.unpackbits(np.asarray(msks[i]))[:1024 * 1024].reshape(1024, 1024)
            m = cv2.resize(m1024, (S, S), interpolation=cv2.INTER_AREA) > 0
            r = {"idx": i, "ptx_px": int(m.sum())}
            for ci, nm in enumerate(names):
                a = int(zz[ci].sum()); o = int((zz[ci] & m).sum())
                r[nm + "_area"] = a
                r[nm + "_ovl"] = o
                r[nm + "_frac"] = o / a if a > 0 else 0.0
            recs.append(r)
            if i % 2000 == 0:
                print("stats", i, flush=True)
        pd.DataFrame(recs).to_csv(os.path.join(EXT, "siim_zone_stats.csv"), index=False)
        print("siim zones done", flush=True)
    else:
        imgs = np.load(os.path.join(EXT, "nih_img_512.npy"), mmap_mode="r")
        z = infer(model, imgs)
        np.save(os.path.join(EXT, "nih_zones_512.npy"), z)
        print("nih zones done", flush=True)
