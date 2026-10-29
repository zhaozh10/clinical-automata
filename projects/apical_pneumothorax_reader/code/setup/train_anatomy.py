"""Train the anatomy locator: 7 zone masks from a frontal chest film.

This model is *not* the clinical model. Its only job is to tell the pneumothorax
model where the apical zones (and, for the exploratory regions, the lung bases,
lungs and heart) are on an arbitrary frontal film, so that region evidence can be
pooled inside an anatomically defined box rather than a fixed rectangle.
Trained once during setup and then frozen.
"""
import os, json, time, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

PAX_IMG = "/path/to/your/datasets/pax_ray++/paxray_images_unfiltered/images_patlas"
EXT = "/path/to/your/external_data/ptxsecond_v1"
S = 512
NCH = 7

class PaxZones(Dataset):
    def __init__(self, items, train):
        self.items = items; self.train = train; self.arr = None
    def __len__(self): return len(self.items)
    def _a(self):
        if self.arr is None:
            self.arr = np.load(os.path.join(EXT, "paxray_zones.npy"), mmap_mode="r")
        return self.arr
    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(os.path.join(PAX_IMG, it["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((S, S), np.uint8)
        if img.shape != (S, S):
            img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        packed = np.asarray(self._a()[it["idx"]])
        m = np.unpackbits(packed)[: NCH * S * S].reshape(NCH, S, S).astype(np.float32)
        img = img.astype(np.float32) / 255.0
        if self.train:
            ang = np.random.uniform(-8, 8); sc = np.random.uniform(0.90, 1.12)
            tx, ty = np.random.uniform(-0.05, 0.05, 2) * S
            M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc); M[:, 2] += (tx, ty)
            img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0.0)
            m = np.stack([cv2.warpAffine(c, M, (S, S), flags=cv2.INTER_NEAREST,
                                         borderValue=0.0) for c in m])
            # intensity jitter, so the locator survives the SIIM / NIH windowing shift
            img = np.clip(img, 0, 1) ** np.random.uniform(0.6, 1.6)
            img = np.clip(img * np.random.uniform(0.8, 1.25)
                          + np.random.uniform(-0.12, 0.12), 0, 1)
            if np.random.rand() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.5, 1.6))
            if np.random.rand() < 0.3:
                img = np.clip(img + np.random.randn(S, S).astype(np.float32) * 0.03, 0, 1)
            if np.random.rand() < 0.3:
                img = cv2.equalizeHist((img * 255).astype(np.uint8)).astype(np.float32) / 255.
        img = (img - 0.5) / 0.25
        return torch.from_numpy(img[None].astype(np.float32)), torch.from_numpy(m)

def dice_loss(logit, tgt, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * tgt).sum((2, 3)) + eps
    den = p.sum((2, 3)) + tgt.sum((2, 3)) + eps
    return (1 - num / den).mean()

def main():
    idx = json.load(open(os.path.join(EXT, "paxray_zones_index.json")))
    items = idx["items"]
    tr = [it for it in items if it["split"] == "trainval"]
    va = [it for it in items if it["split"] == "test"]
    print("train", len(tr), "val", len(va), flush=True)
    dl_tr = DataLoader(PaxZones(tr, True), batch_size=16, shuffle=True, num_workers=10,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(PaxZones(va, False), batch_size=16, shuffle=False, num_workers=8,
                       pin_memory=True)
    dev = "cuda:0"
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1,
                     classes=NCH).to(dev).to(memory_format=torch.channels_last)
    EP = 18
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EP * len(dl_tr),
                                                pct_start=0.1)
    best = -1
    for ep in range(EP):
        model.train(); t0 = time.time(); tot = 0
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                lo = model(x)
            lo = lo.float()
            loss = F.binary_cross_entropy_with_logits(lo, y) + dice_loss(lo, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        model.eval(); inter = np.zeros(NCH); s1 = np.zeros(NCH); s2 = np.zeros(NCH)
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(dev).to(memory_format=torch.channels_last); y = y.to(dev)
                with torch.autocast("cuda", torch.bfloat16):
                    p = (torch.sigmoid(model(x).float()) > 0.5).float()
                inter += (p * y).sum((0, 2, 3)).cpu().numpy()
                s1 += p.sum((0, 2, 3)).cpu().numpy(); s2 += y.sum((0, 2, 3)).cpu().numpy()
        dice = 2 * inter / np.maximum(s1 + s2, 1)
        md = float(dice.mean())
        print(f"ep{ep} loss {tot/len(dl_tr):.4f} dice {md:.4f} "
              f"[{' '.join(f'{d:.3f}' for d in dice)}] {time.time()-t0:.0f}s", flush=True)
        if md > best:
            best = md
            torch.save({"sd": model.state_dict(), "names": idx["names"], "dice": dice.tolist()},
                       os.path.join(EXT, "anatomy_unet.pth"))
    print("best mean dice", best, flush=True)

if __name__ == "__main__":
    main()
