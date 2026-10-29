"""Post-hoc analysis for the apical second reader.

Nothing here feeds back into training. It produces the clinician-facing numbers:
how clean the alerts are, how many real apical pneumothoraces are caught, how
often the tool speaks up on films read as normal, and whether an obvious finding
elsewhere on the film suppresses the apical check.
"""
import os, sys, json, argparse, numpy as np, pandas as pd, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import data as D
import train as T

EXT = D.EXT
NIH_MASTER = ("/path/to/your/datasets/Chest_Xray/preprocessed_512/"
              "NIH_ChestX-ray14/final_cxr14_master_list.csv")
SIDE = {0: "right", 1: "left"}


def load_model(ckpt, dev="cuda:0"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = T.build_model(cfg)
    model.load_state_dict(ck["sd"])
    return model.eval().to(dev).to(memory_format=torch.channels_last), cfg


@torch.no_grad()
def score_siim(model, cfg, df, dev="cuda:0", bs=16, workers=6, save_prob_dir=None):
    ds = D.SiimZoneDS(df, cfg["res"], train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True)
    S, Y, FID, DICE = [], [], [], []
    for x, m, z, y, idx in dl:
        x = x.to(dev).to(memory_format=torch.channels_last)
        z = z.to(dev); m = m.to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            lo = model(x)
        prob = torch.sigmoid(lo.float())[:, 0]
        sc = T.zone_score(prob, z[:, :2], cfg["topk_frac"], cfg["topk_min"])
        b = (prob > 0.5).float()
        inter = (b * m[:, 0]).sum((1, 2)); den = b.sum((1, 2)) + m[:, 0].sum((1, 2))
        DICE.append(((2 * inter + 1e-6) / (den + 1e-6)).cpu().numpy())
        S.append(sc.cpu().numpy()); Y.append(y.numpy()); FID.append(idx.numpy())
        if save_prob_dir:
            os.makedirs(save_prob_dir, exist_ok=True)
            p = (prob.cpu().numpy() * 255).astype(np.uint8)
            for k, i in enumerate(idx.numpy()):
                cv2.imwrite(os.path.join(save_prob_dir, f"{int(i)}.png"), p[k])
    return (np.concatenate(S), np.concatenate(Y), np.concatenate(FID), np.concatenate(DICE))


class NihDS(torch.utils.data.Dataset):
    def __init__(self, df, res):
        self.df = df.reset_index(drop=True); self.res = res; self.img = self.zon = None
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        if self.img is None:
            self.img = np.load(os.path.join(EXT, "nih_img_512.npy"), mmap_mode="r")
            self.zon = np.load(os.path.join(EXT, "nih_zones_512.npy"), mmap_mode="r")
        k = int(self.df.iloc[i]["idx"]); R = self.res
        im = np.asarray(self.img[k])
        if R != 512:
            im = cv2.resize(im, (R, R), interpolation=cv2.INTER_AREA)
        z = D.zone_masks(self.zon, k, out=R)
        im = (im.astype(np.float32) / 255.0 - 0.5) / 0.25
        return torch.from_numpy(im[None]), torch.from_numpy(z), torch.tensor(k)


@torch.no_grad()
def score_nih(model, cfg, df, dev="cuda:0", bs=16, workers=6):
    dl = torch.utils.data.DataLoader(NihDS(df, cfg["res"]), batch_size=bs,
                                     num_workers=workers, pin_memory=True)
    S, FID = [], []
    for x, z, k in dl:
        x = x.to(dev).to(memory_format=torch.channels_last); z = z.to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            lo = model(x)
        prob = torch.sigmoid(lo.float())[:, 0]
        S.append(T.zone_score(prob, z[:, :2], cfg["topk_frac"], cfg["topk_min"]).cpu().numpy())
        FID.append(k.numpy())
    return np.concatenate(S), np.concatenate(FID)


def alert_text(side_idx, finding="possible small pneumothorax"):
    return f"take another look at the {SIDE[side_idx]} apex - {finding}"


def cofinding_table(df, sim_t=0.95):
    """Attach ChestX-ray14 whole-film labels to SIIM films via the perceptual match,
    purely to split the films into 'another finding present' vs 'not'. These are
    image-level labels only and are used for stratification, never as region truth."""
    m2 = pd.read_csv(os.path.join(EXT, "siim_nih_match_v2.csv"))
    master = pd.read_csv(NIH_MASTER).set_index("image_id")
    d = df.merge(m2, on="new_filename", how="left")
    d = d[d.sim > sim_t].copy()
    cols = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass",
            "nodule", "pneumonia", "consolidation", "edema", "emphysema",
            "fibrosis", "pleural_thickening", "hernia"]
    lab = master.loc[d.nih_image_id, cols].reset_index(drop=True)
    d = d.reset_index(drop=True)
    d["other_finding"] = (lab.sum(1) > 0).astype(int)
    return d
