"""Apical-zone pneumothorax second reader.

Dense pneumothorax segmentation, then evidence is pooled *inside each anatomical
apical zone independently*. Because scoring is per zone and purely local, a large
obvious abnormality elsewhere on the film cannot switch the apical check off --
that is the behavioural requirement from the clinician, not an implementation detail.
"""
import os, sys, time, json, math, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy
import segmentation_models_pytorch as smp
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from prepare import TIME_BUDGET_MINUTES

# ================================ run config =======================================
CFG = dict(
    run_dir=os.environ.get("RUN_DIR", "projects/20260828_09/run29"),
    desc="FINAL: per-side apical crop, U-Net + per-apex head, slow weight averaging, 5-view test-time averaging",
    encoder="resnet34", res=512, bs=16, lr=3e-4, wd=1e-4, gpus=1,
    cls_w=1.0, cls_dropout=0.3, score_mode="cls", hardneg_w=3.0, ema=0.9995,
    tta=[(0.0, 1.0), (0.0, 0.92), (0.0, 1.08), (5.0, 1.0), (-5.0, 1.0)],
    crop="side", src_res=1024, margin=0.35, gpu_aug=True,
    workers=12, topk_frac=0.02, topk_min=20, seed=0,
    bce_w=0.5, dice_w=0.5, pos_weight=2.0,
    sched="cosine", warmup=0.05, lr_min_frac=0.05, dice_pos_only=True,
    aug=dict(affine=True, rot=10, scale=(0.90, 1.12), shift=0.05,
             gamma=(0.75, 1.35), bright=0.10, contrast=0.15, p_blur=0.15, p_noise=0.15),
)
# The clinician authorised extending the per-run budget from TIME_BUDGET_MINUTES (30)
# to 40 minutes for this project. prepare.py itself is left untouched.
BUDGET_MINUTES = float(os.environ.get("BUDGET_MIN", 40))
BUDGET = BUDGET_MINUTES * 60
MAX_EVALS = 12
PRIMARY = "apex_ap"
DIRECTION = "higher"
# ===================================================================================


# ---------------- FROZEN EVALUATION (do not change without the clinician) ----------
def compute_metrics(scores, labels, film_id, n_films, dice_num=0.0, dice_den=0.0):
    """scores/labels: one entry per apical zone. labels: 1 positive, 0 negative,
    -1 ignored (a sub-1% sliver of pneumothorax in the zone).

    apex_ap -- area under the precision/recall curve over apical zones -- is the
    primary metric: it is dominated by how clean the alerts are, which is the
    clinician's stated priority.
    """
    scores = np.asarray(scores, np.float64); labels = np.asarray(labels, np.int64)
    keep = labels >= 0
    s, y = scores[keep], labels[keep]
    fid = np.asarray(film_id)[keep]
    out = dict(apex_ap=float("nan"), apex_auc=float("nan"), recall_at_ppv50=0.0,
               ppv_at_op=0.0, thr_at_op=1.0, alerts_per_100_films=0.0,
               film_recall=0.0, fp_zones_per_100_neg_films=0.0,
               recall_at_ppv70=0.0, seg_dice=0.0, n_pos_zones=int((y == 1).sum()))
    if dice_den > 0:
        out["seg_dice"] = float(2 * dice_num / dice_den)
    if y.sum() == 0 or y.sum() == len(y):
        return out
    out["apex_ap"] = float(average_precision_score(y, s))
    out["apex_auc"] = float(roc_auc_score(y, s))

    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(ys.sum(), 1)
    for tgt, key in [(0.50, "recall_at_ppv50"), (0.70, "recall_at_ppv70")]:
        ok = np.where(prec >= tgt)[0]
        out[key] = float(rec[ok].max()) if len(ok) else 0.0
    # operating point the tool would actually ship at: most recall while >=50% of
    # alerts are genuine
    ok = np.where(prec >= 0.50)[0]
    if len(ok):
        i = ok[int(np.argmax(rec[ok]))]
        thr = float(s[order][i])
        out["thr_at_op"] = thr
        out["ppv_at_op"] = float(prec[i])
        alert = s >= thr
        out["alerts_per_100_films"] = float(100.0 * alert.sum() / max(n_films, 1))
        pos_films = set(fid[y == 1].tolist())
        hit = set(fid[(y == 1) & alert].tolist())
        out["film_recall"] = float(len(hit) / max(len(pos_films), 1))
        neg_film_ids = set(fid.tolist()) - pos_films
        fp_zone = int(((y == 0) & alert & np.isin(fid, list(neg_film_ids))).sum())
        out["fp_zones_per_100_neg_films"] = float(100.0 * fp_zone / max(len(neg_film_ids), 1))
    return out
# -----------------------------------------------------------------------------------


def zone_score(prob, zmask, topk_frac, topk_min):
    """Pool the pneumothorax probability inside one anatomical zone.
    Mean of the top-k pixels: a small but confident collection scores high, while a
    faint wash over the whole zone (the signature of normal overlapping structure)
    does not. prob (B,H,W), zmask (B,K,H,W) -> (B,K)."""
    B, K, H, W = zmask.shape
    p = prob.reshape(B, 1, -1).expand(B, K, -1)
    z = zmask.reshape(B, K, -1)
    area = z.sum(-1)
    masked = torch.where(z > 0.5, p, torch.full_like(p, -1.0))
    k = torch.clamp((area * topk_frac).long(), min=topk_min)
    kmax = int(k.max().clamp(min=1).item())
    v, _ = torch.topk(masked, min(kmax, masked.shape[-1]), dim=-1)
    idx = torch.arange(v.shape[-1], device=v.device).view(1, 1, -1)
    m = (idx < k.unsqueeze(-1)) & (v >= 0)
    return (v * m).sum(-1) / m.sum(-1).clamp(min=1)


def gpu_augment(trip, aug, res):
    """Batched augmentation on the GPU. trip: (B,3,H,W) uint8 = film / pneumothorax
    mask / apical-zone mask, warped together so they stay aligned."""
    B, _, H, W = trip.shape
    t = trip.float().div_(255.)
    x, m, z = t[:, :1], t[:, 1:2], t[:, 2:3]
    dev = t.device
    a = aug
    if a.get("affine", True):
        rot = math.radians(a.get("rot", 10))
        ang = (torch.rand(B, device=dev) * 2 - 1) * rot
        s0, s1 = a.get("scale", (0.90, 1.12))
        sc = torch.rand(B, device=dev) * (s1 - s0) + s0
        sh = a.get("shift", 0.05)
        tx = (torch.rand(B, device=dev) * 2 - 1) * sh * 2
        ty = (torch.rand(B, device=dev) * 2 - 1) * sh * 2
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, (B, 1, res, res), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
        m = F.grid_sample(m, grid, mode="nearest", align_corners=False)
        z = F.grid_sample(z, grid, mode="nearest", align_corners=False)
    elif res != H:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        m = F.interpolate(m, size=(res, res), mode="nearest")
        z = F.interpolate(z, size=(res, res), mode="nearest")
    g0, g1 = a.get("gamma", (0.75, 1.35))
    gam = (torch.rand(B, 1, 1, 1, device=dev) * (g1 - g0) + g0)
    x = x.clamp(0, 1).pow(gam)
    c = a.get("contrast", 0.15); b = a.get("bright", 0.10)
    x = x * (1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * c) \
          + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * b
    if a.get("p_noise", 0.15) > 0:
        sel = (torch.rand(B, 1, 1, 1, device=dev) < a["p_noise"]).float()
        x = x + sel * torch.randn_like(x) * 0.02
    x = x.clamp(0, 1)
    x = (x - 0.5) / 0.25
    return x.contiguous(), (m > 0.5).float(), (z > 0.5).float()


def dice_loss(logit, tgt, eps=1.0, pos_only=False):
    """On a film with no pneumothorax the Dice term only says "predict nothing",
    and since ~78% of films are negative that term alone drives the network to an
    all-background solution. Restricting Dice to films that actually contain a
    pneumothorax keeps the shape signal without that pull."""
    p = torch.sigmoid(logit)
    num = 2 * (p * tgt).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + eps
    d = 1 - num / den
    if pos_only:
        w = (tgt.sum((1, 2, 3)) > 0).float()
        return (d * w).sum() / w.sum().clamp(min=1)
    return d.mean()


def build_model(cfg):
    """U-Net for the pneumothorax map, optionally with a second head that judges the
    whole crop. Pooling a probability map answers "is there a bright patch here";
    the extra head can answer "does this apex look like a pneumothorax", which is
    the question the alert actually asks."""
    aux = (dict(classes=1, dropout=cfg.get("cls_dropout", 0.3))
           if cfg.get("cls_w", 0.0) > 0 else None)
    return smp.Unet(cfg["encoder"], encoder_weights="imagenet", in_channels=1,
                    classes=1, aux_params=aux)


def split_out(out):
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out, None


def combine(seg_score, cls_prob, mode):
    if cls_prob is None or mode == "seg":
        return seg_score
    if mode == "cls":
        return cls_prob
    return 0.5 * (seg_score + cls_prob)


def _affine(x, ang_deg, scale, dev, mode="bilinear"):
    B = x.shape[0]
    ang = torch.full((B,), math.radians(ang_deg), device=dev)
    sc = torch.full((B,), float(scale), device=dev)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = cos; th[:, 0, 1] = -sin
    th[:, 1, 0] = sin; th[:, 1, 1] = cos
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode=mode, align_corners=False)


@torch.no_grad()
def run_eval(model, dl, dev, cfg):
    model.eval()
    S, Y, FID, SG, CP = [], [], [], [], []
    dn = dd = 0.0
    for x, m, z, y, idx in dl:
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        m = m.to(dev, non_blocking=True); z = z.to(dev, non_blocking=True)
        K = y.shape[1]
        views = cfg.get("tta", [(0.0, 1.0)])
        sgs, cps, prob0 = [], [], None
        for ang, scl in views:
            xv = x if (ang == 0.0 and scl == 1.0) else _affine(x, ang, scl, dev)
            zv = z if (ang == 0.0 and scl == 1.0) else _affine(z, ang, scl, dev, "nearest")
            with torch.autocast("cuda", torch.bfloat16):
                out = model(xv)
            lo, cl = split_out(out)
            pv = torch.sigmoid(lo.float())[:, 0]
            if prob0 is None:
                prob0 = pv
            sgs.append(zone_score(pv, zv[:, :K], cfg["topk_frac"], cfg["topk_min"]))
            if cl is not None:
                cps.append(torch.sigmoid(cl.float()).view(-1, 1).expand_as(sgs[-1]))
        sg = torch.stack(sgs).mean(0)
        cp = torch.stack(cps).mean(0) if cps else None
        prob = prob0
        b = (prob > 0.5).float()
        dn += (b * m[:, 0]).sum().item(); dd += (b.sum() + m[:, 0].sum()).item()
        S.append(combine(sg, cp, cfg.get("score_mode", "seg")).cpu().numpy())
        SG.append(sg.cpu().numpy())
        CP.append(cp.cpu().numpy() if cp is not None else sg.cpu().numpy())
        Y.append(y.numpy())
        FID.append(np.repeat(idx.numpy(), K).reshape(-1, K))
    S = np.concatenate(S).ravel(); Y = np.concatenate(Y).ravel()
    FID = np.concatenate(FID).ravel()
    met = compute_metrics(S, Y, FID, len(np.unique(FID)), dn, dd)
    met["_ap_seg"] = compute_metrics(np.concatenate(SG).ravel(), Y, FID, 1)["apex_ap"]
    met["_ap_cls"] = compute_metrics(np.concatenate(CP).ravel(), Y, FID, 1)["apex_ap"]
    return met


def worker(rank, world):
    cfg = CFG
    ddp = world > 1
    if ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.manual_seed(cfg["seed"] + rank); np.random.seed(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    main_proc = rank == 0
    if main_proc:
        os.makedirs(os.path.join(cfg["run_dir"], "checkpoints"), exist_ok=True)
    t_start = time.time()

    df = D.load_index()
    tr = df[df.split == "train"]
    va = df[(df.split == "val") & df.eval_ok]
    crop = cfg.get("crop")
    gaug = cfg.get("gpu_aug", False)
    if crop == "side":
        mk = lambda d, t: D.ApexSideDS(d, cfg["res"], t, cfg["aug"], "siim",
                                       cfg.get("src_res", 1024), cfg.get("margin", 0.35),
                                       gpu_aug=gaug and t)
    else:
        mk = lambda d, t: D.SiimZoneDS(d, cfg["res"], t, cfg["aug"], crop=crop)
    ds_tr = mk(tr, True)
    # Error analysis on run11 showed false alerts are ~4x more common on normal apices
    # of films that carry a chest drain or other device. Showing those negatives more
    # often is a direct attack on the alert-fatigue failure mode the clinician named.
    hw = cfg.get("hardneg_w", 1.0)
    samp_w = None
    if hw > 1.0 and crop == "side":
        lab = np.stack([tr["apex_r"].to_numpy(), tr["apex_l"].to_numpy()], 1).reshape(-1)
        dev_film = np.repeat(tr["drain"].to_numpy(), 2)
        samp_w = np.ones(len(lab), np.float64)
        samp_w[(lab == 0) & (dev_film == 1)] = hw
        if rank == 0:
            print(f"up-weighted device-bearing normal apices: "
                  f"{int(((lab == 0) & (dev_film == 1)).sum())} of {len(lab)}", flush=True)
    nr = cfg.get("nih_ratio", 0.0)
    if nr > 0:
        nd = pd.read_csv(os.path.join(D.EXT, "nih_index.csv"))
        nd = nd[nd.role == "train_neg"].sample(int(len(tr) * nr), random_state=cfg["seed"])
        if crop == "side":
            ds_nih = D.ApexSideDS(nd, cfg["res"], True, cfg["aug"], "nih")
        else:
            ds_nih = D.NihNegDS(cfg["res"], True, cfg["aug"], "train_neg",
                                n=int(len(tr) * nr), seed=cfg["seed"], crop=crop)
        ds_tr = torch.utils.data.ConcatDataset([ds_tr, ds_nih])
        if rank == 0:
            print(f"extra ChestX-ray14 negative films: {len(nd)}", flush=True)
    w = cfg["workers"] // world if ddp else cfg["workers"]
    samp = DistributedSampler(ds_tr, world, rank, shuffle=True, drop_last=True) if ddp else None
    if samp is None and samp_w is not None:
        samp = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(samp_w), num_samples=len(ds_tr), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], shuffle=samp is None, sampler=samp,
                       num_workers=w, pin_memory=True, drop_last=True,
                       persistent_workers=w > 0)
    dl_va = DataLoader(mk(va, False), batch_size=max(8, cfg["bs"]),
                       shuffle=False, num_workers=max(2, w), pin_memory=True)
    if main_proc:
        print(f"train films {len(tr)}  val films {len(va)}  "
              f"val apex zones + {int((va.apex_r==1).sum()+(va.apex_l==1).sum())}  "
              f"world {world}", flush=True)

    model = build_model(cfg).to(dev).to(memory_format=torch.channels_last)
    net = DDP(model, device_ids=[rank]) if ddp else model
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    def set_lr(frac):
        wu = cfg.get("warmup", 0.0)
        if frac < wu:
            f = frac / max(wu, 1e-8)
        else:
            t = (frac - wu) / max(1 - wu, 1e-8)
            lo = cfg.get("lr_min_frac", 0.05)
            f = lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * f
    pw = torch.tensor(cfg["pos_weight"], device=dev)

    # Validation swings by several points from one pass to the next, so the single
    # best pass is partly luck. An exponential moving average of the weights follows
    # the trend instead of the wobble.
    ema_decay = cfg.get("ema", 0.0)
    ema = copy.deepcopy(model).eval() if ema_decay > 0 else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)

    best = -1e9 if DIRECTION == "higher" else 1e9
    best_met, hist = None, []
    train_s, last_eval, ep = 0.0, -1e9, 0
    stop = torch.zeros(1, device=dev)
    while stop.item() < 1:
        if samp is not None and hasattr(samp, "set_epoch"):
            samp.set_epoch(ep)
        net.train(); t0 = time.time(); tot = 0.0; nb = 0
        for batch in dl_tr:
            if ddp:
                stop[0] = 1.0 if (main_proc and train_s + time.time() - t0 > BUDGET) else 0.0
                dist.broadcast(stop, 0)
                if stop.item() >= 1:
                    break
            elif train_s + (time.time() - t0) > BUDGET:
                stop[0] = 1.0
                break
            set_lr(min(1.0, (train_s + time.time() - t0) / BUDGET))
            if gaug:
                trip = batch[0].to(dev, non_blocking=True)
                x, m, _ = gpu_augment(trip, cfg["aug"], cfg["res"])
                x = x.to(memory_format=torch.channels_last)
            else:
                x, m = batch[0], batch[1]
                x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
                m = m.to(dev, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                out = net(x)
            lo, cl = split_out(out)
            lo = lo.float()
            loss = cfg["bce_w"] * F.binary_cross_entropy_with_logits(lo, m, pos_weight=pw) \
                 + cfg["dice_w"] * dice_loss(lo, m, pos_only=cfg.get("dice_pos_only", False))
            if cl is not None:
                yb = (batch[1] if gaug else batch[3]).to(dev).float().view(-1)
                valid = yb >= 0                      # sub-1% slivers are not supervised
                if valid.any():
                    loss = loss + cfg["cls_w"] * F.binary_cross_entropy_with_logits(
                        cl.float().view(-1)[valid], yb[valid])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for a, b in zip(ema.state_dict().values(), model.state_dict().values()):
                        if a.dtype.is_floating_point:
                            a.mul_(ema_decay).add_(b.detach(), alpha=1 - ema_decay)
                        else:
                            a.copy_(b)
            tot += loss.item(); nb += 1
        train_s += time.time() - t0
        ep += 1
        # the eval decision must be identical on every rank or the barrier deadlocks
        ev = torch.zeros(1, device=dev)
        if main_proc:
            ev[0] = 1.0 if ((train_s - last_eval >= BUDGET / MAX_EVALS) or stop.item() >= 1) else 0.0
        if ddp:
            dist.broadcast(ev, 0)
        do_eval = ev.item() >= 1
        if not do_eval:
            if main_proc:
                print(f"ep{ep} loss {tot/max(nb,1):.4f} train_s {train_s:.0f}", flush=True)
            continue
        last_eval = train_s
        if main_proc:
            scored = ema if ema is not None else model
            met = run_eval(scored, dl_va, dev, cfg)
            hist.append(dict(ep=ep, train_s=train_s, **met))
            print(f"ep{ep} loss {tot/max(nb,1):.4f} train_s {train_s:.0f} "
                  f"{PRIMARY} {met[PRIMARY]:.4f} auc {met['apex_auc']:.4f} "
                  f"r@ppv50 {met['recall_at_ppv50']:.3f} alerts/100 "
                  f"{met['alerts_per_100_films']:.1f} dice {met['seg_dice']:.3f} "
                  f"[seg {met['_ap_seg']:.3f} cls {met['_ap_cls']:.3f}]", flush=True)
            torch.save({"sd": scored.state_dict(), "cfg": cfg, "met": met},
                       os.path.join(cfg["run_dir"], "checkpoints", "last.pth"))
            v = met[PRIMARY]
            if v == v and ((v > best) if DIRECTION == "higher" else (v < best)):
                best = v; best_met = met
                torch.save({"sd": scored.state_dict(), "cfg": cfg, "met": met},
                           os.path.join(cfg["run_dir"], "checkpoints", "best.pth"))
        if ddp:
            dist.barrier()

    if main_proc:
        if best_met is None:
            best_met = run_eval(ema if ema is not None else model, dl_va, dev, cfg)
            best = best_met[PRIMARY]
        peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 2
        json.dump({"cfg": cfg, "best": best_met, "hist": hist},
                  open(os.path.join(cfg["run_dir"], "metrics.json"), "w"), indent=1)
        order = ["apex_auc", "recall_at_ppv50", "recall_at_ppv70", "ppv_at_op",
                 "alerts_per_100_films", "film_recall", "fp_zones_per_100_neg_films",
                 "seg_dice", "thr_at_op", "n_pos_zones"]
        print("\n---")
        print(f"{PRIMARY}:      {best:.4f}\n")
        print("metrics_summary:")
        for k in order:
            print(f"{k}:\t{best_met[k]:.4f}")
        print(f"\ntraining_seconds: {train_s:.1f}")
        print(f"total_seconds:    {time.time()-t_start:.1f}")
        print(f"peak_vram_mb:     {peak:.1f}")
    if ddp:
        dist.barrier(); dist.destroy_process_group()


def main():
    world = CFG.get("gpus", 1)
    if world > 1:
        mp.spawn(worker, args=(world,), nprocs=world, join=True)
    else:
        worker(0, 1)


if __name__ == "__main__":
    main()
