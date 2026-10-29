"""Test-set inference and the clinician-facing deliverables.

The test films are touched here for the first time. Both alert thresholds are the
ones chosen on the validation split; nothing on this page is tuned on test data.
"""
import os, sys, json, argparse, numpy as np, pandas as pd, cv2, torch
sys.path.insert(0, os.getcwd())
import data as D, train as T

EXT = D.EXT
OUT = "projects/20260828_09"
NIH_MASTER = ("/path/to/your/datasets/Chest_Xray/preprocessed_1024/"
              "NIH_ChestX-ray14/final_cxr14_master_list.csv")
SIDE_EN = {0: "right", 1: "left"}
SIDE_ZH = {0: "右", 1: "左"}


def load(ckpt, dev="cuda:0"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    m = T.build_model(cfg)
    m.load_state_dict(ck["sd"])
    return m.eval().to(dev).to(memory_format=torch.channels_last), cfg, ck["met"]


@torch.no_grad()
def score(model, cfg, ds, dev="cuda:0", bs=32, prob_dir=None):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=8, pin_memory=True)
    SC, SG, Y, F, SIDE, DICE = [], [], [], [], [], []
    dn = dd = 0.0
    for bi, (x, m, z, y, idx) in enumerate(dl):
        x = x.to(dev).to(memory_format=torch.channels_last)
        z = z.to(dev); m = m.to(dev)
        sgs, cps, p0 = [], [], None
        for ang, scl in cfg.get("tta", [(0.0, 1.0)]):
            xv = x if (ang == 0 and scl == 1) else T._affine(x, ang, scl, dev)
            zv = z if (ang == 0 and scl == 1) else T._affine(z, ang, scl, dev, "nearest")
            with torch.autocast("cuda", torch.bfloat16):
                out = model(xv)
            lo, cl = T.split_out(out)
            pv = torch.sigmoid(lo.float())[:, :1]
            if p0 is None:
                p0 = pv[:, 0]
            sgs.append(T.zone_score(pv[:, 0], zv, cfg["topk_frac"], cfg["topk_min"]))
            cps.append(torch.sigmoid(cl.float()).view(-1, 1))
        sg = torch.stack(sgs).mean(0)
        cp = torch.stack(cps).mean(0)
        sc = T.combine(sg, cp, cfg.get("score_mode", "seg"))
        b = (p0 > 0.5).float()
        dn += (b * m[:, 0]).sum().item(); dd += (b.sum() + m[:, 0].sum()).item()
        inter = (b * m[:, 0]).sum((1, 2)); den = b.sum((1, 2)) + m[:, 0].sum((1, 2))
        DICE.append(((2 * inter) / den.clamp(min=1e-6)).cpu().numpy())
        SC.append(sc[:, 0].cpu().numpy()); SG.append(sg[:, 0].cpu().numpy())
        Y.append(y[:, 0].numpy()); F.append(idx.numpy())
        SIDE.append((np.arange(len(idx)) + bi * bs) % 2)
        if prob_dir:
            os.makedirs(prob_dir, exist_ok=True)
            pm = (p0.cpu().numpy() * 255).astype(np.uint8)
            for k in range(pm.shape[0]):
                gi = bi * bs + k
                cv2.imwrite(os.path.join(prob_dir, f"{int(idx[k])}_{gi % 2}.png"), pm[k])
    return (np.concatenate(SC), np.concatenate(Y), np.concatenate(F),
            np.concatenate(SIDE), np.concatenate(DICE), dn, dd)


def prevalence_adjust(se, sp, p):
    return se * p / max(se * p + (1 - sp) * (1 - p), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    model, cfg, valmet = load(os.path.join(a.run, "checkpoints", "best.pth"), dev)
    print("selected run:", a.run, "| validation apex_ap", round(valmet["apex_ap"], 4), flush=True)

    df = D.load_index()
    va = df[(df.split == "val") & df.eval_ok]
    te_all = df[df.split == "test"]
    te = te_all[te_all.eval_ok]

    # ---- operating points, both fixed on validation --------------------------------
    sv, yv, fv, _, _, _, _ = score(model, cfg, D.ApexSideDS(va, cfg["res"], False, None, "siim"), dev)
    vm = T.compute_metrics(sv, yv, fv, len(np.unique(fv)))
    keep = yv >= 0
    order = np.argsort(-sv[keep]); ys = yv[keep][order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(ys.sum(), 1)
    ops = {}
    for tgt in [0.50, 0.70]:
        ok = np.where(prec >= tgt)[0]
        ops[tgt] = float(sv[keep][order][ok[int(np.argmax(rec[ok]))]]) if len(ok) else 1.0
    print("validation thresholds:", {k: round(v, 4) for k, v in ops.items()}, flush=True)

    # ---- test set -------------------------------------------------------------------
    prob_dir = os.path.join(EXT, "test_prob_maps")
    st, yt, ft, side, dice, dn, dd = score(
        model, cfg, D.ApexSideDS(te, cfg["res"], False, None, "siim"), dev, prob_dir=prob_dir)
    tm = T.compute_metrics(st, yt, ft, len(np.unique(ft)), dn, dd)
    print("\nTEST (frozen metric, drain-free films):", flush=True)
    for k, v in tm.items():
        if not k.startswith("_"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)

    res = {"selected_run": a.run, "validation": valmet, "test_frozen": tm,
           "val_thresholds": {str(k): v for k, v in ops.items()}}

    ok = yt >= 0
    for tgt, thr in ops.items():
        al = st >= thr
        tp_ = int((al & (yt == 1) & ok).sum()); fp_ = int((al & (yt == 0) & ok).sum())
        fn_ = int((~al & (yt == 1) & ok).sum()); tn_ = int((~al & (yt == 0) & ok).sum())
        se = tp_ / max(tp_ + fn_, 1); sp = tn_ / max(tn_ + fp_, 1)
        nfilms = len(np.unique(ft))
        pos_films = set(ft[(yt == 1) & ok].tolist())
        hit_films = set(ft[(yt == 1) & ok & al].tolist())
        d = dict(threshold=thr, alerts=int(al.sum()),
                 alerts_per_100_films=100 * float(al.sum()) / nfilms,
                 ppv=tp_ / max(tp_ + fp_, 1), sensitivity=se, specificity=sp,
                 film_level_recall=len(hit_films) / max(len(pos_films), 1),
                 tp=tp_, fp=fp_, fn=fn_, tn=tn_,
                 ppv_if_prevalence={f"{p:.0%}": prevalence_adjust(se, sp, p)
                                    for p in [0.05, 0.02, 0.01]})
        res[f"operating_point_ppv{int(tgt*100)}"] = d
        print(f"\noperating point tuned for PPV>={tgt:.0%} on validation -> on test:", flush=True)
        print(f"  alerts {d['alerts']} ({d['alerts_per_100_films']:.1f} per 100 films); "
              f"of those {d['ppv']:.1%} were real", flush=True)
        print(f"  caught {se:.1%} of apical pneumothorax zones; "
              f"{d['film_level_recall']:.1%} of affected films got at least one alert", flush=True)
        print(f"  PPV if true prevalence were "
              + ", ".join(f"{k}: {v:.1%}" for k, v in d["ppv_if_prevalence"].items()), flush=True)

    # ---- does an obvious finding elsewhere silence the apical check? -----------------
    m2 = pd.read_csv(os.path.join(EXT, "siim_nih_match_v2.csv"))
    master = pd.read_csv(NIH_MASTER).set_index("image_id")
    cols = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule",
            "pneumonia", "consolidation", "edema", "emphysema", "fibrosis",
            "pleural_thickening", "hernia"]
    mm = te.merge(m2, on="new_filename", how="left")
    mm = mm[mm.sim > 0.95]
    lab = master.reindex(mm.nih_image_id)[cols]
    other = dict(zip(mm["idx"].to_numpy(), (lab.sum(1).to_numpy() > 0).astype(int)))
    thr = ops[0.50]
    sub = {}
    for tag, want in [("another finding on the film", 1), ("no other finding recorded", 0)]:
        sel = np.array([other.get(int(f), -1) == want for f in ft]) & (yt == 1) & ok
        n = int(sel.sum())
        sub[tag] = dict(n_zones=n, detected=float((st[sel] >= thr).mean()) if n else None)
        print(f"\napical pneumothorax, {tag}: n={n} "
              f"detected {sub[tag]['detected'] if n else 'n/a'}", flush=True)
    res["second_finding_subgroup"] = sub

    # ---- treated (drain) vs untreated ------------------------------------------------
    sa, ya, fa, _, _, _, _ = score(model, cfg, D.ApexSideDS(te_all, cfg["res"], False, None, "siim"), dev)
    info = df.set_index("idx")[["drain", "has_pneumo"]]
    dr = info.loc[fa, "drain"].to_numpy(); hp = info.loc[fa, "has_pneumo"].to_numpy()
    grp = {}
    for tag, sel in [("untreated pneumothorax at the apex", (ya == 1) & (dr == 0)),
                     ("pneumothorax with a drain already in place", (ya == 1) & (dr == 1)),
                     ("normal apex, no device on the film", (ya == 0) & (dr == 0)),
                     ("normal apex, device on the film", (ya == 0) & (dr == 1))]:
        n = int(sel.sum())
        grp[tag] = dict(n=n, alert_rate=float((sa[sel] >= thr).mean()) if n else None)
        print(f"{tag}: n={n} alert rate {grp[tag]['alert_rate']}", flush=True)
    res["device_subgroup"] = grp

    # ---- external: films read as normal in ChestX-ray14 -------------------------------
    nd = pd.read_csv(os.path.join(EXT, "nih_index.csv"))
    nd = nd[nd.role == "ext_normal"]
    sn, _, fn2, _, _, _, _ = score(model, cfg, D.ApexSideDS(nd, cfg["res"], False, None, "nih"), dev)
    ext = dict(n_films=len(nd), n_zones=len(sn),
               alerts_per_100_films=100 * float((sn >= thr).sum()) / len(nd))
    res["external_normal_films"] = ext
    print(f"\nChestX-ray14 films reported as normal (n={len(nd)}, no overlap with SIIM): "
          f"{ext['alerts_per_100_films']:.1f} alerts per 100 films", flush=True)

    # ---- coverage gap: pneumothorax outside the apex ----------------------------------
    ptx = te[te.has_pneumo == 1]
    apex_any = ptx[["right_apex_frac", "left_apex_frac"]].max(1) >= D.POS_FRAC
    base_any = ptx[["right_base_frac", "left_base_frac"]].max(1) >= D.POS_FRAC
    res["coverage"] = dict(test_ptx_films=int(len(ptx)),
                           with_apical_involvement=float(apex_any.mean()),
                           with_costophrenic_involvement=float(base_any.mean()),
                           costophrenic_only=float((base_any & ~apex_any).mean()))
    print("\ncoverage of the version-1 region:", res["coverage"], flush=True)

    # ---- deliverables ----------------------------------------------------------------
    idx2name = df.set_index("idx")["new_filename"].to_dict()
    rows = []
    for s_, y_, f_, sd_, dc_ in zip(st, yt, ft, side, dice):
        alert = bool(s_ >= ops[0.50])
        rows.append(dict(
            sample_id=f"{idx2name[int(f_)]}|{SIDE_EN[int(sd_)]}_apex",
            film=idx2name[int(f_)], patient_id="",
            zone=f"{SIDE_EN[int(sd_)]}_apex",
            label=int(y_),
            prediction="alert" if alert else "silent",
            alert_text_en=(f"take another look at the {SIDE_EN[int(sd_)]} apex - "
                           "possible small pneumothorax") if alert else "",
            alert_text_zh=(f"再看一下{SIDE_ZH[int(sd_)]}肺尖——可能有小气胸") if alert else "",
            confidence=float(s_),
            prob_map=os.path.join(prob_dir, f"{int(f_)}_{int(sd_)}.png"),
            zone_dice=float(dc_)))
    pred = pd.DataFrame(rows)
    pred.to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)

    alerts = pred[pred.prediction == "alert"][["film", "zone", "alert_text_zh", "alert_text_en"]]
    alerts.to_csv(os.path.join(OUT, "test_alerts.csv"), index=False)
    with open(os.path.join(OUT, "test_alerts.txt"), "w") as f:
        for film, g in alerts.groupby("film"):
            f.write(film + "\n")
            for t in g.alert_text_zh:
                f.write("    " + t + "\n")
    silent = int((pred.prediction == "silent").sum())
    print(f"\nwrote {len(pred)} zone rows, {len(alerts)} alerts, {silent} silent zones", flush=True)
    json.dump(res, open(os.path.join(OUT, "test_report.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
