"""Second-reader apical check for frontal chest radiographs.

Reads a folder of frontal chest films and prints one short line per apex that is
worth re-inspecting. Films it has nothing to say about produce no output.

    uv run python projects/20260828_09/predict.py --images <folder> --out <folder>

Two stages, both frozen:
  1. an anatomy locator marks the left and right apical zone on the film;
  2. each apex is cut out, magnified, and judged on its own.
Because each apex is judged on its own, an obvious abnormality elsewhere on the
film cannot stop the apical check from running.
"""
import os, sys, glob, argparse, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import segmentation_models_pytorch as smp
import data as D, train as T

EXT = D.EXT
ANATOMY = os.path.join(EXT, "anatomy_unet.pth")
SIDE_EN = {0: "right", 1: "left"}
SIDE_ZH = {0: "右", 1: "左"}
# both thresholds were fixed on the validation split, never on test data
THRESHOLDS = {"standard": 0.1226, "quiet": 0.3770}


def load_models(ckpt, dev):
    a = torch.load(ANATOMY, map_location="cpu", weights_only=False)
    loc = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=7)
    loc.load_state_dict(a["sd"]); loc.eval().to(dev)
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(c["cfg"])
    net = T.build_model(cfg); net.load_state_dict(c["sd"]); net.eval().to(dev)
    return loc, net, cfg


@torch.no_grad()
def analyse(path, loc, net, cfg, dev, thr):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    H = raw.shape[0]
    small = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).float().div_(255.).sub_(0.5).div_(0.25)[None, None].to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        z = torch.sigmoid(loc(x).float())[0] > 0.5
    z = z.cpu().numpy()
    out = []
    for s in range(2):
        y0, x0, sd = D.side_box(z[2 + s].astype(np.float32), 512, 0.35)
        sf = raw.shape[0] / 512.0
        c = raw[int(y0 * sf):int((y0 + sd) * sf), int(x0 * sf):int((x0 + sd) * sf)]
        if c.size == 0:
            continue
        c = cv2.resize(c, (512, 512), interpolation=cv2.INTER_LINEAR)
        zc = cv2.resize(z[2 + s].astype(np.float32)[y0:y0 + sd, x0:x0 + sd], (512, 512),
                        interpolation=cv2.INTER_NEAREST)
        if s == 1:
            c = np.ascontiguousarray(c[:, ::-1]); zc = np.ascontiguousarray(zc[:, ::-1])
        xi = torch.from_numpy(c).float().div_(255.).sub_(0.5).div_(0.25)[None, None].to(dev)
        zi = torch.from_numpy(zc)[None, None].to(dev)
        cps, maps = [], []
        for ang, scl in cfg.get("tta", [(0.0, 1.0)]):
            xv = xi if (ang == 0 and scl == 1) else T._affine(xi, ang, scl, dev)
            with torch.autocast("cuda", torch.bfloat16):
                lo, cl = T.split_out(net(xv))
            cps.append(float(torch.sigmoid(cl.float()).view(-1)[0]))
            if ang == 0 and scl == 1:
                maps.append(torch.sigmoid(lo.float())[0, 0].cpu().numpy())
        p = float(np.mean(cps))
        out.append(dict(side=s, score=p, alert=p >= thr,
                        crop=c, prob=maps[0] if maps else None,
                        box=(y0, x0, sd)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of frontal chest films")
    ap.add_argument("--out", default="apex_check_out")
    ap.add_argument("--checkpoint",
                    default="projects/20260828_09/run29/checkpoints/best.pth")
    ap.add_argument("--setting", choices=list(THRESHOLDS), default="standard")
    ap.add_argument("--save-overlays", action="store_true")
    a = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    loc, net, cfg = load_models(a.checkpoint, dev)
    thr = THRESHOLDS[a.setting]
    os.makedirs(a.out, exist_ok=True)
    files = sorted(sum([glob.glob(os.path.join(a.images, e))
                        for e in ("*.png", "*.jpg", "*.jpeg", "*.PNG")], []))
    lines, rows = [], []
    for f in files:
        r = analyse(f, loc, net, cfg, dev, thr)
        if r is None:
            print(f"could not read {f}", file=sys.stderr); continue
        name = os.path.basename(f)
        for zz in r:
            rows.append(f"{name},{SIDE_EN[zz['side']]}_apex,{zz['score']:.4f},"
                        f"{'alert' if zz['alert'] else 'silent'}")
            if zz["alert"]:
                lines.append(f"{name}\t再看一下{SIDE_ZH[zz['side']]}肺尖——可能有小气胸")
                if a.save_overlays and zz["prob"] is not None:
                    ov = cv2.cvtColor(zz["crop"], cv2.COLOR_GRAY2BGR)
                    m = zz["prob"] > 0.5
                    ov[m] = (0.45 * ov[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
                    cv2.imwrite(os.path.join(a.out,
                                f"{os.path.splitext(name)[0]}_{SIDE_EN[zz['side']]}.png"), ov)
    with open(os.path.join(a.out, "alerts.txt"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    with open(os.path.join(a.out, "zone_scores.csv"), "w") as fh:
        fh.write("film,zone,score,decision\n" + "\n".join(rows) + "\n")
    print(f"{len(files)} films read, {len(lines)} alerts "
          f"({a.setting} setting, threshold {thr})")
    for l in lines[:20]:
        print("  " + l.replace("\t", "  "))


if __name__ == "__main__":
    main()
