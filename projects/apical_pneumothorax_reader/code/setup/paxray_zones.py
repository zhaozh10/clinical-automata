"""Extract the 7 anatomical zones we need from PAXRay++ (frontal films only).

Channels (in order):
  0 right lung (135)      1 left lung (136)
  2 right apical zone(144) 3 left apical zone (148)
  4 right lung base (143)  5 left lung base (147)
  6 heart (121)
Saved as a bit-packed memmap so the whole set fits in ~2 GB.
"""
import os, json, numpy as np
from pycocotools import mask as maskutil

PAX = "/path/to/your/datasets/pax_ray++"
EXT = "/path/to/your/external_data/ptxsecond_v1"
CATS = [135, 136, 144, 148, 143, 147, 121]
NAMES = ["right_lung", "left_lung", "right_apex", "left_apex",
         "right_base", "left_base", "heart"]
S = 512

def collect(split_json, split_name, rows, masks, offset):
    d = json.load(open(os.path.join(PAX, split_json)))
    imgs = {im["id"]: im for im in d["images"] if "frontal" in im["file_name"]}
    per = {i: {} for i in imgs}
    want = set(CATS)
    for a in d["annotations"]:
        if a["image_id"] in per and a["category_id"] in want:
            per[a["image_id"]][a["category_id"]] = a["segmentation"]
    del d
    for iid, im in sorted(imgs.items()):
        buf = np.zeros((len(CATS), S, S), np.uint8)
        for ci, c in enumerate(CATS):
            seg = per[iid].get(c)
            if seg is None:
                continue
            rle = dict(seg)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode()
            buf[ci] = maskutil.decode(rle)
        masks.append(np.packbits(buf.astype(bool)))
        rows.append((im["file_name"], split_name, offset + len(rows)))
    return rows, masks

if __name__ == "__main__":
    os.makedirs(EXT, exist_ok=True)
    rows, masks = [], []
    for j, name in [("paxray_train_val.json", "trainval"), ("paxray_test.json", "test")]:
        print("loading", j, flush=True)
        rows, masks = collect(j, name, rows, masks, 0)
        print(name, "cumulative frontal", len(rows), flush=True)
    arr = np.stack(masks)
    np.save(os.path.join(EXT, "paxray_zones.npy"), arr)
    with open(os.path.join(EXT, "paxray_zones_index.json"), "w") as f:
        json.dump({"names": NAMES, "cats": CATS, "shape": [len(CATS), S, S],
                   "items": [{"file_name": r[0], "split": r[1], "idx": i}
                             for i, r in enumerate(rows)]}, f)
    print("saved", arr.shape, arr.nbytes / 1e9, "GB", flush=True)
