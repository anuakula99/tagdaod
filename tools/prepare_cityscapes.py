"""Convert Cityscapes / Foggy Cityscapes to the project JSON (8 classes, boxes from
gtFine instance masks, the standard DAOD protocol).

Expected layout
  <cityscapes>/leftImg8bit/{train,val}/<city>/<stem>_leftImg8bit.png
  <cityscapes>/gtFine/{train,val}/<city>/<stem>_gtFine_instanceIds.png
  <foggy>/leftImg8bit_foggy/{train,val}/<city>/<stem>_leftImg8bit_foggy_beta_0.02.png

Examples
  python tools/prepare_cityscapes.py --cityscapes data/cityscapes --split train --domain clear \
      --out data/json/cityscapes_train.json
  python tools/prepare_cityscapes.py --cityscapes data/cityscapes --foggy data/foggy --split train \
      --domain foggy --no-boxes --out data/json/foggy_train_unlabeled.json
  python tools/prepare_cityscapes.py --cityscapes data/cityscapes --foggy data/foggy --split val \
      --domain foggy --out data/json/foggy_val.json
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tagdaod.data import CITYSCAPES_CLASSES  # noqa: E402

# Cityscapes label ids of the 8 instance classes
LABEL_ID = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 31: 5, 32: 6, 33: 7}


def boxes_from_instances(path):
    arr = np.array(Image.open(path))
    boxes, labels = [], []
    for iid in np.unique(arr):
        if iid < 1000 or iid // 1000 not in LABEL_ID:
            continue
        ys, xs = np.nonzero(arr == iid)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        labels.append(LABEL_ID[iid // 1000])
    return boxes, labels, arr.shape[1], arr.shape[0]


def work(job):
    inst, img, no_boxes = job
    boxes, labels, w, h = boxes_from_instances(inst)
    rec = {"file": os.path.abspath(img), "width": w, "height": h, "boxes": boxes, "labels": labels,
           "difficult": [0] * len(labels), "tags": sorted(set(labels))}
    if no_boxes:
        rec.update(boxes=[], labels=[], difficult=[], tags=None)  # unsupervised: no tags either
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityscapes", required=True)
    ap.add_argument("--foggy", default=None)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--domain", choices=["clear", "foggy"], required=True)
    ap.add_argument("--beta", default="0.02")
    ap.add_argument("--no-boxes", action="store_true", help="unlabelled target: drop boxes and tags")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    jobs = []
    for inst in sorted(glob.glob(os.path.join(a.cityscapes, "gtFine", a.split, "*", "*_gtFine_instanceIds.png"))):
        city = os.path.basename(os.path.dirname(inst))
        stem = os.path.basename(inst).replace("_gtFine_instanceIds.png", "")
        if a.domain == "clear":
            img = os.path.join(a.cityscapes, "leftImg8bit", a.split, city, f"{stem}_leftImg8bit.png")
        else:
            img = os.path.join(a.foggy, "leftImg8bit_foggy", a.split, city,
                               f"{stem}_leftImg8bit_foggy_beta_{a.beta}.png")
        if not os.path.exists(img):
            print("missing image, skipped:", img)
            continue
        jobs.append((inst, img, a.no_boxes))
    with Pool(a.workers) as pool:
        images = pool.map(work, jobs, chunksize=8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"classes": CITYSCAPES_CLASSES, "images": images}, open(a.out, "w"))
    print(f"{len(images)} images, {sum(len(r['boxes']) for r in images)} boxes -> {a.out}")


if __name__ == "__main__":
    main()
