"""Convert VOC-format data (PASCAL VOC 2007/2012, Clipart1k) to the project JSON.

Examples
  # source: VOC 07+12 trainval
  python tools/prepare_voc.py --dirs data/VOCdevkit/VOC2007 data/VOCdevkit/VOC2012 \
      --split trainval --out data/json/voc_trainval.json
  # target train, WEAK labels only (tags, no boxes)
  python tools/prepare_voc.py --dirs data/clipart --split train --tags-only --out data/json/clipart_train_tags.json
  # target train WITH boxes: only for pseudo-label analysis and the oracle, never for our method
  python tools/prepare_voc.py --dirs data/clipart --split train --out data/json/clipart_train_full.json
  # target test
  python tools/prepare_voc.py --dirs data/clipart --split test --out data/json/clipart_test.json
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tagdaod.data import VOC_CLASSES  # noqa: E402


def parse(xml_path, img_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.find("width").text)), int(float(size.find("height").text))
    boxes, labels, difficult = [], [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in VOC_CLASSES:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([x1 - 1, y1 - 1, x2, y2])  # VOC is 1-based
        labels.append(VOC_CLASSES.index(name))
        d = obj.find("difficult")
        difficult.append(int(d.text) if d is not None else 0)
    return {"file": os.path.abspath(img_path), "width": w, "height": h, "boxes": boxes,
            "labels": labels, "difficult": difficult, "tags": sorted(set(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="dataset roots containing Annotations/, JPEGImages/, ImageSets/")
    ap.add_argument("--split", required=True, help="name of ImageSets/Main/<split>.txt")
    ap.add_argument("--tags-only", action="store_true", help="drop boxes, keep image-level tags (weak target)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    images = []
    for d in a.dirs:
        ids = [l.strip() for l in open(os.path.join(d, "ImageSets", "Main", f"{a.split}.txt")) if l.strip()]
        for i in ids:
            rec = parse(os.path.join(d, "Annotations", f"{i}.xml"), os.path.join(d, "JPEGImages", f"{i}.jpg"))
            if a.tags_only:
                rec.update(boxes=[], labels=[], difficult=[])
            images.append(rec)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"classes": VOC_CLASSES, "images": images}, open(a.out, "w"))
    nbox = sum(len(r["boxes"]) for r in images)
    print(f"{len(images)} images, {nbox} boxes -> {a.out}")


if __name__ == "__main__":
    main()
