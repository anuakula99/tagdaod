"""Tiny synthetic source/target data for checking the pipeline runs end to end.

Source: solid shapes on light noise. Target: the same shapes drawn "stylised"
(outlined, inverted colours, blurred, noisy). Classes: square, circle, triangle.
  python tools/make_toy_data.py --out data/toy
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

CLASSES = ["square", "circle", "triangle"]


def draw_image(path, stylised, size=256):
    bg = (np.random.rand(size, size, 3) * 40 + 200).astype(np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    boxes, labels = [], []
    for _ in range(random.randint(1, 3)):
        c = random.randrange(3)
        s = random.randint(40, 90)
        x, y = random.randint(0, size - s), random.randint(0, size - s)
        col = tuple(random.randint(0, 150) for _ in range(3))
        kw = dict(outline=(0, 0, 0), width=5, fill=None) if stylised else dict(fill=col)
        if c == 0:
            d.rectangle([x, y, x + s, y + s], **kw)
        elif c == 1:
            d.ellipse([x, y, x + s, y + s], **kw)
        else:
            d.polygon([(x + s / 2, y), (x, y + s), (x + s, y + s)], **({"outline": (0, 0, 0), "fill": None} if stylised else {"fill": col}))
        boxes.append([x, y, x + s, y + s]); labels.append(c)
    if stylised:
        img = ImageOps.invert(img).filter(ImageFilter.GaussianBlur(1.5))
    img.save(path)
    return {"file": os.path.abspath(path), "width": size, "height": size, "boxes": boxes,
            "labels": labels, "difficult": [0] * len(labels), "tags": sorted(set(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    random.seed(0); np.random.seed(0)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    def split(name, n, stylised):
        return [draw_image(os.path.join(a.out, "images", f"{name}_{i:03d}.png"), stylised) for i in range(n)]

    src, tgt, test = split("src", a.n, False), split("tgt", a.n, True), split("test", a.n // 2, True)
    dump = lambda imgs, f: json.dump({"classes": CLASSES, "images": imgs}, open(os.path.join(a.out, f), "w"))
    dump(src, "source_train.json")
    dump(tgt, "target_train_full.json")
    dump([dict(r, boxes=[], labels=[], difficult=[]) for r in tgt], "target_train_tags.json")
    dump(test, "target_test.json")
    print("toy data in", a.out)


if __name__ == "__main__":
    main()
