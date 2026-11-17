"""Inference demo: a folder of images in, annotated images out. Works on CPU.

Single model (viva demo):
  python demo.py --ckpt runs/clipart/ours/best.pth --input samples/ --out outputs/
Several models side by side (qualitative figure for the report):
  python demo.py --ckpt runs/clipart/b0/best.pth runs/clipart/b2/best.pth runs/clipart/ours/best.pth \
      --names "Source-only" "DINOv2, no tags" "Ours" --input samples/ --out qualitative/
Add --gt data/json/clipart_test.json to append a ground-truth panel.
"""
import argparse
import glob
import os
import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from tagdaod.data import load_json
from tagdaod.models import load_checkpoint

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40),
           (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128),
           (128, 128, 128), (255, 225, 25)]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def font(size):
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw(img, boxes, labels, scores, classes, title=None):
    img = img.copy()
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(img.size) / 400))
    f = font(max(12, round(max(img.size) / 50)))
    for b, l, s in zip(boxes, labels, scores):
        col = PALETTE[l % len(PALETTE)]
        d.rectangle(b, outline=col, width=lw)
        text = classes[l] if s is None else f"{classes[l]} {s:.2f}"
        tb = d.textbbox((b[0], b[1]), text, font=f)
        y = b[1] - (tb[3] - tb[1]) - 4 if b[1] > (tb[3] - tb[1]) + 4 else b[1] + 1  # inside the box near the top edge
        d.rectangle([b[0], y, b[0] + tb[2] - tb[0] + 6, y + tb[3] - tb[1] + 4], fill=col)
        d.text((b[0] + 3, y + 1), text, fill=(0, 0, 0), font=f)
    if title:  # caption in a white strip above the image, so it never hides boxes or labels
        f2 = font(max(16, round(max(img.size) / 35)))
        tb = ImageDraw.Draw(img).textbbox((0, 0), title, font=f2)
        h = tb[3] - tb[1] + 16
        out = Image.new("RGB", (img.width, img.height + h), (255, 255, 255))
        out.paste(img, (0, h))
        ImageDraw.Draw(out).text(((img.width - (tb[2] - tb[0])) // 2, 6), title, fill=(0, 0, 0), font=f2)
        return out
    return img


@torch.no_grad()
def run(model, img, device, thresh):
    p = model([TF.to_tensor(img).to(device)])[0]
    k = p["scores"] >= thresh
    return p["boxes"][k].cpu().tolist(), (p["labels"][k] - 1).cpu().tolist(), p["scores"][k].cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--input", required=True, help="folder of images (or a single image)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--gt", default=None, help="optional labelled JSON to add a ground-truth panel")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = [load_checkpoint(c, device) for c in a.ckpt]
    names = a.names or [os.path.basename(os.path.dirname(c)) for c in a.ckpt]
    classes = models[0][1]["classes"]
    gt = {}
    if a.gt:
        gt = {os.path.basename(r["file"]): r for r in load_json(a.gt)["images"]}

    files = [a.input] if os.path.isfile(a.input) else sorted(
        f for f in glob.glob(os.path.join(a.input, "*")) if f.lower().endswith(EXTS))
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(files)} images, {len(models)} model(s), device {device}")
    for f in files:
        img = Image.open(f).convert("RGB")
        panels = []
        for (model, _), name in zip(models, names):
            t = time.time()
            boxes, labels, scores = run(model, img, device, a.thresh)
            dt = time.time() - t
            panels.append(draw(img, boxes, labels, scores, classes, name if len(models) > 1 or gt else None))
            summary = ", ".join(f"{classes[l]} {s:.2f}" for l, s in list(zip(labels, scores))[:8]) or "no detections"
            more = f" (+{len(labels) - 8} more)" if len(labels) > 8 else ""
            print(f"{os.path.basename(f)} [{name}] {dt:.2f}s, {len(labels)} boxes: {summary}{more}")
        g = gt.get(os.path.basename(f))
        if g:
            panels.append(draw(img, g["boxes"], g["labels"], [None] * len(g["labels"]), classes, "Ground truth"))
        w, h = panels[0].size
        canvas = Image.new("RGB", (w * len(panels) + 8 * (len(panels) - 1), h), (255, 255, 255))
        for k, p in enumerate(panels):
            canvas.paste(p, (k * (w + 8), 0))
        canvas.save(os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + "_det.jpg"), quality=92)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
