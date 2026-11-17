"""Crop the same region out of every panel of a demo.py side-by-side image (for report figures).

  python tools/crop_panels.py --img qualitative_foggy/x_det.jpg --panels 4 --box 0.35 0.25 1.0 0.65 --out fig/x_crop.jpg

--box is left top right bottom as fractions of one panel (0..1). The panel title strip is kept.
"""
import argparse

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--img", required=True)
ap.add_argument("--panels", type=int, default=4)
ap.add_argument("--gap", type=int, default=8, help="gap between panels in pixels (demo.py uses 8)")
ap.add_argument("--box", type=float, nargs=4, required=True, metavar=("L", "T", "R", "B"))
ap.add_argument("--title-h", type=int, default=None, help="height of the title strip; auto-detected if omitted")
ap.add_argument("--out", required=True)
a = ap.parse_args()

im = Image.open(a.img).convert("RGB")
pw = (im.width - a.gap * (a.panels - 1)) // a.panels
# title strip (demo.py draws it white with centred text): rows that are mostly white
th = a.title_h
if th is None:
    th = 0
    for y in range(min(300, im.height)):
        row = [im.getpixel((x, y)) for x in range(0, pw, 5)]
        if sum(min(p) > 240 for p in row) / len(row) > 0.6:
            th = y + 1
        else:
            break
ph = im.height - th
l, t, r, b = a.box
crops = []
for k in range(a.panels):
    x0 = k * (pw + a.gap)
    title = im.crop((x0, 0, x0 + pw, th)) if th else None
    body = im.crop((x0 + int(l * pw), th + int(t * ph), x0 + int(r * pw), th + int(b * ph)))
    if title is not None:  # re-centre the title over the cropped width
        tw = body.width
        strip = Image.new("RGB", (tw, th), (255, 255, 255))
        mid = title.crop((max(0, pw // 2 - tw // 2), 0, min(pw, pw // 2 + tw // 2), th))
        strip.paste(mid, ((tw - mid.width) // 2, 0))
        panel = Image.new("RGB", (tw, th + body.height), (255, 255, 255))
        panel.paste(strip, (0, 0)); panel.paste(body, (0, th))
        body = panel
    crops.append(body)
w = sum(c.width for c in crops) + a.gap * (len(crops) - 1)
out = Image.new("RGB", (w, crops[0].height), (255, 255, 255))
x = 0
for c in crops:
    out.paste(c, (x, 0)); x += c.width + a.gap
out.save(a.out, quality=95)
print("saved", a.out, out.size)
