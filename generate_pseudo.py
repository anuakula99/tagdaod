"""Step 1b: run the trained DINOv2 labeller once on the target training images.

Saves every detection with score >= --min-score (default 0.05) in original pixel
coordinates, so refine_pseudo.py can apply both the high and the low threshold.
  python generate_pseudo.py --ckpt runs/clipart/labeller/last.pth \
      --data data/json/clipart_train_tags.json --out runs/clipart/pseudo_raw.json
"""
import argparse

import torch

from tagdaod.data import DetectionDataset, save_json
from tagdaod.evaluation import predict_dataset
from tagdaod.models import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="target train JSON (tags only / unlabelled)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-score", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(a.ckpt, device)
    ds = DetectionDataset(a.data, train=False, use_boxes=False, min_size=cfg["min_size"], max_size=cfg["max_size"])
    assert ds.classes == cfg["classes"], "checkpoint and data use different class lists"
    out = [None] * len(ds)
    for i, pred, t in predict_dataset(model, ds, device, a.batch, a.workers):
        keep = pred["scores"] >= a.min_score
        rec = dict(ds.images[i])
        rec["boxes"] = (pred["boxes"][keep] / t["scale"]).tolist()
        rec["labels"] = (pred["labels"][keep] - 1).tolist()
        rec["scores"] = pred["scores"][keep].tolist()
        rec.pop("difficult", None)
        out[i] = rec
    save_json({"classes": ds.classes, "images": out}, a.out)
    n = sum(len(r["boxes"]) for r in out)
    print(f"{len(out)} images, {n} raw detections (score >= {a.min_score}) -> {a.out}")


if __name__ == "__main__":
    main()
