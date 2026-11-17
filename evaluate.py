"""mAP@0.5 of a checkpoint on a labelled JSON.
  python evaluate.py --ckpt runs/clipart/ours/best.pth --data data/json/clipart_test.json
"""
import argparse
import json

import torch

from tagdaod.evaluation import evaluate
from tagdaod.models import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None, help="optional JSON file for the result")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(a.ckpt, device)
    res = evaluate(model, a.data, device, cfg["min_size"], cfg["max_size"], workers=a.workers)
    for c, ap in res["per_class"].items():
        print(f"  {c:14s} {ap:6.2f}")
    print(f"mAP@0.5 {res['mAP50']:.2f}   (VOC07 11-point: {res['mAP50_voc07']:.2f})")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
