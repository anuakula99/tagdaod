"""Step 2: tag-guided refinement of raw pseudo-labels (the new part of the method).

modes
  threshold       keep boxes with score >= high                      (B2, and Foggy: no tags)
  filter          + drop boxes whose class is not in the image tags
  recover         threshold + for each tagged class with no box, keep its best box if score >= low
  filter+recover  both                                                (Ours)

Optional --gt (target train JSON WITH boxes) reports pseudo-label precision / recall at
IoU 0.5. Those boxes are used only for this analysis table, never for training.
  python refine_pseudo.py --raw runs/clipart/pseudo_raw.json --mode filter+recover \
      --out runs/clipart/pseudo_ours.json --gt data/json/clipart_train_full.json
"""
import argparse

import numpy as np

from tagdaod.data import load_json, save_json
from tagdaod.evaluation import iou


def refine(rec, mode, high, low):
    dets = list(zip(rec["boxes"], rec["labels"], rec["scores"]))
    tags = rec.get("tags")
    use_tags = mode != "threshold"
    if use_tags and tags is None:
        raise ValueError(f"mode {mode} needs image tags, but {rec['file']} has none")
    kept = [d for d in dets if d[2] >= high]
    if "filter" in mode:
        kept = [d for d in kept if d[1] in tags]
    if "recover" in mode:
        present = {d[1] for d in kept}
        for c in tags:
            if c in present:
                continue
            cands = [d for d in dets if d[1] == c and d[2] >= low]
            if cands:
                kept.append(max(cands, key=lambda d: d[2]))
    out = dict(rec)
    out["boxes"] = [list(d[0]) for d in kept]
    out["labels"] = [d[1] for d in kept]
    out["scores"] = [d[2] for d in kept]
    return out


def precision_recall(pseudo, gt, num_classes, thr=0.5):
    gt_by_file = {r["file"]: r for r in gt["images"]}
    tp = fp = npos = 0
    per_class_tp, per_class_pos = np.zeros(num_classes), np.zeros(num_classes)
    for r in pseudo["images"]:
        g = gt_by_file[r["file"]]
        gb = np.array(g["boxes"], dtype=float).reshape(-1, 4)
        gl = np.array(g["labels"], dtype=int)
        npos += len(gl)
        np.add.at(per_class_pos, gl, 1)
        used = np.zeros(len(gl), bool)
        for b, l, _ in sorted(zip(r["boxes"], r["labels"], r["scores"]), key=lambda x: -x[2]):
            m = np.where((gl == l) & ~used)[0]
            if len(m):
                ov = iou(np.array(b), gb[m])
                j = int(ov.argmax())
                if ov[j] >= thr:
                    used[m[j]] = True; tp += 1; per_class_tp[l] += 1
                    continue
            fp += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(npos, 1)
    return prec, rec, per_class_tp / np.maximum(per_class_pos, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--mode", choices=["threshold", "filter", "recover", "filter+recover"], required=True)
    p.add_argument("--high", type=float, default=0.8)
    p.add_argument("--low", type=float, default=0.3)
    p.add_argument("--out", required=True)
    p.add_argument("--gt", default=None)
    a = p.parse_args()

    raw = load_json(a.raw)
    refined = {"classes": raw["classes"], "images": [refine(r, a.mode, a.high, a.low) for r in raw["images"]]}
    save_json(refined, a.out)
    n = sum(len(r["boxes"]) for r in refined["images"])
    print(f"mode {a.mode}: {n} pseudo-boxes on {len(refined['images'])} images "
          f"({n / max(1, len(refined['images'])):.2f}/img) -> {a.out}")
    if a.gt:
        prec, rec, cls_rec = precision_recall(refined, load_json(a.gt), len(raw["classes"]))
        print(f"pseudo-label precision {prec * 100:.1f}  recall {rec * 100:.1f}  (IoU 0.5)")
        print("per-class recall: " + ", ".join(f"{c} {r * 100:.0f}" for c, r in zip(raw["classes"], cls_rec)))


if __name__ == "__main__":
    main()
