"""PASCAL VOC style mAP@0.5 (difficult objects ignored)."""
import numpy as np
import torch

from .data import DetectionDataset, make_loader


def voc_ap(rec, prec, use07=False):
    if use07:  # 11-point interpolation (VOC2007 devkit)
        return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0
                              for t in np.arange(0.0, 1.1, 0.1)]))
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def class_ap(gts, dets, num_classes, iou_thr=0.5):
    """gts/dets: per-image dicts with numpy boxes, 0-based labels (+ difficult / scores).
    Returns (APs all-point, APs 11-point) per class; NaN for classes absent from GT."""
    ap_all, ap_07 = [], []
    for c in range(num_classes):
        gt_c = []
        npos = 0
        for g in gts:
            m = g["labels"] == c
            diff = g["difficult"][m]
            gt_c.append({"boxes": g["boxes"][m], "difficult": diff, "used": np.zeros(m.sum(), bool)})
            npos += int((~diff).sum())
        items = [(i, s, b) for i, d in enumerate(dets)
                 for b, l, s in zip(d["boxes"], d["labels"], d["scores"]) if l == c]
        if npos == 0:
            ap_all.append(float("nan")); ap_07.append(float("nan")); continue
        items.sort(key=lambda x: -x[1])
        tp = np.zeros(len(items)); fp = np.zeros(len(items))
        for k, (i, _, b) in enumerate(items):
            g = gt_c[i]
            if len(g["boxes"]) == 0:
                fp[k] = 1; continue
            ov = iou(b, g["boxes"]); j = int(ov.argmax())
            if ov[j] >= iou_thr:
                if g["difficult"][j]:
                    continue
                if not g["used"][j]:
                    tp[k] = 1; g["used"][j] = True
                else:
                    fp[k] = 1
            else:
                fp[k] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        rec = tp / npos
        prec = tp / np.maximum(tp + fp, 1e-9)
        ap_all.append(voc_ap(rec, prec)); ap_07.append(voc_ap(rec, prec, use07=True))
    return ap_all, ap_07


@torch.no_grad()
def predict_dataset(model, ds, device, batch_size=2, workers=2):
    """Yields (index, prediction, target). Boxes are in the dataset's resized coordinates;
    target['scale'] maps back to original pixels."""
    model.eval()
    for imgs, targets in make_loader(ds, batch_size, workers, shuffle=False):
        preds = model([im.to(device) for im in imgs])
        for p, t in zip(preds, targets):
            yield t["index"], {k: v.cpu() for k, v in p.items()}, t


def evaluate(model, ann, device, min_size, max_size, batch_size=2, workers=2):
    ds = DetectionDataset(ann, train=False, min_size=min_size, max_size=max_size)
    gts, dets = [], []
    for _, p, t in predict_dataset(model, ds, device, batch_size, workers):
        gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy() - 1,
                    "difficult": t["difficult"].numpy()})
        dets.append({"boxes": p["boxes"].numpy(), "labels": p["labels"].numpy() - 1,
                     "scores": p["scores"].numpy()})
    ap_all, ap_07 = class_ap(gts, dets, len(ds.classes))
    return {"mAP50": float(np.nanmean(ap_all)) * 100, "mAP50_voc07": float(np.nanmean(ap_07)) * 100,
            "per_class": {c: round(a * 100, 2) for c, a in zip(ds.classes, ap_all)}}
