"""One trainer for every run in the report.

  Labeller (step 1)      --arch dino --source SRC
  B0 source-only         --arch r50  --source SRC
  Oracle                 --arch r50  --source SRC --extra-labeled TGT_FULL
  B1 Mean Teacher        --arch r50  --source SRC --target TGT --target-labels teacher
  B3 MT + tag filtering  ... --target-labels teacher --tag-filter
  B2 / Ours (step 3)     --arch r50  --source SRC --target-labels pseudo --pseudo PSEUDO.json

The student always gets strong augmentation on target images; the EMA copy of the
student is evaluated, saved and used as the final model (and as the online teacher).
"""
import argparse
import copy
import json
import math
import os
import random
import time

import torch

from tagdaod.data import DetectionDataset, infinite, load_json, make_loader
from tagdaod.evaluation import evaluate
from tagdaod.models import build_model, save_checkpoint


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["dino", "r50"], required=True)
    p.add_argument("--dino-model", default="vitb", help="vits | vitb | vitl or a timm model name")
    p.add_argument("--pretrained", default="coco", choices=["coco", "imagenet", "none"],
                   help="r50: COCO/ImageNet init. dino: anything but 'none' loads DINOv2 weights")
    p.add_argument("--source", required=True, help="labelled source JSON")
    p.add_argument("--extra-labeled", default=None, help="add labelled target images (oracle only)")
    p.add_argument("--target", default=None, help="target train JSON (tags only or no labels)")
    p.add_argument("--target-labels", choices=["none", "pseudo", "teacher"], default="none")
    p.add_argument("--pseudo", default=None, help="refined pseudo-label JSON (target-labels=pseudo)")
    p.add_argument("--teacher-thresh", type=float, default=0.8)
    p.add_argument("--tag-filter", action="store_true", help="online teacher: drop classes not in tags")
    p.add_argument("--target-weight", type=float, default=1.0)
    p.add_argument("--burn-in", type=int, default=0, help="source-only iterations before target loss")
    p.add_argument("--source-strong", action="store_true", help="strong augmentation on source images too")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4, help="images per domain per iteration")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.9996)
    p.add_argument("--min-size", type=int, default=600)
    p.add_argument("--max-size", type=int, default=1000)
    p.add_argument("--eval", default=None, help="target test JSON evaluated during training")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def clean(targets, device):
    return [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]


@torch.no_grad()
def teacher_targets(teacher, imgs, targets, thresh, tag_filter, device):
    out = []
    for p, t in zip(teacher(imgs), targets):
        keep = p["scores"] >= thresh
        if tag_filter:
            tags = t["tags"].to(device)
            keep &= tags[p["labels"] - 1]
        out.append({"boxes": p["boxes"][keep].detach(), "labels": p["labels"][keep].detach()})
    return out


@torch.no_grad()
def ema_update(ema, model, decay):
    msd = model.state_dict()
    for k, v in ema.state_dict().items():
        if k.startswith("backbone.vit."):
            continue
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
        else:
            v.copy_(msd[k])


def main():
    a = args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = a.amp and device.type == "cuda"
    sizes = dict(min_size=a.min_size, max_size=a.max_size)

    src_ann = load_json(a.source)
    if a.extra_labeled:
        src_ann = {"classes": src_ann["classes"], "images": src_ann["images"] + load_json(a.extra_labeled)["images"]}
    src_ds = DetectionDataset(src_ann, train=True, strong=a.source_strong, **sizes)
    src_it = infinite(make_loader(src_ds, a.batch, a.workers, shuffle=True))

    tgt_it = None
    if a.target_labels == "pseudo":
        tgt_ds = DetectionDataset(a.pseudo, train=True, strong=True, **sizes)
        tgt_it = infinite(make_loader(tgt_ds, a.batch, a.workers, shuffle=True))
    elif a.target_labels == "teacher":
        tgt_ds = DetectionDataset(a.target, train=True, strong=True, use_boxes=False, **sizes)
        tgt_it = infinite(make_loader(tgt_ds, a.batch, a.workers, shuffle=True))
    if tgt_it is not None:
        assert tgt_ds.classes == src_ds.classes, "source and target class lists differ"

    cfg = dict(arch=a.arch, num_classes=len(src_ds.classes), classes=src_ds.classes,
               dino_model=a.dino_model, pretrained=a.pretrained, **sizes)
    model = build_model(cfg).to(device).train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M on {device}")
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log = open(os.path.join(a.out, "log.txt"), "a")
    best, history, t0 = -1.0, [], time.time()
    for it in range(1, a.iters + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, it / max(1, a.warmup))

        # Source and target losses are back-propagated one after the other: the summed gradient is
        # identical to backward(src + w * tgt), but only one batch of activations is held in GPU
        # memory at a time (avoids Windows spilling GPU memory into system RAM).
        opt.zero_grad(set_to_none=True)
        batch = next(src_it)
        imgs, targets = (batch[1], batch[2]) if a.source_strong else batch
        with torch.autocast(device.type, enabled=use_amp):
            losses = model([im.to(device) for im in imgs], clean(targets, device))
            loss = sum(losses.values())
        stats = {"src": loss.item()}
        if not math.isfinite(stats["src"]):
            raise RuntimeError(f"source loss is {stats['src']} at iter {it}; lower --lr")
        scaler.scale(loss).backward()
        del losses, loss

        if tgt_it is not None and it > a.burn_in:
            weak, strong, tt = next(tgt_it)
            with torch.autocast(device.type, enabled=use_amp):
                if a.target_labels == "pseudo":
                    ttargets = clean(tt, device)
                else:
                    ttargets = teacher_targets(ema, [im.to(device) for im in weak], tt,
                                               a.teacher_thresh, a.tag_filter, device)
                tl = model([im.to(device) for im in strong], ttargets)
                tloss = sum(tl.values())
            stats["tgt"] = tloss.item()
            stats["tgt_boxes/img"] = sum(len(t["boxes"]) for t in ttargets) / len(ttargets)
            if not math.isfinite(stats["tgt"]):
                raise RuntimeError(f"target loss is {stats['tgt']} at iter {it}; lower --lr")
            scaler.scale(a.target_weight * tloss).backward()
            del tl, tloss
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt); scaler.update()
        ema_update(ema, model, min(a.ema, (1 + it) / (10 + it)))

        if it % a.log_every == 0 or it == 1:
            el = time.time() - t0
            msg = (f"iter {it}/{a.iters}  " + "  ".join(f"{k} {v:.3f}" for k, v in stats.items())
                   + f"  lr {opt.param_groups[0]['lr']:.4f}  {el / it:.2f}s/it  eta {el / it * (a.iters - it) / 3600:.1f}h")
            print(msg, flush=True); log.write(msg + "\n"); log.flush()

        if a.eval and (it % a.eval_every == 0 or it == a.iters):
            res = evaluate(ema, a.eval, device, a.min_size, a.max_size, workers=a.workers)
            history.append({"iter": it, **res})
            msg = f"[eval] iter {it}  mAP50 {res['mAP50']:.2f}  (VOC07 11-pt {res['mAP50_voc07']:.2f})"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            json.dump(history, open(os.path.join(a.out, "eval_history.json"), "w"), indent=1)
            if res["mAP50"] > best:
                best = res["mAP50"]
                save_checkpoint(ema, cfg, os.path.join(a.out, "best.pth"), iter=it, result=res)
            ema.eval()

        if it % 5000 == 0 or it == a.iters:
            save_checkpoint(ema, cfg, os.path.join(a.out, "last.pth"), iter=it)
    print("done; best mAP50", round(best, 2) if best >= 0 else "n/a")


if __name__ == "__main__":
    main()
