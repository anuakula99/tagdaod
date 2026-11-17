# Tag-Guided Foundation-Model Pseudo-Labels for Domain Adaptive Object Detection

CS6350 TPA-1. A frozen DINOv2 detector labels the target images once, image-level tags clean
those pseudo-labels, and a standard ResNet-50 Faster R-CNN student learns from source boxes plus
the cleaned labels. Only the student runs at test time.

```
source boxes ──► [1] train DINOv2 labeller ──► pseudo_raw.json ──► [2] refine with tags ──► [3] train student ──► demo / eval
                    train.py --arch dino      generate_pseudo.py     refine_pseudo.py        train.py --arch r50
```

Everything is plain PyTorch + torchvision + timm (no Detectron2 build). Training follows the ALDI
protocol: strong augmentation, an EMA copy of the student as the final model, and fair baselines.

## Files

| File | Purpose |
|---|---|
| `tools/prepare_voc.py` | VOC / Clipart1k XML → project JSON (`--tags-only` keeps only image tags) |
| `tools/prepare_cityscapes.py` | Cityscapes / Foggy Cityscapes instance masks → 8-class boxes |
| `train.py` | One trainer for the labeller, every baseline, our method and the oracle |
| `generate_pseudo.py` | Step 1b: labeller → raw detections on target train images |
| `refine_pseudo.py` | Step 2: `threshold` / `filter` / `recover` / `filter+recover`; `--gt` prints precision/recall |
| `evaluate.py` | mAP@0.5 (all-point and VOC07 11-point) with per-class AP |
| `demo.py` | Folder of images → annotated images; several checkpoints → side-by-side panels |
| `scripts/run_clipart.sh`, `scripts/run_foggy.sh` | Exact commands for every row of the report tables |
| `scripts/smoke_test.sh` | Whole pipeline on synthetic data in a few minutes (CPU) |

## Setup

```bash
pip install -r requirements.txt          # install the CUDA build of torch for your GPU first
bash scripts/smoke_test.sh               # must end with "SMOKE TEST PASSED"
```
The first real run downloads DINOv2 (from Hugging Face, via timm) and COCO Faster R-CNN weights
(from download.pytorch.org).

## Data layout

```
data/VOCdevkit/VOC2007, data/VOCdevkit/VOC2012     PASCAL VOC (Annotations/, JPEGImages/, ImageSets/)
data/clipart                                       Clipart1k, same VOC layout, ImageSets/Main/{train,test}.txt
data/cityscapes/{leftImg8bit,gtFine}               Cityscapes
data/foggy/leftImg8bit_foggy                       Foggy Cityscapes
```
Sources: VOC http://host.robots.ox.ac.uk/pascal/VOC/ · Clipart1k
https://github.com/naoto0804/cross-domain-detection (`datasets/download_clipart.sh`) ·
Cityscapes and Foggy Cityscapes https://www.cityscapes-dataset.com (login needed).

## What each run is

| Report ID | Command (see the run scripts for full arguments) |
|---|---|
| B0 source-only | `train.py --arch r50 --source SRC --source-strong` |
| Labeller | `train.py --arch dino --dino-model vitb --source SRC` |
| B2 DINOv2, no tags | `refine_pseudo.py --mode threshold`, then `train.py --target-labels pseudo` |
| **Ours** | `refine_pseudo.py --mode filter+recover`, then `train.py --target-labels pseudo` |
| B1 Mean Teacher | `train.py --target-labels teacher --burn-in 5000` |
| B3 MT + tag filter | `train.py --target-labels teacher --tag-filter --burn-in 5000` |
| Oracle | `train.py --extra-labeled TGT_FULL --source-strong` |

The `*_full.json` target files contain boxes. They are used only for the oracle and for the
pseudo-label precision/recall table, never by our method.

## Running order with limited time

Do these first; they give the main table and the demo model:
1. data prep, then `smoke_test.sh` on the real machine;
2. Clipart: labeller → pseudo-labels → 4 refinement modes (gives the precision/recall table) → **Ours** → B2 → B0;
3. Foggy: labeller → **Ours** → B0;
4. if time remains: B1, B3, oracle, ablation students.

Read the `s/it` and `eta` printed in the training log after a few hundred iterations. If 20k
iterations does not fit, use `--iters 10000` for all runs of a table so they stay comparable.
Out of GPU memory: `--batch 2 --lr 0.005`, or for the labeller `--dino-model vits`.

## Demo (viva)

```bash
python demo.py --ckpt runs/clipart/ours/best.pth --input SAMPLES/ --out outputs_clipart/ --device cpu
python demo.py --ckpt runs/foggy/ours/best.pth   --input SAMPLES/ --out outputs_foggy/   --device cpu
```
The two models have different class lists (20 VOC classes vs 8 Cityscapes classes), so run the one
that matches the sample images (or both). Only the ResNet-50 student runs, about 1–3 s per image on
a laptop CPU. Qualitative figure for the report:
```bash
python demo.py --ckpt runs/clipart/b0/best.pth runs/clipart/b2/best.pth runs/clipart/ours/best.pth \
    --names "Source-only" "DINOv2, no tags" "Ours" --gt data/json/clipart_test.json \
    --input some_test_images/ --out qualitative/
```

## Notes for the report

- mAP is reported at IoU 0.5; `evaluate.py` prints both all-point and VOC07 11-point AP. Say which
  one you report (papers on Clipart1k mostly use VOC07-style evaluation; check before comparing).
- Images: shorter side 600 px (VOC/Clipart longer side ≤ 1000, Cityscapes ≤ 1200).
- Labeller: DINOv2 ViT-B/14 frozen; a simple feature pyramid (strides 8–128), RPN and box head are trained.
- Pseudo-label thresholds: high 0.8, low 0.3; EMA decay 0.9996; SGD momentum 0.9, lr 0.01 for 4+4 images.
