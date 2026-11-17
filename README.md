# Tag-Guided Foundation-Model Pseudo-Labels for Domain Adaptive Object Detection

CS6350 Computer Vision, Term Project Assignment 1 (TPA-1), IIT Madras.
Team: Pokala Mohana Vignesh Kumar (DA25S016), Akula Anurag (DA26D007).

A detector trained on labelled real photographs is adapted to a new visual domain (clipart, fog)
without any boxes in that domain:

1. **Labeller.** A Faster R-CNN head on a **frozen DINOv2 ViT-B/14** backbone is trained on source boxes
   and labels every target training image once (pseudo-labels).
2. **Tag refinement.** Image-level tags clean the pseudo-labels:
   *filter* drops boxes of classes not in the tags; *recover* keeps the best box of a tagged class
   that has no confident box.
3. **Student.** A standard **ResNet-50 FPN Faster R-CNN** is trained on source boxes plus the refined
   pseudo-labels. Only the student runs at test time (about 0.5 s per image on a laptop CPU).

```
source boxes -> [1] DINOv2 labeller -> pseudo-labels -> [2] tag filter + recover -> [3] ResNet-50 student -> demo / eval
                  train.py --arch dino   generate_pseudo.py   refine_pseudo.py         train.py --arch r50
```

Plain PyTorch + torchvision + timm (no Detectron2).

## Results (mAP@0.5, final checkpoint)

**PASCAL VOC 07+12 -> Clipart1k** (500 test images, target train images carry image tags only)

| Method | Target labels used | mAP@0.5 |
|---|---|---|
| B0 Source-only | none | 35.8 |
| B1 Mean Teacher | none | 50.8 |
| B3 Mean Teacher + tag filter | tags | 52.8 |
| B2 DINOv2 pseudo-labels, no tags | none | 53.9 |
| **Ours: DINOv2 pseudo-labels + tag filter + recover** | tags | **57.6** |
| Oracle (same pipeline, real target boxes) | boxes | 65.0 |
| *DINOv2 labeller (reference, not deployed)* | none | *72.4* |

**Cityscapes -> Foggy Cityscapes** (beta 0.02, 500 val images, target unlabelled)

| Method | mAP@0.5 |
|---|---|
| B0 Source-only | 40.3 |
| B1 Mean Teacher | 45.7 |
| **Ours (DINOv2 pseudo-labels, no tags)** | **50.5** |
| *DINOv2 labeller (reference)* | *49.3* |

Ablation on Clipart (student mAP): threshold only 53.9, + filter 53.5, + recover 57.5, + both 57.6.
Pseudo-label precision / recall: 84.1 / 58.9 (threshold) -> 86.2 / 66.4 (filter + recover).

## Files

| File | Purpose |
|---|---|
| `tools/prepare_voc.py` | VOC / Clipart1k XML -> project JSON (`--tags-only` keeps only image tags) |
| `tools/prepare_cityscapes.py` | Cityscapes / Foggy Cityscapes instance masks -> 8-class boxes |
| `train.py` | One trainer for the labeller, every baseline, our method and the oracle |
| `generate_pseudo.py` | Labeller -> raw detections on target training images |
| `refine_pseudo.py` | `threshold` / `filter` / `recover` / `filter+recover`; `--gt` prints precision / recall |
| `evaluate.py` | mAP@0.5 (all-point and VOC07 11-point) with per-class AP |
| `demo.py` | Folder of images -> annotated images; several checkpoints -> side-by-side panels |
| `tools/crop_panels.py` | Crops the same region from every panel of a side-by-side image (figures) |
| `scripts/*_priority.ps1` | Windows PowerShell: every run, stage by stage (`-Stage data`, `labeller`, ...) |
| `scripts/*.sh` | Same commands for Linux / bash |
| `scripts/smoke_test.*` | Whole pipeline on synthetic data in a few minutes |
| `VIVA_COMMANDS.txt` | Demo and evaluation commands with comments |

## Setup

```bash
# Python 3.11. Install the CUDA build of PyTorch for your driver first, e.g. CUDA 12.1:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1     # Windows; or: bash scripts/smoke_test.sh
```
The first real run downloads DINOv2 (Hugging Face, via timm) and the COCO Faster R-CNN weights
(download.pytorch.org). After that, training and the demo work offline.

## Data

Not included in this repository (licences). Expected layout:
```
data/VOCdevkit/VOC2007, data/VOCdevkit/VOC2012    PASCAL VOC trainval
data/clipart                                      Clipart1k (Annotations/, JPEGImages/, ImageSets/Main/{train,test}.txt)
data/cityscapes/{leftImg8bit,gtFine}              Cityscapes
data/foggy/leftImg8bit_foggy                      Foggy Cityscapes (leftImg8bit_trainvaltest_foggy.zip)
```
Sources: PASCAL VOC http://host.robots.ox.ac.uk/pascal/VOC/ ·
Clipart1k https://github.com/naoto0804/cross-domain-detection (`datasets/prepare.sh`, Google Drive via `gdown`) ·
Cityscapes and Foggy Cityscapes https://www.cityscapes-dataset.com (account required; `pip install cityscapesscripts`, then `csDownload`).

## Reproducing (Windows PowerShell)

```powershell
# VOC -> Clipart1k
powershell -ExecutionPolicy Bypass -File scripts\run_clipart_priority.ps1 -Stage data
powershell -ExecutionPolicy Bypass -File scripts\run_clipart_priority.ps1 -Stage labeller            # 20k iterations
powershell -ExecutionPolicy Bypass -File scripts\run_clipart_priority.ps1 -Stage night -Iters 10000  # pseudo-labels, 4 refinements, Ours, B2
powershell -ExecutionPolicy Bypass -File scripts\run_clipart_priority.ps1 -Stage b0 -Iters 10000     # also: b1, b3, ablation
# Cityscapes -> Foggy Cityscapes
powershell -ExecutionPolicy Bypass -File scripts\run_foggy_priority.ps1 -Stage data
powershell -ExecutionPolicy Bypass -File scripts\run_foggy_priority.ps1 -Stage labeller
powershell -ExecutionPolicy Bypass -File scripts\run_foggy_priority.ps1 -Stage ours -Iters 10000     # also: b0, b1
```
Matched oracle (Clipart): `train.py --arch r50 ... --target-labels pseudo --pseudo data/json/clipart_train_full.json`.
The `*_full.json` target files contain boxes; they are used only for the oracle and the pseudo-label
precision / recall analysis, never by our method.

Settings: shorter side 600 px (longer side <= 1000 for VOC/Clipart, <= 1200 for Cityscapes); batch 4 source
+ 4 target images; SGD, lr 0.01, momentum 0.9, weight decay 1e-4, 500 warm-up iterations; mixed precision;
EMA decay 0.9996 (the EMA model is the final model); thresholds high 0.8, low 0.3. Labeller 20k iterations,
students 10k. Hardware: one NVIDIA RTX 4060 Laptop GPU (8 GB); about 2 to 3 h per labeller and 1.2 to 2.2 h per student.

## Demo

```powershell
python demo.py --ckpt runs/clipart/ours/last.pth --input samples --out outputs --device cpu   # clipart / art images, 20 VOC classes
python demo.py --ckpt runs/foggy/ours/last.pth   --input samples --out outputs --device cpu   # street scenes, 8 Cityscapes classes
python demo.py --ckpt runs/clipart/b0/last.pth runs/clipart/b2/last.pth runs/clipart/ours/last.pth `
    --names "Source-only" "DINOv2, no tags" "Ours" --gt data/json/clipart_test.json --input samples --out compare
```
Model checkpoints are not in the repository (size); train them with the commands above.

## References

- N. Inoue et al., "Cross-domain weakly-supervised object detection through progressive domain adaptation," CVPR 2018.
- M.-A. Lavoie, A. Mahmoud, S. L. Waslander, "Large self-supervised models bridge the gap in domain adaptive object detection" (DINO Teacher), CVPR 2025.
- J. Kay et al., "Align and distill: Unifying and improving domain adaptive object detection" (ALDI), TMLR 2025.
- M. Oquab et al., "DINOv2: Learning robust visual features without supervision," TMLR 2024.
- M. Cordts et al., "The Cityscapes dataset for semantic urban scene understanding," CVPR 2016.
- C. Sakaridis, D. Dai, L. Van Gool, "Semantic foggy scene understanding with synthetic data," IJCV 2018.
