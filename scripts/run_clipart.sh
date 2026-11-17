#!/usr/bin/env bash
# PASCAL VOC -> Clipart1k (weak: image tags on the target). Every run in the report table.
# Run one block at a time; each block is independent once its inputs exist.
set -euo pipefail
cd "$(dirname "$0")/.."
J=data/json; R=runs/clipart
EV="--eval $J/clipart_test.json --eval-every 2000"
COMMON="--min-size 600 --max-size 1000 --batch 4 --lr 0.01 --amp"

# ---- 0. data -----------------------------------------------------------------------------
python tools/prepare_voc.py --dirs data/VOCdevkit/VOC2007 data/VOCdevkit/VOC2012 --split trainval --out $J/voc_trainval.json
python tools/prepare_voc.py --dirs data/clipart --split train --tags-only --out $J/clipart_train_tags.json
python tools/prepare_voc.py --dirs data/clipart --split train --out $J/clipart_train_full.json   # analysis + oracle only
python tools/prepare_voc.py --dirs data/clipart --split test --out $J/clipart_test.json

# ---- B0: fair source-only (strong aug + EMA, as in ALDI) ----------------------------------
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --source-strong --iters 20000 $EV --out $R/b0

# ---- Step 1: DINOv2 labeller (frozen ViT-B, trains the head only) -------------------------
python train.py --arch dino --dino-model vitb $COMMON --source $J/voc_trainval.json --iters 20000 $EV --out $R/labeller
python generate_pseudo.py --ckpt $R/labeller/best.pth --data $J/clipart_train_tags.json --out $R/pseudo_raw.json

# ---- Step 2: refinement (also prints the pseudo-label precision/recall table) -------------
for m in threshold filter recover filter+recover; do
  python refine_pseudo.py --raw $R/pseudo_raw.json --mode $m --high 0.8 --low 0.3 \
      --out $R/pseudo_${m/+/_}.json --gt $J/clipart_train_full.json
done

# ---- B2: student on untagged DINOv2 pseudo-labels -----------------------------------------
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --target-labels pseudo \
    --pseudo $R/pseudo_threshold.json --iters 20000 $EV --out $R/b2
# ---- Ours: student on tag-refined pseudo-labels (filter + recover) ------------------------
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --target-labels pseudo \
    --pseudo $R/pseudo_filter_recover.json --iters 20000 $EV --out $R/ours

# ---- B1: Mean Teacher (online EMA teacher, no tags) / B3: + tag filtering -----------------
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --target $J/clipart_train_tags.json \
    --target-labels teacher --burn-in 5000 --iters 20000 $EV --out $R/b1
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --target $J/clipart_train_tags.json \
    --target-labels teacher --tag-filter --burn-in 5000 --iters 20000 $EV --out $R/b3

# ---- Oracle: target boxes added to training -----------------------------------------------
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --extra-labeled $J/clipart_train_full.json \
    --source-strong --iters 20000 $EV --out $R/oracle

# ---- Ablation rows: filter-only / recover-only students -----------------------------------
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --target-labels pseudo \
    --pseudo $R/pseudo_filter.json --iters 20000 $EV --out $R/abl_filter
python train.py --arch r50 $COMMON --source $J/voc_trainval.json --target-labels pseudo \
    --pseudo $R/pseudo_recover.json --iters 20000 $EV --out $R/abl_recover
