#!/usr/bin/env bash
# Cityscapes -> Foggy Cityscapes (beta 0.02), fully unsupervised target: no tags.
# "Ours" here is the DINOv2 labeller with the confidence threshold only (= B2 setting).
set -euo pipefail
cd "$(dirname "$0")/.."
J=data/json; R=runs/foggy
EV="--eval $J/foggy_val.json --eval-every 2000"
COMMON="--min-size 600 --max-size 1200 --batch 4 --lr 0.01 --amp"

# ---- 0. data -----------------------------------------------------------------------------
python tools/prepare_cityscapes.py --cityscapes data/cityscapes --split train --domain clear --out $J/cityscapes_train.json
python tools/prepare_cityscapes.py --cityscapes data/cityscapes --foggy data/foggy --split train --domain foggy --no-boxes --out $J/foggy_train_unlabeled.json
python tools/prepare_cityscapes.py --cityscapes data/cityscapes --foggy data/foggy --split train --domain foggy --out $J/foggy_train_full.json  # analysis + oracle only
python tools/prepare_cityscapes.py --cityscapes data/cityscapes --foggy data/foggy --split val --domain foggy --out $J/foggy_val.json

# ---- B0 source-only ------------------------------------------------------------------------
python train.py --arch r50 $COMMON --source $J/cityscapes_train.json --source-strong --iters 20000 $EV --out $R/b0

# ---- Step 1 labeller + pseudo-labels -------------------------------------------------------
python train.py --arch dino --dino-model vitb $COMMON --source $J/cityscapes_train.json --iters 20000 $EV --out $R/labeller
python generate_pseudo.py --ckpt $R/labeller/best.pth --data $J/foggy_train_unlabeled.json --out $R/pseudo_raw.json
python refine_pseudo.py --raw $R/pseudo_raw.json --mode threshold --high 0.8 --out $R/pseudo_threshold.json --gt $J/foggy_train_full.json

# ---- Ours (no tags) --------------------------------------------------------------------------
python train.py --arch r50 $COMMON --source $J/cityscapes_train.json --target-labels pseudo \
    --pseudo $R/pseudo_threshold.json --iters 20000 $EV --out $R/ours

# ---- B1 Mean Teacher ------------------------------------------------------------------------
python train.py --arch r50 $COMMON --source $J/cityscapes_train.json --target $J/foggy_train_unlabeled.json \
    --target-labels teacher --burn-in 5000 --iters 20000 $EV --out $R/b1

# ---- Oracle ---------------------------------------------------------------------------------
python train.py --arch r50 $COMMON --source $J/cityscapes_train.json --extra-labeled $J/foggy_train_full.json \
    --source-strong --iters 20000 $EV --out $R/oracle
