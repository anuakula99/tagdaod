#!/usr/bin/env bash
# Runs every step of the pipeline on tiny synthetic data, CPU friendly, ~10 minutes.
# Checks the code paths only; the numbers mean nothing (random-init backbones, few iterations).
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/toy; R=runs/toy
S="--min-size 224 --max-size 256 --workers 0 --log-every 10 --pretrained none"

python tools/make_toy_data.py --out $D --n 16
# step 1: labeller (small DINOv2 so it runs on CPU)
python train.py --arch dino --dino-model vits $S --source $D/source_train.json --iters 20 --batch 2 \
    --eval $D/target_test.json --eval-every 20 --out $R/labeller
python generate_pseudo.py --ckpt $R/labeller/last.pth --data $D/target_train_tags.json --out $R/pseudo_raw.json --workers 0
# step 2: refinement, every mode, with the precision/recall analysis
for m in threshold filter recover filter+recover; do
  python refine_pseudo.py --raw $R/pseudo_raw.json --mode $m --high 0.3 --low 0.05 \
      --out $R/pseudo_${m/+/_}.json --gt $D/target_train_full.json
done
# baselines and step 3
python train.py --arch r50 $S --source $D/source_train.json --iters 20 --batch 2 --source-strong --out $R/b0
python train.py --arch r50 $S --source $D/source_train.json --target $D/target_train_tags.json \
    --target-labels teacher --tag-filter --teacher-thresh 0.3 --burn-in 10 --iters 20 --batch 2 --out $R/b3
python train.py --arch r50 $S --source $D/source_train.json --target-labels pseudo \
    --pseudo $R/pseudo_filter_recover.json --iters 20 --batch 2 \
    --eval $D/target_test.json --eval-every 20 --out $R/ours
python evaluate.py --ckpt $R/ours/best.pth --data $D/target_test.json --workers 0
mkdir -p $R/samples && cp $D/images/test_00{0,1}.png $R/samples/
python demo.py --ckpt $R/b0/last.pth $R/ours/best.pth --names "Source-only" "Ours" \
    --input $R/samples --out $R/demo --thresh 0.05 --gt $D/target_test.json
echo "SMOKE TEST PASSED"
