# Windows PowerShell: the essential VOC -> Clipart1k runs in priority order.
#   powershell -ExecutionPolicy Bypass -File scripts\run_clipart_priority.ps1 -Stage data
# Stages: data, labeller, night (pseudo-labels + refine + Ours + B2), b0, b1, b3, oracle, ablation.
# Change -Iters to 10000 if short on time.
param([Parameter(Mandatory)][ValidateSet("data", "labeller", "night", "b0", "b1", "b3", "oracle", "ablation")][string]$Stage,
      [int]$Iters = 20000, [int]$Batch = 4, [double]$Lr = 0.01)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Py { & python @args; if ($LASTEXITCODE -ne 0) { throw "FAILED: python $args" } }

$J = "data/json"; $R = "runs/clipart"
$C = @("--min-size", "600", "--max-size", "1000", "--batch", "$Batch", "--lr", "$Lr", "--amp", "--iters", "$Iters",
       "--eval", "$J/clipart_test.json", "--eval-every", "2000")
$SRC = @("--source", "$J/voc_trainval.json")

switch ($Stage) {
  "data" {
    Py tools/prepare_voc.py --dirs data/VOCdevkit/VOC2007 data/VOCdevkit/VOC2012 --split trainval --out "$J/voc_trainval.json"
    Py tools/prepare_voc.py --dirs data/clipart --split train --tags-only --out "$J/clipart_train_tags.json"
    Py tools/prepare_voc.py --dirs data/clipart --split train --out "$J/clipart_train_full.json"
    Py tools/prepare_voc.py --dirs data/clipart --split test --out "$J/clipart_test.json"
  }
  "labeller" {
    Py train.py --arch dino --dino-model vitb @C @SRC --out "$R/labeller"
  }
  "night" {
    Py generate_pseudo.py --ckpt "$R/labeller/best.pth" --data "$J/clipart_train_tags.json" --out "$R/pseudo_raw.json"
    foreach ($m in "threshold", "filter", "recover", "filter+recover") {
      $n = $m -replace '\+', '_'
      Py refine_pseudo.py --raw "$R/pseudo_raw.json" --mode $m --high 0.8 --low 0.3 `
          --out "$R/pseudo_$n.json" --gt "$J/clipart_train_full.json" | Tee-Object -Append "$R/refine_report.txt"
    }
    Py train.py --arch r50 @C @SRC --target-labels pseudo --pseudo "$R/pseudo_filter_recover.json" --out "$R/ours"
    Py train.py --arch r50 @C @SRC --target-labels pseudo --pseudo "$R/pseudo_threshold.json" --out "$R/b2"
  }
  "b0" {
    Py train.py --arch r50 @C @SRC --source-strong --out "$R/b0"
  }
  "b1" {
    Py train.py --arch r50 @C @SRC --target "$J/clipart_train_tags.json" --target-labels teacher --burn-in 5000 --out "$R/b1"
  }
  "b3" {
    Py train.py --arch r50 @C @SRC --target "$J/clipart_train_tags.json" --target-labels teacher --tag-filter --burn-in 5000 --out "$R/b3"
  }
  "oracle" {
    Py train.py --arch r50 @C @SRC --extra-labeled "$J/clipart_train_full.json" --source-strong --out "$R/oracle"
  }
  "ablation" {
    Py train.py --arch r50 @C @SRC --target-labels pseudo --pseudo "$R/pseudo_filter.json" --out "$R/abl_filter"
    Py train.py --arch r50 @C @SRC --target-labels pseudo --pseudo "$R/pseudo_recover.json" --out "$R/abl_recover"
  }
}
Write-Host "Stage $Stage done."
