# Windows PowerShell: Cityscapes -> Foggy Cityscapes (beta 0.02), no tags on the target.
#   powershell -ExecutionPolicy Bypass -File scripts\run_foggy_priority.ps1 -Stage data
# Stages: data, labeller, ours (pseudo-labels + student), b0, b1, oracle.
param([Parameter(Mandatory)][ValidateSet("data", "labeller", "ours", "b0", "b1", "oracle")][string]$Stage,
      [int]$Iters = 20000, [int]$Batch = 4, [double]$Lr = 0.01)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Py { & python @args; if ($LASTEXITCODE -ne 0) { throw "FAILED: python $args" } }

$J = "data/json"; $R = "runs/foggy"
$C = @("--min-size", "600", "--max-size", "1200", "--batch", "$Batch", "--lr", "$Lr", "--amp", "--iters", "$Iters",
       "--eval", "$J/foggy_val.json", "--eval-every", "2000")
$SRC = @("--source", "$J/cityscapes_train.json")
$CS = @("--cityscapes", "data/cityscapes", "--foggy", "data/foggy", "--workers", "4")

switch ($Stage) {
  "data" {
    Py tools/prepare_cityscapes.py @CS --split train --domain clear --out "$J/cityscapes_train.json"
    Py tools/prepare_cityscapes.py @CS --split train --domain foggy --no-boxes --out "$J/foggy_train_unlabeled.json"
    Py tools/prepare_cityscapes.py @CS --split train --domain foggy --out "$J/foggy_train_full.json"
    Py tools/prepare_cityscapes.py @CS --split val --domain foggy --out "$J/foggy_val.json"
  }
  "labeller" {
    Py train.py --arch dino --dino-model vitb @C @SRC --out "$R/labeller"
  }
  "ours" {
    Py generate_pseudo.py --ckpt "$R/labeller/last.pth" --data "$J/foggy_train_unlabeled.json" --out "$R/pseudo_raw.json"
    Py refine_pseudo.py --raw "$R/pseudo_raw.json" --mode threshold --high 0.8 `
        --out "$R/pseudo_threshold.json" --gt "$J/foggy_train_full.json" | Tee-Object "$R/refine_report.txt"
    Py train.py --arch r50 @C @SRC --target-labels pseudo --pseudo "$R/pseudo_threshold.json" --out "$R/ours"
  }
  "b0" {
    Py train.py --arch r50 @C @SRC --source-strong --out "$R/b0"
  }
  "b1" {
    Py train.py --arch r50 @C @SRC --target "$J/foggy_train_unlabeled.json" --target-labels teacher --burn-in 5000 --out "$R/b1"
  }
  "oracle" {
    Py train.py --arch r50 @C @SRC --extra-labeled "$J/foggy_train_full.json" --source-strong --out "$R/oracle"
  }
}
Write-Host "Stage $Stage done."
