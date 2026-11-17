# Windows PowerShell version of smoke_test.sh. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Py { & python @args; if ($LASTEXITCODE -ne 0) { throw "FAILED: python $args" } }

$D = "data/toy"; $R = "runs/toy"
$S = @("--min-size", "224", "--max-size", "256", "--workers", "0", "--log-every", "10", "--pretrained", "none")

Py tools/make_toy_data.py --out $D --n 16
Py train.py --arch dino --dino-model vits @S --source "$D/source_train.json" --iters 20 --batch 2 `
    --eval "$D/target_test.json" --eval-every 20 --out "$R/labeller"
Py generate_pseudo.py --ckpt "$R/labeller/last.pth" --data "$D/target_train_tags.json" --out "$R/pseudo_raw.json" --workers 0
foreach ($m in "threshold", "filter", "recover", "filter+recover") {
    $n = $m -replace '\+', '_'
    Py refine_pseudo.py --raw "$R/pseudo_raw.json" --mode $m --high 0.3 --low 0.05 `
        --out "$R/pseudo_$n.json" --gt "$D/target_train_full.json"
}
Py train.py --arch r50 @S --source "$D/source_train.json" --iters 20 --batch 2 --source-strong --out "$R/b0"
Py train.py --arch r50 @S --source "$D/source_train.json" --target "$D/target_train_tags.json" `
    --target-labels teacher --tag-filter --teacher-thresh 0.3 --burn-in 10 --iters 20 --batch 2 --out "$R/b3"
Py train.py --arch r50 @S --source "$D/source_train.json" --target-labels pseudo `
    --pseudo "$R/pseudo_filter_recover.json" --iters 20 --batch 2 `
    --eval "$D/target_test.json" --eval-every 20 --out "$R/ours"
Py evaluate.py --ckpt "$R/ours/best.pth" --data "$D/target_test.json" --workers 0
New-Item -ItemType Directory -Force "$R/samples" | Out-Null
Copy-Item "$D/images/test_000.png", "$D/images/test_001.png" "$R/samples/"
Py demo.py --ckpt "$R/b0/last.pth" "$R/ours/best.pth" --names "Source-only" "Ours" `
    --input "$R/samples" --out "$R/demo" --thresh 0.05 --gt "$D/target_test.json"
Write-Host "SMOKE TEST PASSED"
