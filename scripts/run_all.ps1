# One-shot bootstrap for the Windows GPU workstation: installs everything,
# gets/verifies data, benchmarks, smoke-tests, then launches the full queue
# detached. Safe to re-run: every step skips work that is already done.
#
# Usage (from the repo root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#   ... -SkipData      # data already copied into data\malimg + data\big2015
#   ... -Seeds 3       # fewer seeds than the default 5
param(
    [switch]$SkipData,
    [int]$Seeds = 5,
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)
# Continue (not Stop): PS 5.1 turns any native-command stderr (pip warnings, import errors) into a terminating error under Stop. Failures are caught via Check.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" } }

Step "1/8 Find Python"
$py = $null
foreach ($c in @("py -3.11", "py -3.12", "py -3.10", "python")) {
    try { $v = & cmd /c "$c --version 2>&1"; if ($v -match "Python 3\.(1[0-3])") { $py = $c; break } } catch {}
}
if (-not $py) { throw "No Python 3.10-3.13 found. Install it (winget install -e --id Python.Python.3.11), reopen PowerShell, re-run." }
Write-Host "Using: $py"

Step "2/8 Create venv"
if (-not (Test-Path .venv\Scripts\python.exe)) { & cmd /c "$py -m venv .venv"; Check "venv" }
$vpy = ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip; Check "pip upgrade"

Step "3/8 Install CUDA PyTorch (a plain 'pip install torch' on Windows gives a CPU-only build)"
$hasCuda = (& cmd /c ".venv\Scripts\python.exe -c `"import torch; print(torch.cuda.is_available())`" 2>nul") | Select-Object -Last 1
if ($hasCuda -ne "True") {
    & $vpy -m pip install torch torchvision --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) {
        Write-Host "cu128 failed, trying cu126" -ForegroundColor Yellow
        & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126; Check "torch install"
    }
}
& $vpy -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"; Check "CUDA check"

Step "4/8 Install project dependencies + tests"
& $vpy scripts\setup_env.py --dev; Check "setup_env"
& $vpy -m pip install kaggle; Check "kaggle install"
& $vpy -m pytest -q; Check "pytest"

Step "5/8 Data"
if (-not $SkipData) {
    $kj = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
    $have = (Test-Path data\malimg) -and (Test-Path data\big2015)
    if (-not $have) {
        $tok = Join-Path $env:USERPROFILE ".kaggleccess_token"
        if (-not ((Test-Path $kj) -or (Test-Path $tok) -or $env:KAGGLE_API_TOKEN)) {
            throw "No data and no Kaggle credentials ($kj, $tok, or KAGGLE_API_TOKEN). Either copy datasets into data\malimg and data\big2015, or download kaggle.json (Kaggle > Settings > API) into %USERPROFILE%\.kaggle\ and re-run."
        }
        & $vpy scripts\fetch_data.py --dataset both; Check "fetch_data"
    }
}
& $vpy -m lrmc verify-data --dataset malimg --root data\malimg; Check "verify malimg"
& $vpy -m lrmc verify-data --dataset big2015 --root data\big2015; Check "verify big2015"

Step "6/8 Benchmark real GPU"
& $vpy -m lrmc benchmark --device auto --seeds $Seeds; Check "benchmark"

Step "7/8 Smoke test"
& $vpy -m lrmc train configs\rehearsal_mock_malimg.yaml; Check "smoke train"
& $vpy -m lrmc evaluate configs\rehearsal_mock_malimg.yaml; Check "smoke evaluate"

Step "8/8 Launch full queue (detached, survives AnyDesk disconnect)"
powershell -ExecutionPolicy Bypass -File scripts\run_detached.ps1 -Queue experiments/queue.yaml
Write-Host "`nRunning. Check progress:  .venv\Scripts\python.exe -m lrmc status"
Write-Host "When finished:           .venv\Scripts\python.exe -m lrmc aggregate"
