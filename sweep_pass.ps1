<#
.SYNOPSIS
    One-shot benchmark sweep: preflight the machine, measure all configurations,
    regenerate charts.

.DESCRIPTION
    Preflight exists because the two defects this harness has shipped were both
    invisible at the point of measurement and only found by audit afterwards.
    Checking the machine's state before a 30-minute sweep is cheaper than
    discovering afterwards that the GPU was driving a display or sharing the
    board with a browser.

    Nothing here silently corrects the environment — it reports, and lets you
    decide. The harness itself still verifies clock state per run and refuses
    to publish what it cannot measure.

.PARAMETER Runs
    Clean runs required per configuration. Default 5.

.PARAMETER Tag
    Log subdirectory under results/logs. Default is a UTC timestamp.

.PARAMETER Force
    Continue past preflight warnings instead of stopping.

.EXAMPLE
    .\sweep_pass.ps1
    .\sweep_pass.ps1 -Runs 7 -Tag pass-D
#>
[CmdletBinding()]
param(
    [int]$Runs = 5,
    [string]$Tag = (Get-Date -Format "yyyyMMdd-HHmm"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Virtualenv not found at $python" }

$logDir = "results/logs/$Tag"
New-Item -ItemType Directory -Force $logDir | Out-Null

# --- Preflight ------------------------------------------------------------

function Get-GpuField([string]$field) {
    (nvidia-smi --query-gpu=$field --format=csv,noheader,nounits).Trim()
}

Write-Host "`n=== Preflight ===" -ForegroundColor Cyan
$warnings = @()

# 1. Is the dGPU driving a display? Its framebuffer and the desktop compositor
#    occupy VRAM that the model then cannot have. On a hybrid laptop, switching
#    to Optimus/Hybrid routes the panel through the iGPU and hands it back.
$displayActive = Get-GpuField "display_active"
if ($displayActive -eq "Enabled") {
    $warnings += "dGPU is driving a display — switch to Optimus/Hybrid GPU mode and reboot to free its framebuffer."
    Write-Host "  [warn] display_active = Enabled (dGPU owns the panel)" -ForegroundColor Yellow
} else {
    Write-Host "  [ok]   dGPU is not driving a display" -ForegroundColor Green
}

# 2. Free VRAM. Reported against the largest model present so the number means
#    something rather than being an abstract total.
$free = [double](Get-GpuField "memory.free")
$total = [double](Get-GpuField "memory.total")
$used = $total - $free
$largest = (Get-ChildItem models -Filter *.gguf -ErrorAction SilentlyContinue |
            Sort-Object Length -Descending | Select-Object -First 1)
Write-Host ("  [info] VRAM {0:N0} MiB free of {1:N0} ({2:N0} held)" -f $free, $total, $used)
if ($largest) {
    $largestMb = [math]::Round($largest.Length / 1MB)
    Write-Host ("  [info] largest model {0} at {1:N0} MB" -f $largest.Name, $largestMb)
}
if ($used -gt 400) {
    $warnings += "$([math]::Round($used)) MiB of VRAM is already held — close browsers, VS Code, Teams, Discord."
    Write-Host "  [warn] more than 400 MiB already held" -ForegroundColor Yellow
}

# 3. Power profile. A conservative profile leaves the memory clock oscillating
#    between P-states mid-run, which the harness will reject run by run.
$powerLimit = [double]((nvidia-smi -q -d POWER | Select-String "Current Power Limit" |
    Select-Object -First 1) -replace '[^\d\.]', '')
$maxPower = [double]((nvidia-smi -q -d POWER | Select-String "Max Power Limit" |
    Select-Object -First 1) -replace '[^\d\.]', '')
if ($powerLimit -gt 0 -and $maxPower -gt 0) {
    $pct = $powerLimit / $maxPower
    Write-Host ("  [info] GPU power limit {0:N0} W of {1:N0} W max" -f $powerLimit, $maxPower)
    if ($pct -lt 0.85) {
        $warnings += "GPU power limit is $([math]::Round($pct * 100))% of maximum — set the vendor utility to its highest performance profile."
        Write-Host "  [warn] power limit well below maximum" -ForegroundColor Yellow
    }
}

# 4. Is anything else using the GPU right now?
$util = [double](Get-GpuField "utilization.gpu")
if ($util -gt 10) {
    $warnings += "GPU is $util% busy before the sweep started — another process is competing."
    Write-Host "  [warn] GPU already $util% busy" -ForegroundColor Yellow
} else {
    Write-Host "  [ok]   GPU idle ($util% utilization)" -ForegroundColor Green
}

if ($warnings.Count -gt 0) {
    Write-Host "`n$($warnings.Count) preflight warning(s):" -ForegroundColor Yellow
    $warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    if (-not $Force) {
        Write-Host "`nFix these and re-run, or pass -Force to proceed anyway." -ForegroundColor Yellow
        Write-Host "Measurements taken now may be rejected by the clock gate or refused for VRAM.`n"
        exit 1
    }
    Write-Host "`n-Force given — proceeding despite warnings.`n" -ForegroundColor Yellow
} else {
    Write-Host "`nPreflight clean.`n" -ForegroundColor Green
}

# --- Sweep ----------------------------------------------------------------

$configs = @(
    @{f = "llama2";  q = "Q4_K_M"; m = "models/llama-2-7b-chat.Q4_K_M.gguf" },
    @{f = "llama2";  q = "Q5_K_M"; m = "models/llama-2-7b-chat.Q5_K_M.gguf" },
    @{f = "llama2";  q = "Q8_0";   m = "models/llama-2-7b-chat.Q8_0.gguf" },
    @{f = "mistral"; q = "Q4_K_M"; m = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf" },
    @{f = "mistral"; q = "Q5_K_M"; m = "models/mistral-7b-instruct-v0.1.Q5_K_M.gguf" },
    @{f = "mistral"; q = "Q8_0";   m = "models/mistral-7b-instruct-v0.1.Q8_0.gguf" },
    @{f = "qwen2.5"; q = "Q4_K_M"; m = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" },
    @{f = "qwen2.5"; q = "Q5_K_M"; m = "models/Qwen2.5-7B-Instruct-Q5_K_M.gguf" },
    @{f = "qwen2.5"; q = "Q8_0";   m = "models/Qwen2.5-7B-Instruct-Q8_0.gguf" }
)

$started = Get-Date
Write-Host "=== Sweep: $($configs.Count) configurations, $Runs clean runs each ===" -ForegroundColor Cyan
Write-Host "Logs: $logDir`n"

foreach ($c in $configs) {
    if (-not (Test-Path $c.m)) {
        Write-Host "[skip] $($c.f) $($c.q): model file not found" -ForegroundColor Yellow
        continue
    }
    Write-Host "--- $($c.f) $($c.q)  $(Get-Date -Format HH:mm:ss) ---" -ForegroundColor Cyan
    & $python benchmark.py --model $c.m --quant-type $c.q --model-family $c.f `
        --n-gpu-layers 99 --n-runs $Runs *>&1 |
        Tee-Object -FilePath "$logDir/$($c.f)-$($c.q).log" |
        Select-String -Pattern "accept|reject|Warmed up|warn|skip|Geometry|Decode t|Timing source"
}

# --- Charts and summary ---------------------------------------------------

Write-Host "`n=== Regenerating charts ===" -ForegroundColor Cyan
& $python compare_quants.py
& $python compare_quants.py --group-by model_family

$elapsed = (Get-Date) - $started
Write-Host ("`n=== Complete in {0:hh\:mm\:ss} ===" -f $elapsed) -ForegroundColor Green
Write-Host "Rows by timing_source:"
Import-Csv results/benchmark_results.csv |
    Group-Object timing_source |
    Sort-Object Count -Descending |
    ForEach-Object { Write-Host ("  {0,-28} {1}" -f $_.Name, $_.Count) }
Write-Host "`nOnly perf_counters rows are published. Anything else was refused or flagged.`n"
