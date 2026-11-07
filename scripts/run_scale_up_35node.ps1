param (
    [Alias("run-id")]
    [string]$RunId = "scale_up_35node_final"
)

# Catch the common Unix-style parameter typo in PowerShell
if ($RunId -eq "--run-id" -and $args.Count -gt 0) {
    $RunId = $args[0]
}

$ErrorActionPreference = "Stop"

Write-Host "=== Starting 35-Node Topology Run ===" -ForegroundColor Cyan
Write-Host "Run ID: $RunId" -ForegroundColor Cyan

# Execute the scale-up pipeline
& .\.venv\Scripts\congestion-lab.exe pipeline full `
    --experiment configs/experiments/scale_up_35node.yaml `
    --run-id $RunId `
    --publish-root results

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== 35-Node Run Completed Successfully ===" -ForegroundColor Green
} else {
    Write-Host "=== Run Failed ===" -ForegroundColor Red
}