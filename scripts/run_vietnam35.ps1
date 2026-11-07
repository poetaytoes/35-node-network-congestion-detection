param([string]$RunId = 'vietnam35_v1')

$ErrorActionPreference = 'Stop'
$Python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
$PythonVersion = if ($Python -eq 'py') { @('-3.11') } else { @() }

& $Python @PythonVersion -m congestion_lab.cli pipeline full `
    --experiment configs/experiments/core_35node.yaml `
    --run-id $RunId `
    --publish-root results

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
