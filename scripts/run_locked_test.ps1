param([Parameter(Mandatory = $true)][string]$RunId)
$ErrorActionPreference = 'Stop'
$run = Join-Path 'runs' $RunId
if (-not (Test-Path -LiteralPath (Join-Path $run 'seals\locked_test_seal.json'))) {
    throw "Locked-test seal missing for $RunId"
}
& .\.venv\Scripts\congestion-lab.exe verify run --run-dir $run

