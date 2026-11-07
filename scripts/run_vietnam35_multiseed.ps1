param(
    [int[]]$Seeds = @(20260814, 20260815, 20260816, 20260817, 20260818),
    [string]$RunPrefix = 'vietnam35_multiseed_v1',
    [switch]$CompactGrid
)

$ErrorActionPreference = 'Stop'
$RunsRoot = 'runs'
$PublishRoot = 'results'
$Python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
$PythonVersion = if ($Python -eq 'py') { @('-3.11') } else { @() }

if ($Seeds.Count -lt 3 -or $Seeds.Count -ne @($Seeds | Select-Object -Unique).Count) {
    throw 'Provide at least three distinct root seeds.'
}

foreach ($Seed in $Seeds) {
    $RunId = "${RunPrefix}_seed${Seed}"
    $RunDir = Join-Path $RunsRoot $RunId
    $ReportDir = Join-Path $PublishRoot "${RunId}_report"
    $RequiredScienceArtifacts = @(
        (Join-Path $RunDir 'evaluation/binary_metrics_locked.csv'),
        (Join-Path $RunDir 'evaluation/locked_predictions.parquet'),
        (Join-Path $RunDir 'system/c0_c1_d0_d1_metrics.csv'),
        (Join-Path $RunDir 'robustness/cross_topology.csv'),
        (Join-Path $RunDir 'models/xgboost.joblib')
    )
    $ScienceComplete = ($RequiredScienceArtifacts | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0

    if (Test-Path -LiteralPath $RunDir) {
        if (-not $ScienceComplete) {
            throw "Existing run is incomplete before locked evaluation and cannot be resumed safely: $RunDir. Use a new RunPrefix."
        }
        Write-Host "Reusing completed scientific artifacts for seed $Seed from $RunDir"
    }
    else {
    $Arguments = @(
        'pipeline', 'full',
        '--experiment', 'configs/experiments/core_35node.yaml',
        '--run-id', $RunId,
        '--root-seed', $Seed,
        '--publish-root', 'results'
    )
    if ($CompactGrid) {
        $Arguments += '--compact-grid'
    }
    & $Python @PythonVersion -m congestion_lab.cli @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    }

    if (-not (Test-Path -LiteralPath (Join-Path $ReportDir 'RESULTS_README.md'))) {
        if (Test-Path -LiteralPath $ReportDir) {
            throw "Report recovery namespace already exists but is incomplete: $ReportDir"
        }
        & $Python @PythonVersion scripts/regenerate_report_from_run.py `
            --run-dir $RunDir `
            --output-dir $ReportDir
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    else {
        Write-Host "Reusing refreshed PNG/SVG report for seed $Seed from $ReportDir"
    }
}

$SeedList = $Seeds -join ','
$SummaryDir = Join-Path $PublishRoot "${RunPrefix}_summary"
if (-not (Test-Path -LiteralPath (Join-Path $SummaryDir 'README.md'))) {
    if (Test-Path -LiteralPath $SummaryDir) {
        throw "Multi-seed summary namespace already exists but is incomplete: $SummaryDir"
    }
    & $Python @PythonVersion scripts/summarize_multiseed.py `
        --run-prefix $RunPrefix `
        --seeds $SeedList `
        --runs-root $RunsRoot `
        --publish-root $PublishRoot

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
else {
    Write-Host "Reusing existing multi-seed summary from $SummaryDir"
}
