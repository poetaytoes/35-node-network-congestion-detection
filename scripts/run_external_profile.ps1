param(
    [string]$RunId = 'external_mawi_profile_v1',
    [string]$Model = ''
)
$ErrorActionPreference = 'Stop'
$arguments = @(
    'external', 'replay',
    '--experiment', 'configs/experiments/pilot.yaml',
    '--manifest', 'data/external/external_sources.json',
    '--profile', 'data/external/profiles/mawi_202608011400_profile.json',
    '--sample', 'data/external/samples/mawi_202608011400_privacy_safe_sample.csv',
    '--run-id', $RunId
)
if ($Model) {
    $arguments += @('--model', $Model)
}
& .\.venv\Scripts\congestion-lab.exe @arguments
