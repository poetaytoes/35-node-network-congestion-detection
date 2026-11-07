param([string]$RunId = 'core_locked_v1')
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\congestion-lab.exe pipeline full `
    --experiment configs/experiments/core_18_episodes.yaml `
    --run-id $RunId `
    --publish-root results
