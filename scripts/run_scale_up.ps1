param(
    [string]$PilotRunId = 'scaleup_30node_pilot_v1',
    [string]$DeclaredRunId = 'scaleup_30node_declared_v1'
)
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\congestion-lab.exe pipeline scale-up `
    --experiment configs/experiments/scale_up_30node_pilot.yaml `
    --run-id $PilotRunId
& .\.venv\Scripts\congestion-lab.exe pipeline scale-up `
    --experiment configs/experiments/scale_up_30node.yaml `
    --run-id $DeclaredRunId
