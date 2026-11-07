$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\congestion-lab.exe simulate paired `
    --experiment configs/experiments/pilot.yaml `
    --output-dir runs/pilot_manual
& .\.venv\Scripts\congestion-lab.exe dataset build `
    --experiment configs/experiments/pilot.yaml `
    --run-dir runs/pilot_manual

