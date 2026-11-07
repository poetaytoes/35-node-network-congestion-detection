# 35-Node Network Congestion Detection

[Tiếng Việt](README.vi.md)

Reproducible centralized/distributed congestion prediction on a 35-node Vietnam network topology.
The pipeline generates paired traffic, builds causal 5-second telemetry windows, trains baseline and
ML models, evaluates a sealed test split, and exports PNG/SVG reports.

## Quick start (Windows, no virtual-environment activation)

Requirements: Git and Python 3.11+.

```powershell
git clone https://github.com/poetaytoes/35-node-network-congestion-detection.git
cd 35-node-network-congestion-detection
py -3.11 -m pip install --user -e .
py -3.11 -m congestion_lab.cli config validate --config configs/experiments/core_35node.yaml
```

Run one seed:

```powershell
.\scripts\run_vietnam35.ps1 -RunId vietnam35_v1
```

Run the five-seed experiment used for reporting:

```powershell
.\scripts\run_vietnam35_multiseed.ps1 -RunPrefix vietnam35_multiseed_v1
```

Use a new `RunId`/`RunPrefix` for every experiment. Outputs are written to `runs/` and `results/`;
both directories are intentionally excluded from Git.

## What is included

- `src/congestion_lab/`: simulator, feature pipeline, models, policies, evaluation, and reporting.
- `configs/`: 35-node topology and experiment YAML files.
- `scripts/`: single-seed, multi-seed, verification, and report commands.
- `tests/`: unit, contract, property, integration, and regression tests.

## Optional development checks

```powershell
py -3.11 -m pip install --user -e ".[dev]"
py -3.11 -m pytest
```

License: MIT.
