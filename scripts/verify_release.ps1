param([Parameter(Mandatory = $true)][string]$RunId)
$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m mypy src
& .\.venv\Scripts\python.exe -m pytest --cov=congestion_lab --cov-report=term-missing
& .\.venv\Scripts\python.exe -m build
& .\.venv\Scripts\congestion-lab.exe verify run --run-dir (Join-Path 'runs' $RunId)

