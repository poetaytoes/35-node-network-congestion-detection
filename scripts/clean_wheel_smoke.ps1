param(
    [string]$Python = 'python',
    [string]$Venv = '.clean_venv311'
)
$ErrorActionPreference = 'Stop'
& $Python -m venv $Venv
$venvPython = Join-Path $Venv 'Scripts\python.exe'
& $venvPython -m pip install --upgrade pip
$wheel = Get-ChildItem dist\congestion_lab-*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $wheel) { throw 'Build the wheel first with python -m build.' }
& $venvPython -m pip install $wheel.FullName
& $venvPython -c "import congestion_lab; print(congestion_lab.__version__)"
& (Join-Path $Venv 'Scripts\congestion-lab.exe') config validate `
    --config configs/experiments/pilot.yaml
