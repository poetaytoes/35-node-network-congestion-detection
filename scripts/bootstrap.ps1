$ErrorActionPreference = 'Stop'
$python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
if ($python -eq 'py') {
    & py -3.11 -m venv .venv
} else {
    & python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e '.[dev,external,explain]'
& .\.venv\Scripts\python.exe -m pytest

