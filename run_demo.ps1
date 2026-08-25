# PowerShell script to launch the Streamlit demo UI from the repository root
# Usage: .\run_demo.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$demoDir = Join-Path $scriptDir "demo"
$venvActivate = Join-Path $scriptDir "demo\.venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "Activated virtual environment."
}

# Ensure dependencies are installed (idempotent)
python -m pip install -r "${demoDir}\requirements-demo.txt" --quiet

# Run the Streamlit app
streamlit run "${demoDir}\app.py" --server.port 8501
