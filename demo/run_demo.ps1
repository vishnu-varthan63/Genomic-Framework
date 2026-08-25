# PowerShell script to launch the Streamlit demo UI
# Usage: .\run_demo.ps1
# It activates a virtual environment if present, otherwise runs directly.

$venvPath = "${PSScriptRoot}\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "Activated virtual environment."
}

# Ensure required packages are installed (idempotent)
python -m pip install -r "${PSScriptRoot}\requirements-demo.txt" --quiet

# Run the Streamlit app
streamlit run "${PSScriptRoot}\app.py" --server.port 8501
