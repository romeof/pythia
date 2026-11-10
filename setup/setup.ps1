# How to run (Windows):
# 1) Open PowerShell and cd to your repo root, e.g.:
#    C:\work\pythia
# 2) Execute the setup script:
#    powershell -ExecutionPolicy Bypass -File setup\setup.ps1
# 3) Activate the environment for your current PowerShell session:
#    . .\.venv\Scripts\Activate.ps1
#    (if activation is blocked, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
# 4) Deactivate when done:  deactivate
# Notes:
# Requires Python 3.11.x, 64-bit: tries the py launcher (py -3.11), then python / python3 on PATH, and accepts an
# interpreter only if it reports 3.11. Any failing step stops the script with a non-zero exit code.

$ErrorActionPreference = "Stop"

# Resolve paths
$setupDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path $setupDir -Parent
Set-Location $repoRoot

$venvName = ".venv"
$venvPath = Join-Path $repoRoot $venvName

# Native commands do not throw on a non-zero exit code; check it after every call.
function Assert-LastExit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)." }
}

function Get-PythonMinor([string]$Command, [string[]]$BaseArgs) {
    # Returns "3.11"-style version text ("3.11-32bit" for a 32-bit build), or '' when the command does not run.
    # Function-scoped Continue: under Stop, PS 5.1 turns any stderr line of a working interpreter into an error.
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Command @BaseArgs -c "import sys; print('%d.%d' % sys.version_info[:2] + ('' if sys.maxsize > 2**32 else '-32bit'))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ([string]$out).Trim() }
    } catch {}
    return ''
}

function Get-PythonCommand {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += [pscustomobject]@{ Command = "py"; BaseArgs = @("-3.11") }
    }
    foreach ($name in @("python", "python3")) {
        if (Get-Command $name -ErrorAction SilentlyContinue) {
            $candidates += [pscustomobject]@{ Command = $name; BaseArgs = @() }
        }
    }
    foreach ($c in $candidates) {
        if ((Get-PythonMinor $c.Command $c.BaseArgs) -eq '3.11') { return $c }
    }
    throw "Python 3.11 (64-bit) is required (py -3.11, python or python3 must be 3.11.x). Install Python 3.11 64-bit from python.org and rerun."
}

Write-Host "Using project root:" $repoRoot

# Determine Python command
$py = Get-PythonCommand
Write-Host "Using Python: $($py.Command) $($py.BaseArgs -join ' ')" -ForegroundColor Cyan

# Create virtual environment if it doesn't exist
if (Test-Path $venvPath) {
    Write-Host "Virtual environment '$venvName' already exists. Reusing." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment '$venvName'..." -ForegroundColor Cyan
    & $py.Command @($py.BaseArgs) -m venv $venvPath
    Assert-LastExit "Creating the virtual environment"
}

# Use the venv's python directly (no activation required inside script)
$venvPython = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Expected venv python at '$venvPython' not found. Delete '$venvPath' and rerun."
}
$venvVersion = Get-PythonMinor $venvPython @()
if ($venvVersion -ne '3.11') {
    throw "'$venvPath' uses Python '$venvVersion', not 3.11. Delete '$venvPath' and rerun."
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
Assert-LastExit "Upgrading pip"

# Install requirements
$requirements = Join-Path $setupDir "requirements.txt"
if (-not (Test-Path $requirements)) {
    throw "requirements.txt not found at '$requirements'"
}

Write-Host "Installing packages from setup/requirements.txt..." -ForegroundColor Cyan
& $venvPython -m pip install -r $requirements
Assert-LastExit "Installing setup/requirements.txt"

# Register a named Jupyter kernel for this environment so it appears in kernel pickers (optional).
Write-Host "Registering Jupyter kernel 'pythia-venv'..." -ForegroundColor Cyan
& $venvPython -m ipykernel install --user --name "pythia-venv" --display-name "Python (pythia .venv)" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Kernel registration skipped (non-fatal, exit code $LASTEXITCODE)." -ForegroundColor Yellow
}

Write-Host "Environment setup complete." -ForegroundColor Green
Write-Host "To activate the environment in PowerShell for this session, run:" -ForegroundColor Yellow
Write-Host ". $(Join-Path $venvPath 'Scripts/Activate.ps1')" -ForegroundColor Yellow
Write-Host "To deactivate later, run:  deactivate" -ForegroundColor Yellow
Write-Host "Notebook kernel label: Python (pythia .venv)" -ForegroundColor Yellow
