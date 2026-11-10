#!/bin/bash
# Linux setup: creates .venv in the pythia folder with Python 3.11 and installs setup/requirements.txt.
# On Windows use setup/setup.ps1 instead (this script refuses to run under Git Bash / MSYS / Cygwin).

# Get the absolute path of this script's directory
SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Change to the parent directory (pythia folder)
cd "$SCRIPT_DIR/.."
echo "Current directory: $(pwd)"

# Stop at the first failing command from here on.
set -e
set -o pipefail

fail() {
    echo "Error: $*" >&2
    exit 1
}

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        fail "setup.sh is for Linux. On Windows run: powershell -ExecutionPolicy Bypass -File setup\\setup.ps1" ;;
esac

# Find a Python 3.11 interpreter (the pins reproduce the reference environment, validated on 3.11 only)
PYTHON_BIN=""
for cand in python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
        if [ "$ver" = "3.11" ]; then
            PYTHON_BIN=$(command -v "$cand")
            break
        fi
    fi
done
[ -n "$PYTHON_BIN" ] || fail "Python 3.11 is required (python3.11, python3 or python on PATH must be 3.11.x)."
echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

# Create the virtual environment (.venv) in the pythia folder if it doesn't exist
if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv || { rm -rf .venv; fail "could not create .venv (Debian/Ubuntu: sudo apt install python3.11-venv)"; }
    echo "Virtual environment created at .venv"
else
    echo "Virtual environment .venv already exists."
fi

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || fail "$VENV_PY not found: .venv is not a Linux virtual environment. Delete .venv and rerun."
venv_ver=$("$VENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$venv_ver" = "3.11" ] || fail ".venv uses Python $venv_ver, not 3.11. Delete .venv and rerun."
"$VENV_PY" -m pip --version >/dev/null 2>&1 || fail ".venv has no pip (incomplete venv). Delete .venv and rerun."

# Upgrade pip and install the pinned requirements into the venv (never into another environment)
"$VENV_PY" -m pip install -U pip || fail "pip upgrade failed"
"$VENV_PY" -m pip install -r setup/requirements.txt || fail "installing setup/requirements.txt failed"

# Register a named Jupyter kernel so this venv is easy to pick in notebooks (optional)
"$VENV_PY" -m ipykernel install --user --name pythia-venv --display-name "Python (pythia .venv)" \
    || echo "Warning: Jupyter kernel registration failed (non-fatal)." >&2

echo "Setup complete! Virtual environment is ready and requirements have been installed."
echo "To activate: source .venv/bin/activate"
echo "Notebook kernel label: Python (pythia .venv)"
