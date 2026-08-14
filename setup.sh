#!/usr/bin/env bash
set -e

echo "=== TrustKnee Environment Setup ==="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Create virtualenv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# Activate and install dependencies
source .venv/bin/activate
echo "Installing project and development dependencies..."
pip install -r requirements.txt

# Register pre-commit and commit-msg hooks
echo "Registering git hooks..."
pre-commit install --hook-type pre-commit --hook-type commit-msg

echo "=== Setup complete! Hooks are active. ==="
