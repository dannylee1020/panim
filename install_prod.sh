#!/bin/bash

# --- Production Dependency Installation Script ---
# Installs dependencies typically needed in the production/runtime environment
# for the Panim project, beyond the core Python packages listed in pyproject.toml's
# main dependencies (though bitsandbytes might be added there eventually).


set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Installing Production Dependencies ---"

# install uv
echo "Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Installing bitsandbytes..."
uv pip install bitsandbytes

# 2. Install tmux system package using apt-get
echo "Updating package list and installing tmux..."
apt-get update && apt-get install -y tmux

echo "--- Production dependencies installed successfully ---"
