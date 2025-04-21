#!/bin/bash

# --- Production Dependency Installation Script ---
# Installs dependencies typically needed in the production/runtime environment
# for the Panim project, beyond the core Python packages listed in pyproject.toml's
# main dependencies (though bitsandbytes might be added there eventually).


set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Installing Production Dependencies ---"

Install system packages using apt-get
echo "Updating package list and installing tmux and vim..."
apt-get update && apt-get install -y tmux vim

echo "--- Production dependencies installed successfully ---"

# install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source environment variables to make uv available in the current script session
# Using the path provided by the user: /root/.local/bin
echo "Updating PATH to include uv installation directory (/root/.local/bin)..."
export PATH="/root/.local/bin:$PATH"

# Verify uv installation (optional)
if ! command -v uv &> /dev/null
then
    echo "ERROR: uv command could not be found after installation and PATH update."
    echo "Installation path used: /root/.local/bin"
    echo "Current PATH: $PATH"
    echo "Please check the installation output and update the PATH manually if needed."
    exit 1
fi
echo "uv command found."

echo "Installing bitsandbytes using uv..."
uv venv
source .venv/bin/activate
uv pip install bitsandbytes
