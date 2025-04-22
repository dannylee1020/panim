#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Installing Production Dependencies ---"

# Install system packages using apt-get
echo "Updating package list and installing tmux and vim..."
apt-get update && apt-get install -y tmux vim

echo "--- Production dependencies installed successfully ---"

# install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# update path
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
# Create virtual environment (default name: .venv)
uv venv
# Activate the virtual environment using the correct path
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Virtual environment activated."

# Install bitsandbytes into the activated environment
uv pip install bitsandbytes

# --- Conditionally Copy SSH Keys ---
echo "Checking for local SSH keys..."
if [ -d "ssh" ]; then
    echo "Found ssh/ directory."
    if [ -f "ssh/id_ed25519" ] && [ -f "ssh/id_ed25519.pub" ]; then
        echo "Found id_ed25519 key pair. Copying to /root/.ssh/..."
        chmod 700 /root/.ssh
        cp ssh/id_ed25519 /root/.ssh/id_ed25519
        cp ssh/id_ed25519.pub /root/.ssh/id_ed25519.pub
        chmod 600 /root/.ssh/id_ed25519
        chmod 644 /root/.ssh/id_ed25519.pub
        echo "SSH keys copied and permissions set."
    else
        echo "ssh/ directory exists, but id_ed25519 key pair not found. Skipping copy."
    fi
else
    echo "ssh/ directory not found. Skipping SSH key copy."
fi
# --- End SSH Key Copy ---

echo "Installing project dependencies into the virtual environment..."
uv pip install .
