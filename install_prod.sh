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
SSH_SOURCE_DIR="../ssh"
echo "Checking for SSH keys in $SSH_SOURCE_DIR..."
if [ -d "$SSH_SOURCE_DIR" ]; then
    echo "Found directory: $SSH_SOURCE_DIR."
    if [ -f "$SSH_SOURCE_DIR/id_ed25519" ] && [ -f "$SSH_SOURCE_DIR/id_ed25519.pub" ]; then
        echo "Found id_ed25519 key pair in $SSH_SOURCE_DIR. Copying to $HOME/.ssh/..."
        # Ensure target directory exists and has correct permissions
        mkdir -p $HOME/.ssh
        chmod 700 $HOME/.ssh
        # Copy keys
        cp "$SSH_SOURCE_DIR/id_ed25519" $HOME/.ssh/id_ed25519
        cp "$SSH_SOURCE_DIR/id_ed25519.pub" $HOME/.ssh/id_ed25519.pub
        # Set permissions on copied keys
        chmod 600 $HOME/.ssh/id_ed25519
        chmod 644 $HOME/.ssh/id_ed25519.pub
        echo "SSH keys copied from $SSH_SOURCE_DIR and permissions set."
    else
        echo "$SSH_SOURCE_DIR directory exists, but id_ed25519 key pair not found. Skipping copy."
    fi
else
    echo "Directory $SSH_SOURCE_DIR not found. Skipping SSH key copy."
fi
# --- End SSH Key Copy ---

echo "Installing project dependencies into the virtual environment..."
uv pip install .
