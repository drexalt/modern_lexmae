#!/bin/bash

set -e

UV_BIN="$HOME/.local/bin"

if [ "$0" = "$BASH_SOURCE" ]; then
    echo "Error: Please run this script with 'source ./setup_env.sh' or '. ./setup_env.sh'"
    echo "Running it directly (e.g., './setup_env.sh') won’t update your shell’s PATH."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed, version: $(uv --version)"
fi

if [[ ":$PATH:" != *":$UV_BIN:"* ]]; then
    echo "Adding $UV_BIN to PATH for this session..."
    export PATH="$UV_BIN:$PATH"
else
    echo "$UV_BIN already in PATH"
fi

if ! grep -q "$UV_BIN" ~/.bashrc 2>/dev/null; then
    echo "Adding $UV_BIN to ~/.bashrc..."
    echo "export PATH=\"$UV_BIN:\$PATH\"" >> ~/.bashrc
fi

cd "$(dirname "$0")"

echo "Setting up virtual environment..."
uv venv
source .venv/bin/activate

echo "Installing project dependencies..."
uv sync

echo "Environment setup complete! uv should now be available."
echo "Run 'uv --version' to confirm."

set +e