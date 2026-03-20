#!/usr/bin/env bash
# PAE_katana - Environment Setup Script
# Run this ONCE to set up the environment

set -euo pipefail

echo "======================================"
echo "PAE_katana Environment Setup"
echo "======================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create conda environment
echo "[1/4] Creating conda environment 'pae_katana'..."
conda create -n pae_katana python=3.10 -y

# Activate environment
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pae_katana

# Install dependencies
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install websocietysimulator as editable package
echo "[4/4] Installing websocietysimulator framework..."
cd "$SCRIPT_DIR"
pip install -e .

echo ""
echo "======================================"
echo "Environment setup complete!"
echo ""
echo "IMPORTANT: You need to download the dataset separately."
echo "Dataset should be placed at: $SCRIPT_DIR/dataset/"
echo "Required files:"
echo "  - dataset/user.json   (~1.0 GB)"
echo "  - dataset/item.json   (~1.3 GB)"
echo "  - dataset/review.json (~4.0 GB)"
echo ""
echo "To activate the environment:"
echo "  conda activate pae_katana"
echo "======================================"
