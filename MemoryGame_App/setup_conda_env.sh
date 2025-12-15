#!/bin/bash
# ============================================================
# Memory Game with Eye Tracking - Conda Environment Setup
# ============================================================
# This script creates a conda environment with all dependencies
# for running the Memory Game with eye tracking functionality.
#
# Usage: Run this script from the MemoryGame_App directory
#        chmod +x setup_conda_env.sh
#        ./setup_conda_env.sh
# ============================================================

ENV_NAME="memory-game-env"
PYTHON_VERSION="3.10"

echo ""
echo "============================================================"
echo " Memory Game - Conda Environment Setup"
echo "============================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Initialize conda for bash if needed
eval "$(conda shell.bash hook)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " OVERWRITE
    if [[ "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

echo ""
echo "Creating conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}"
echo ""

# Create the environment
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment."
    exit 1
fi

echo ""
echo "Activating environment and installing packages..."
echo ""

# Activate and install packages
conda activate ${ENV_NAME}

# Install packages via pip (better compatibility for PyQt5)
pip install PyQt5>=5.15.0
pip install opencv-python>=4.5.0
pip install scikit-learn>=1.0.0
pip install numpy>=1.20.0

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install some packages."
    exit 1
fi

echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo "Environment '${ENV_NAME}' has been created successfully."
echo ""
echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To run the Memory Game:"
echo "    cd MemoryGame_App"
echo "    python MemoryGame_v2.py"
echo ""
echo "============================================================"


