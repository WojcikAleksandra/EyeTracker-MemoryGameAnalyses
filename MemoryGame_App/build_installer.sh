#!/bin/bash
# Linux/Mac shell script to build the Memory Game installer

echo "========================================"
echo "Memory Game Installer Builder"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller is not installed. Installing..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install PyInstaller"
        exit 1
    fi
fi

# Run the build script
echo "Starting build process..."
python3 build_installer.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Build failed! Check the error messages above."
    exit 1
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "The installer package is in the 'installer_package' folder"
echo ""

