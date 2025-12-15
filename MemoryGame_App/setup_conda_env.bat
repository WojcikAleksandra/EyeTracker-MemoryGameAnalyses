@echo off
REM ============================================================
REM Memory Game with Eye Tracking - Conda Environment Setup
REM ============================================================
REM This script creates a conda environment with all dependencies
REM for running the Memory Game with eye tracking functionality.
REM
REM Usage: Run this script from the MemoryGame_App directory
REM ============================================================

set ENV_NAME=memory-game-env
set PYTHON_VERSION=3.10

echo.
echo ============================================================
echo  Memory Game - Conda Environment Setup
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Conda is not installed or not in PATH.
    echo Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr /C:"%ENV_NAME%" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Environment '%ENV_NAME%' already exists.
    set /p OVERWRITE="Do you want to remove and recreate it? (y/n): "
    if /i "%OVERWRITE%"=="y" (
        echo Removing existing environment...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo Keeping existing environment. Exiting.
        pause
        exit /b 0
    )
)

echo.
echo Creating conda environment: %ENV_NAME% with Python %PYTHON_VERSION%
echo.

REM Create the environment
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

echo.
echo Activating environment and installing packages...
echo.

REM Activate and install packages
call conda activate %ENV_NAME%

REM Install packages via pip (better compatibility for PyQt5)
pip install PyQt5>=5.15.0
pip install opencv-python>=4.5.0
pip install scikit-learn>=1.0.0
pip install numpy>=1.20.0

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install some packages.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo Environment '%ENV_NAME%' has been created successfully.
echo.
echo To activate the environment, run:
echo     conda activate %ENV_NAME%
echo.
echo To run the Memory Game:
echo     cd MemoryGame_App
echo     python MemoryGame_v2.py
echo.
echo ============================================================
pause


