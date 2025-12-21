# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Memory Game with Eye Tracking

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

# Get the base directory (where the spec file is located)
base_dir = Path(SPECPATH)

# Block cipher for bytecode encryption (set to None to disable)
block_cipher = None

# Data files to include
datas = [
    # Images directory
    (str(base_dir / 'images'), 'images'),
    # Haar cascade XML file - copy to eye-detection-final subdirectory
    (str(base_dir.parent / 'eye-detection-final' / 'haarcascade_frontalface_default.xml'), 
     'eye-detection-final'),
]

# Hidden imports (modules that PyInstaller might miss)
hiddenimports = [
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'cv2',
    'numpy',
    'sklearn',
    'sklearn.linear_model',
    'sklearn.ensemble',
    'sklearn.svm',
    'sklearn.neural_network',
    'calibration_screen',
    'gaze_data_logger',
    'heatmap_view',
    'gaze_localizator',
    'eye_detector',
]

# Collect all Python files from subdirectories
a = Analysis(
    ['MemoryGame_v2.py'],
    pathex=[
        str(base_dir),
        str(base_dir.parent / 'GazeLocalization'),
        str(base_dir.parent / 'eye-detection-final'),
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MemoryGame',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)

