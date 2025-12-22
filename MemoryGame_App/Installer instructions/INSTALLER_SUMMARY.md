# Memory Game Installer - Summary

## Overview

A complete installer system has been created for the Memory Game application. The installer packages the application into a standalone executable that includes all dependencies, images, and required files.

## Files Created

### Installer Build Files
1. **`memory_game_installer.spec`** - PyInstaller specification file that defines how to package the application
2. **`build_installer.py`** - Python script that automates the build process
3. **`build_installer.bat`** - Windows batch script for easy building
4. **`build_installer.sh`** - Linux/Mac shell script for easy building

### Application Updates
5. **`app_data_paths.py`** - New utility module that manages file paths for both development and installed versions
6. **Updated `MemoryGame_v2.py`** - Modified to use proper data directories
7. **Updated `eye-detection-final/eye_detector.py`** - Updated to handle bundled paths

### Documentation
8. **`INSTALLER_README.md`** - Comprehensive build and distribution guide
9. **`QUICK_START_INSTALLER.md`** - Quick reference for building
10. **`INSTALLER_SUMMARY.md`** - This file

## Key Features

### File Management
- **Bundled Files**: All application code, images, and models are included in the executable
- **User Data**: Runtime data (game history, gaze logs, click logs) are stored in user-specific directories:
  - Windows: `%LOCALAPPDATA%\MemoryGame\`
  - Linux/Mac: `~/.local/share/MemoryGame/`
- **Separation**: Application files and user data are kept separate for easy updates and uninstallation

### What's Included in the Executable
- Main application (`MemoryGame_v2.py`)
- Calibration screen (`calibration_screen.py`)
- Gaze data logger (`gaze_data_logger.py`)
- Heatmap viewer (`heatmap_view.py`)
- Gaze localization engine (`GazeLocalization/gaze_localizator.py`)
- Eye detector (`eye-detection-final/eye_detector.py`)
- All card images (`images/*.png`)
- Haar cascade XML file for face detection
- All Python dependencies (PyQt5, OpenCV, scikit-learn, numpy)

### What's Created at Runtime
- `game_history.json` - Game statistics and leaderboard
- `gaze_data/*.csv` - Eye tracking data files
- `click_log.csv` - Click tracking data

## Building the Installer

### Quick Start
**Windows:**
```cmd
build_installer.bat
```

**Linux/Mac:**
```bash
./build_installer.sh
```

### Manual Build
```bash
pip install pyinstaller
pyinstaller --clean memory_game_installer.spec
```

## Distribution

After building, the `installer_package` folder contains:
- `MemoryGame.exe` (or `MemoryGame` on Linux/Mac)
- `README.txt` - User instructions

Simply zip this folder and distribute it. Users can:
1. Extract the ZIP
2. Run the executable
3. Start playing!

## Benefits

1. **No Python Required**: End users don't need Python installed
2. **Single Executable**: Everything is bundled in one file
3. **Proper Data Management**: User data is stored in appropriate system directories
4. **Easy Updates**: Update the executable without losing user data
5. **Clean Uninstall**: Delete the executable folder (user data remains if desired)

## Testing Checklist

Before distributing, test:
- [ ] Application starts without errors
- [ ] All card images load correctly
- [ ] Eye tracking calibration works (if camera available)
- [ ] Games can be played successfully
- [ ] Game history is saved and displayed
- [ ] Gaze data is logged correctly
- [ ] Heatmap visualization works
- [ ] Data files are created in the correct location
- [ ] Application works on a clean system (without Python)

## Troubleshooting

### Build Issues
- Ensure all dependencies are installed: `pip install -r requirements.txt pyinstaller`
- Check that all paths in the spec file are correct
- Verify all required files exist (images, XML files, etc.)

### Runtime Issues
- Check that images are bundled correctly
- Verify Haar cascade XML file is included
- Ensure write permissions in user data directory
- Check application logs for specific errors

## Next Steps

1. **Test the build**: Run `build_installer.bat` (or `.sh`) and test the executable
2. **Customize**: Add an icon, change the name, etc. (see `INSTALLER_README.md`)
3. **Create installer package**: Use Inno Setup, NSIS, or WiX for a professional installer
4. **Distribute**: Share the installer package with users

## Support

For detailed information, see:
- `INSTALLER_README.md` - Full documentation
- `QUICK_START_INSTALLER.md` - Quick reference
- PyInstaller docs: https://pyinstaller.org/

