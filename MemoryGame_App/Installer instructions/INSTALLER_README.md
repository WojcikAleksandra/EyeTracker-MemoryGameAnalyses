# Memory Game Installer - Build Instructions

This document explains how to build an installer for the Memory Game application.

## Prerequisites

1. **Python 3.8 or later** - Make sure Python is installed and in your PATH
2. **All dependencies installed** - Run:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

## Building the Installer

### Option 1: Using the Build Script (Recommended)

1. Open a terminal/command prompt in the `MemoryGame_App` directory
2. Run the build script:
   ```bash
   python build_installer.py
   ```

The script will:
- Check for required dependencies
- Clean previous build artifacts
- Build the executable using PyInstaller
- Create an installer package in the `installer_package` folder

### Option 2: Manual Build

1. Run PyInstaller directly:
   ```bash
   pyinstaller --clean memory_game_installer.spec
   ```

2. The executable will be created in the `dist` folder

## Installer Package Contents

After building, the `installer_package` folder will contain:
- `MemoryGame.exe` - The main application executable
- `README.txt` - Installation instructions for end users

## Distribution

### Simple Distribution (ZIP)

1. Zip the entire `installer_package` folder
2. Distribute the ZIP file
3. Users extract and run `MemoryGame.exe`

### Advanced Installer (Optional)

For a more professional installer, consider using:
- **Inno Setup** (Windows) - Free, creates .exe installers
- **NSIS** (Windows) - Free, creates .exe installers
- **WiX Toolset** (Windows) - Creates MSI installers

## File Management

### Application Files (Bundled)
- `MemoryGame_v2.py` - Main application
- `calibration_screen.py` - Eye tracking calibration
- `gaze_data_logger.py` - Data logging
- `heatmap_view.py` - Heatmap visualization
- `app_data_paths.py` - Path management utility
- `GazeLocalization/gaze_localizator.py` - Gaze estimation
- `eye-detection-final/eye_detector.py` - Eye detection
- `eye-detection-final/haarcascade_frontalface_default.xml` - Face detection model
- `images/*.png` - Card images

### User Data Files (Created at Runtime)

When the application runs, it creates data files in:
- **Windows**: `%LOCALAPPDATA%\MemoryGame\`
- **Linux/Mac**: `~/.local/share/MemoryGame/`

Files created:
- `game_history.json` - Game statistics and leaderboard
- `gaze_data/` - Directory containing gaze tracking CSV files
- `click_log.csv` - Click tracking data

**Note**: These files are stored separately from the application, so:
- Uninstalling the app doesn't delete user data
- User data persists across application updates
- Multiple users on the same computer have separate data

## Troubleshooting

### Build Fails

1. **Missing dependencies**: Make sure all packages in `requirements.txt` are installed
2. **PyInstaller not found**: Install with `pip install pyinstaller`
3. **Import errors**: Check that all Python files are in the correct directories

### Runtime Issues

1. **Images not found**: Ensure the `images` folder is bundled correctly (check the spec file)
2. **Haar cascade not found**: Check that `haarcascade_frontalface_default.xml` is included
3. **Data files not saving**: Check write permissions in the user data directory

### Testing

Before distributing:
1. Test the executable on a clean machine (without Python installed)
2. Verify all images load correctly
3. Test eye tracking calibration
4. Verify data files are created in the correct location
5. Test game functionality end-to-end

## Customization

### Adding an Icon

1. Create or obtain an `.ico` file
2. Update `memory_game_installer.spec`:
   ```python
   icon='path/to/your/icon.ico',
   ```

### Changing Executable Name

Update the `name` parameter in `memory_game_installer.spec`:
```python
name='YourAppName',
```

### Including Additional Files

Add to the `datas` list in `memory_game_installer.spec`:
```python
datas = [
    ('path/to/file', 'destination/in/bundle'),
    ...
]
```

## Support

For issues or questions:
1. Check the main project README
2. Review PyInstaller documentation: https://pyinstaller.org/
3. Check application logs for error messages

