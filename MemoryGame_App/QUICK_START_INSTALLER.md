# Quick Start - Building the Installer

## Windows

1. Open Command Prompt or PowerShell in the `MemoryGame_App` folder
2. Run: `build_installer.bat`
3. Wait for the build to complete (may take 5-10 minutes)
4. Find the executable in `installer_package/MemoryGame.exe`

## Linux/Mac

1. Open Terminal in the `MemoryGame_App` folder
2. Run: `./build_installer.sh` (or `bash build_installer.sh`)
3. Wait for the build to complete
4. Find the executable in `installer_package/MemoryGame`

## Manual Build

If the scripts don't work, run directly:

```bash
pip install pyinstaller
pyinstaller --clean memory_game_installer.spec
```

## What Gets Created

- **Executable**: `dist/MemoryGame.exe` (Windows) or `dist/MemoryGame` (Linux/Mac)
- **Installer Package**: `installer_package/` folder with the executable and README

## Testing

1. Run the executable from the `dist` or `installer_package` folder
2. Test that:
   - The game starts
   - Images load correctly
   - Eye tracking calibration works (if camera available)
   - Game data is saved properly

## Distribution

Zip the `installer_package` folder and distribute it. Users can:
1. Extract the ZIP
2. Run `MemoryGame.exe`
3. Play the game!

**Note**: The first time users run the app, they'll be prompted to calibrate eye tracking (optional).

