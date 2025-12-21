"""
Build script for creating Memory Game installer.
This script uses PyInstaller to create a standalone executable.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required tools are installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("ERROR: PyInstaller is not installed.")
        print("Install it with: pip install pyinstaller")
        return False
    
    return True

def clean_build_directories():
    """Remove previous build artifacts."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)
    
    # Clean .spec file artifacts
    spec_files = list(Path('.').glob('*.spec'))
    for spec in spec_files:
        if spec.name != 'memory_game_installer.spec':
            print(f"Removing old spec file: {spec}")
            spec.unlink()

def build_executable():
    """Build the executable using PyInstaller."""
    spec_file = Path('memory_game_installer.spec')
    
    if not spec_file.exists():
        print(f"ERROR: Spec file not found: {spec_file}")
        return False
    
    print(f"Building executable using {spec_file}...")
    print("This may take several minutes...")
    
    try:
        result = subprocess.run(
            ['pyinstaller', '--clean', str(spec_file)],
            check=True,
            cwd=Path(__file__).parent
        )
        print("Build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Build failed: {e}")
        return False
    except FileNotFoundError:
        print("ERROR: PyInstaller not found in PATH.")
        print("Make sure PyInstaller is installed: pip install pyinstaller")
        return False

def create_installer_package():
    """Create a distributable package with all necessary files."""
    dist_dir = Path('dist')
    if not dist_dir.exists():
        print("ERROR: dist directory not found. Build may have failed.")
        return False
    
    # Find the executable
    exe_files = list(dist_dir.glob('MemoryGame.exe'))
    if not exe_files:
        print("ERROR: Executable not found in dist directory.")
        return False
    
    exe_file = exe_files[0]
    print(f"Found executable: {exe_file}")
    
    # Create installer package directory
    package_dir = Path('installer_package')
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # Copy executable
    print("Creating installer package...")
    shutil.copy2(exe_file, package_dir / 'MemoryGame.exe')
    
    # Create README for installer
    readme_content = """Memory Game with Eye Tracking - Installation Package

INSTALLATION:
1. Extract all files to a folder (e.g., C:\\Program Files\\MemoryGame)
2. Run MemoryGame.exe to start the application
3. On first run, you will be prompted to calibrate eye tracking

REQUIREMENTS:
- Windows 10 or later
- Webcam/camera for eye tracking features
- The application will work without a camera, but eye tracking features will be disabled

DATA FILES:
The application will create the following files in the same directory:
- game_history.json - Your game statistics and leaderboard
- gaze_data_*.csv - Eye tracking data from games
- click_log.csv - Click tracking data

UNINSTALLATION:
Simply delete the installation folder.

For support or issues, please refer to the project documentation.
"""
    
    with open(package_dir / 'README.txt', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\nInstaller package created in: {package_dir.absolute()}")
    print(f"Executable size: {exe_file.stat().st_size / (1024*1024):.1f} MB")
    
    return True

def main():
    """Main build process."""
    print("=" * 60)
    print("Memory Game Installer Builder")
    print("=" * 60)
    print()
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Clean previous builds
    clean_build_directories()
    
    # Build executable
    if not build_executable():
        sys.exit(1)
    
    # Create installer package
    if not create_installer_package():
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Build completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Test the executable in dist/MemoryGame.exe")
    print("2. Package the installer_package folder for distribution")
    print("3. Consider creating a ZIP file or using an installer tool like Inno Setup")

if __name__ == '__main__':
    main()

