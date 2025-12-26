"""
Utility module for managing application data paths.
Handles user data directories for installed applications.
"""

import os
import sys
from pathlib import Path
import re
from datetime import datetime

def get_app_data_dir():
    """
    Get the application data directory.
    For installed apps, uses user's AppData/Local.
    For development, uses the application directory.
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        # Use user's local app data directory
        if sys.platform == 'win32':
            app_data = Path(os.getenv('LOCALAPPDATA', ''))
            if not app_data:
                app_data = Path.home() / 'AppData' / 'Local'
            app_dir = app_data / 'MemoryGame'
        else:
            # Linux/Mac
            app_dir = Path.home() / '.local' / 'share' / 'MemoryGame'
        
        # Create directory if it doesn't exist
        app_dir.mkdir(parents=True, exist_ok=True)
        return str(app_dir)
    else:
        # Running as script (development mode)
        # Use application directory
        return str(Path(__file__).parent)

def get_images_dir():
    """
    Get the images directory path.
    For installed apps, images are bundled with the executable.
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        # Images are in the same directory as the executable
        base_path = Path(sys.executable).parent
        images_path = base_path / 'images'
        return str(images_path)
    else:
        # Running as script (development mode)
        return str(Path(__file__).parent / 'images')

def get_haar_cascade_path():
    """
    Get the path to the Haar cascade XML file.
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = Path(sys.executable).parent
        cascade_path = base_path / 'eye-detection-final' / 'haarcascade_frontalface_default.xml'
        return str(cascade_path)
    else:
        # Running as script (development mode)
        base_dir = Path(__file__).parent.parent
        cascade_path = base_dir / 'eye-detection-final' / 'haarcascade_frontalface_default.xml'
        return str(cascade_path)

def get_game_history_path():
    """Get the path to the game history JSON file."""
    app_data_dir = get_app_data_dir()
    return str(Path(app_data_dir) / 'game_history_data' / 'game_history.json')

def get_gaze_data_dir():
    """Get the directory for gaze data CSV files."""
    # Save directly to MemoryGame_App directory (same as app_data_dir in development)
    app_data_dir = get_app_data_dir()
    # Create directory if it doesn't exist
    #Path(app_data_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(app_data_dir) / 'gaze_data')

def get_click_log_path():
    """Get the path to the click log CSV file."""
    app_data_dir = get_app_data_dir()
    return str(Path(app_data_dir) / 'game_history_data' / 'click_log.csv')

def get_latest_archived_gaze_file_path():
    """ Get the path to the latest archived gaze data CSV file."""
    archived_dir = os.path.join(get_gaze_data_dir(), "archived")
    if not os.path.isdir(archived_dir):
        return None

    FILENAME_RE = re.compile(
        r"gaze_data_(\d{8})_(\d{6})_game(\d+)\.csv"
    )

    candidates = []
    for fname in os.listdir(archived_dir):
        match = FILENAME_RE.match(fname)
        if not match:
            continue

        date_part, time_part, game_num = match.groups()
        timestamp = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        candidates.append((
                    timestamp,
                    int(game_num),
                    fname
                ))

    if not candidates:
        return None

    # Sort: newest timestamp, then highest game number
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return os.path.join(archived_dir, candidates[0][2])

