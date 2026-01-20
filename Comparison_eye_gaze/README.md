# Gaze Localization Comparison App

Compares two gaze localization algorithms:
- **Appearance-based** (Production) - uses raw eye patch pixels
- **Geometric** (Demo2_v4) - uses relative pupil positions and face geometry

Both algorithms share the same calibration and use the shared `eye-detection-final` and `GazeLocalization` modules from the parent directory.

## Directory Structure

```
EyeTracker-MemoryGameAnalyses/
    Comparison_eye_gaze/
        main.py             # Main comparison application
        metrics.py          # Metrics computation script
        README.md           # This file
    eye-detection-final/    # Shared eye detection module
    GazeLocalization/       # Shared gaze localization module
    MemoryGame_App/         # Memory game with gaze tracking
```

## Installation

Uses the same environment as the main project:
```bash
cd EyeTracker-MemoryGameAnalyses
conda env create -f MemoryGame_App/environment.yml
conda activate memory_game
```

Or install dependencies manually:
```bash
pip install opencv-python numpy scikit-learn PyQt5
```

## Usage

Run the comparison app:
```bash
cd Comparison_eye_gaze
python main.py
```

For developer mode (shows camera debug window):
```bash
python main.py --dev
```

## How it works

1. **Calibration Phase**: Look at each red dot and click on it (20 points total). Both algorithms share this calibration data.

2. **Test Phase**: After calibration, green dots appear randomly. Look at each dot and click on it. The app records:
   - Actual dot pixel position (ground truth)
   - Appearance-based algorithm prediction
   - Geometric algorithm prediction
   - Error distance for both methods

3. **Results**: Saved to CSV with columns:
   - `dot_x`, `dot_y` - Actual dot position
   - `appearance_x`, `appearance_y`, `appearance_error` - Appearance algorithm
   - `geometric_x`, `geometric_y`, `geometric_error` - Geometric algorithm

## Analyzing Results

Use the metrics script to compute detailed statistics:
```bash
python metrics.py comparison_results_YYYYMMDD_HHMMSS.csv
```

Output includes: mean, std, median, IQR, p90, p95, max error for each algorithm.

## Requirements

- Python 3.8+
- Webcam
- Good lighting conditions
- Face clearly visible to camera

