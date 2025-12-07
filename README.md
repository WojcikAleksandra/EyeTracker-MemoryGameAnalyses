# EyeTracker-MemoryGameAnalyses

Project focusing on eye tracking as a tool for analyzing visual behavior of players in a Memory Game.

## Overview

This project combines a memory card matching game with real-time eye tracking to collect data about where players look during gameplay. The data can be used to analyze visual attention patterns, memory strategies, and player behavior.

## Components

### 1. Eye Detection Module (`eye-detection-final/`)

Contains the eye tracking implementation:
- `eye_detector.py`: Core eye detection using OpenCV and Haar Cascades
- `gaze_localization_demo3_v1.py`: Standalone gaze tracking demo with calibration
- Feature extraction and gaze estimation using machine learning models

### 2. Memory Game Application (`MemoryGame_App/`)

PyQt5-based memory card game with integrated eye tracking:
- **MemoryGame_v2.py**: Main application with eye tracking integration
- Automatic calibration before first game
- Real-time gaze data collection during gameplay
- Combined output of game events and gaze positions

## Installation

1. Navigate to the MemoryGame_App directory:
```bash
cd MemoryGame_App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python test_integration.py
```

## Running the Game

```bash
cd MemoryGame_App
python MemoryGame_v2.py
```

## How It Works

### First-Time Calibration

1. User clicks "Play" for the first time
2. Calibration screen appears with 20 red points arranged in a grid
3. User looks at each point and clicks on it
4. System collects eye images and trains a gaze estimation model
5. Game starts after successful calibration

### During Gameplay

1. **Memorization Phase**: Cards shown face-up for 5 seconds
2. **Playing Phase**: User clicks cards to find matches
3. **Eye Tracking**: System continuously records:
   - Gaze position (x, y coordinates)
   - Which card user is looking at
   - Validation status of eye detection
4. **Click Tracking**: System records:
   - Click positions and timestamps
   - Which cards were clicked
   - Match results

### Data Output

After each game, a CSV file is generated: `game_data_YYYYMMDD_HHMMSS.csv`

**Columns:**
- `ms`: Milliseconds since game start
- `x_gaze`, `y_gaze`: Screen coordinates of gaze
- `valid`: Eye detection quality (1=valid, 0=invalid)
- `card_id_gaze`: ID of card being looked at (-1 if none)
- `x_click`, `y_click`: Click coordinates when click occurs
- `flip`: Which card in pair (1 or 2, -1 if no click)
- `matched`: Whether pair matched (1=yes, 0=no, -1 if no click)
- `card_id_click`: ID of clicked card (-1 if no click)

## Data Analysis

The generated CSV files can be analyzed to understand:
- **Visual search patterns**: Where do players look when searching for matches?
- **Memory strategies**: Do players look at previous card locations?
- **Attention distribution**: How much time is spent on each card?
- **Click prediction**: Can we predict which card will be clicked based on gaze?

## Recalibration

Eye tracking can be recalibrated at any time:
- **Menu** → **Settings** → **Recalibrate eye-tracking**

This is useful if:
- Lighting conditions change
- User position shifts
- Detection accuracy decreases

## Technical Details

### Eye Tracking Pipeline

1. **Face Detection**: Haar Cascade face detection
2. **Eye Detection**: BT.709 grayscale conversion and edge-based binarization
3. **Iris Localization**: Percentile-based thresholding in YCbCr space
4. **Feature Extraction**: 10x10 pixel patches from each eye
5. **Gaze Estimation**: Ridge regression model (trained during calibration)
6. **Smoothing**: 5-frame moving average for stable gaze position

### Performance

- Gaze sampling rate: ~60 FPS
- Calibration points: 20 (5x4 grid)
- Calibration samples per point: Variable (1 second window)
- Model: Ridge regression with α=1.0

## Troubleshooting

### Camera Issues
- Ensure webcam is connected and not in use
- Check camera permissions in OS settings
- Try different USB ports

### Calibration Failures
- Improve lighting (avoid backlighting)
- Position face clearly in front of camera
- Keep head still during calibration
- Retry calibration from Settings menu

### Poor Tracking Accuracy
- Recalibrate in current lighting conditions
- Maintain consistent head position
- Ensure face is well-lit and visible

## Future Enhancements

Potential improvements:
- Multiple difficulty levels analysis
- Heatmap visualization of gaze patterns
- Real-time gaze cursor overlay
- Export to additional formats (JSON, HDF5)
- Multi-session analysis tools
- Machine learning for click prediction

## Credits

Developed as part of an engineering thesis project on eye tracking applications in cognitive game analysis.
