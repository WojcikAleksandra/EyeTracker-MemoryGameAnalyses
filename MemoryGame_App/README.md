# Memory Game with Eye Tracking

This is a memory card game integrated with eye tracking capabilities for research and analysis purposes.

## Features

- Memory card matching game with customizable difficulty (8, 10, or 12 cards)
- Integrated eye tracking with automatic calibration
- Real-time gaze data collection during gameplay
- Combined output of game events and gaze data for analysis

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have a webcam connected to your computer.

3. Make sure the `eye-detection-final` directory is in the parent directory with the `eye_detector.py` module.

## Running the Game

```bash
python MemoryGame_v2.py
```

## First-Time Setup

When you first click "Play":

1. You'll be prompted to calibrate the eye tracker
2. A calibration screen will appear with 20 red points
3. Look at each point and click on it when ready
4. After calibration completes, the game will start

## Gameplay

1. **Memorization Phase**: Cards are shown face-up for 5 seconds
2. **Playing Phase**: Click on cards to find matching pairs
3. The game tracks:
   - Your clicks and matches
   - Where you're looking at all times
   - Which cards you're viewing

## Output Data

After each game, a timestamped CSV file is generated: `game_data_YYYYMMDD_HHMMSS.csv`

### CSV Columns:

- `ms`: Milliseconds since game start
- `x_gaze`, `y_gaze`: Gaze coordinates on screen
- `valid`: Whether gaze detection was valid (1) or not (0)
- `card_id_gaze`: Which card user is looking at (-1 if none)
- `x_click`, `y_click`: Click coordinates (when a click occurs)
- `flip`: Which card in pair was clicked (1 or 2)
- `matched`: Whether the pair matched (1) or not (0)
- `card_id_click`: ID of the card that was clicked

## Recalibration

You can recalibrate the eye tracker at any time:
- Go to **Settings** → **Recalibrate eye-tracking**

## Troubleshooting

### Eye tracking calibration fails
- Ensure good lighting conditions
- Make sure your face is clearly visible to the webcam
- Keep your head relatively still during calibration
- Try recalibrating from the Settings menu

### Camera not detected
- Check that your webcam is connected and not in use by another application
- Grant camera permissions if prompted by your OS

### Game runs slowly
- Close other applications using the camera
- Ensure you have sufficient system resources available

## Technical Details

- Eye tracking uses the Ridge regression model for gaze estimation
- Gaze is sampled continuously at ~60 FPS during gameplay
- Feature extraction uses 10x10 pixel patches from each eye region
- Gaze position is smoothed using a 5-frame moving average


