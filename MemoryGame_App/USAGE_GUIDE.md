# Memory Game with Eye Tracking - Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
cd MemoryGame_App
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python test_integration.py
```

This will verify that:
- All required packages are installed
- Eye detector can be initialized
- Camera is accessible
- Game components load correctly

### 3. Run the Game

```bash
python MemoryGame_v2.py
```

## First Game Session

### Step 1: Home Screen
- Select number of cards (8, 10, or 12)
- Click "Play" button

### Step 2: Calibration Prompt
On first play, you'll see a dialog:
```
Before starting the game, we need to calibrate the eye tracker.

You will see 20 red points on the screen. Look at each point 
and click on it when ready.

Make sure you're in a well-lit environment and the camera 
can see your face.
```

Click "OK" to proceed or "Cancel" to abort.

### Step 3: Calibration Process
- 20 red points will appear one at a time
- **Look directly at the red point**
- **Click on it** when you're looking at it
- Progress indicator shows: "Point 1/20", "Point 2/20", etc.
- Takes about 1-2 minutes total

**Tips for successful calibration:**
- Sit comfortably in front of the computer
- Keep your head relatively still
- Ensure good, even lighting on your face
- Avoid backlight (windows behind you)
- Click accurately on each point

### Step 4: Calibration Success
After all points are completed:
```
Eye tracking calibration successful!

Starting game...
```

### Step 5: Game Starts
- **3-2-1 countdown**
- **Memorization phase**: Cards shown face-up for 5 seconds
- **Playing phase**: Find all matching pairs

## Playing the Game

### Memorization Phase (5 seconds)
- All cards are visible face-up
- Try to remember as many as possible
- Eye tracking is **not active** during this phase
- Countdown shows remaining time

### Playing Phase
- Cards flip face-down
- Click cards to flip them
- Eye tracking is **active** - your gaze is being recorded
- Find matching pairs:
  - If cards match: They stay face-up
  - If cards don't match: They flip back after 0.8 seconds
- Timer and move counter track your performance

### Game Completion
- When all pairs are found: "You found all pairs! Great job!"
- Statistics screen shows time and moves
- Data is automatically saved to `game_data_YYYYMMDD_HHMMSS.csv`

## Menu Options

### Home
Return to main menu (aborts current game if playing)

### Statistics
View past game statistics (feature placeholder)

### Settings → Recalibrate Eye-Tracking
- Re-run the calibration process
- Useful when:
  - You moved to a different position
  - Lighting changed significantly
  - Tracking accuracy seems poor
  - Starting a new session

## Understanding the Output Data

### File Location
Output files are saved in the `MemoryGame_App` directory:
```
game_data_20231215_143022.csv
game_data_20231215_144530.csv
...
```

### CSV Format

**Header:**
```
ms,x_gaze,y_gaze,valid,card_id_gaze,x_click,y_click,flip,matched,card_id_click
```

**Example rows:**
```
100,512,384,1,3,-1,-1,-1,-1,-1
150,520,390,1,3,-1,-1,-1,-1,-1
768,737,415,1,4,737,415,1,0,4
800,740,420,1,4,-1,-1,-1,-1,-1
```

**Column Descriptions:**

1. **ms**: Milliseconds since game start (playing phase)
2. **x_gaze**: Screen X coordinate where user is looking
3. **y_gaze**: Screen Y coordinate where user is looking
4. **valid**: Eye detection quality
   - `1` = Valid detection
   - `0` = Poor/no detection
5. **card_id_gaze**: Which card user is looking at
   - `1-6` = Card number (for 12-card game)
   - `-1` = Not looking at any card
6. **x_click**: X coordinate of click (when click occurs)
   - `-1` = No click at this timestamp
7. **y_click**: Y coordinate of click (when click occurs)
   - `-1` = No click at this timestamp
8. **flip**: Which card in pair was clicked
   - `1` = First card of pair
   - `2` = Second card of pair
   - `-1` = No click
9. **matched**: Whether the pair matched
   - `1` = Pair matched
   - `0` = Pair didn't match
   - `-1` = No click
10. **card_id_click**: ID of card that was clicked
    - `1-6` = Card number
    - `-1` = No click

### Data Analysis Example

**Python snippet to load and explore data:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('game_data_20231215_143022.csv')

# Filter valid gaze samples
valid_gaze = df[df['valid'] == 1]

# Plot gaze trajectory
plt.figure(figsize=(12, 8))
plt.scatter(valid_gaze['x_gaze'], valid_gaze['y_gaze'], 
            c=valid_gaze['ms'], cmap='viridis', alpha=0.5, s=10)
plt.colorbar(label='Time (ms)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Gaze Trajectory During Game')
plt.gca().invert_yaxis()  # Screen coordinates
plt.show()

# Find clicks
clicks = df[df['flip'] != -1]
print(f"Total clicks: {len(clicks)}")
print(f"Matches: {clicks['matched'].sum()}")
print(f"Mismatches: {(clicks['matched'] == 0).sum()}")

# Time spent looking at each card
for card_id in range(1, 7):
    time_ms = len(df[df['card_id_gaze'] == card_id]) * 16  # ~60 FPS
    print(f"Card {card_id}: {time_ms/1000:.2f} seconds")
```

## Troubleshooting

### "Failed to initialize eye tracking"
**Cause**: Camera not accessible or OpenCV issues
**Solution**: 
- Check camera is connected
- Close other applications using camera
- Reinstall opencv-python: `pip install --upgrade opencv-python`

### "Calibration Failed - Not enough valid frames"
**Cause**: Eye detector cannot consistently detect your eyes
**Solution**:
- Improve lighting
- Remove glasses (if causing reflections)
- Position yourself closer to camera
- Ensure face is fully visible
- Remove obstructions (hair covering eyes, etc.)

### Game window doesn't resize properly
**Expected**: Window is locked during gameplay to maintain consistent coordinates
**Note**: Window will unlock after returning to home screen

### Gaze tracking seems inaccurate
**Solutions**:
- Recalibrate from Settings menu
- Ensure you haven't moved significantly since calibration
- Check lighting hasn't changed
- Maintain similar head position as during calibration

### CSV file is empty or missing
**Cause**: Game wasn't completed or save failed
**Check**:
- Play a complete game (find all pairs)
- Check file permissions in MemoryGame_App directory
- Look for error messages in console

## Best Practices

### For Research/Data Collection

1. **Consistent Environment**
   - Same lighting conditions
   - Same seating position
   - Same screen distance (~50-70cm)

2. **Calibration**
   - Calibrate once per session
   - Recalibrate if user moves significantly
   - Takes 1-2 minutes but improves accuracy

3. **Participant Instructions**
   - "Sit comfortably and naturally"
   - "Try not to move your head too much"
   - "Look naturally at the cards"
   - "Play at your own pace"

4. **Data Management**
   - Files are timestamped automatically
   - Rename files with participant IDs if needed
   - Back up data regularly

### For Best Tracking Accuracy

- **Lighting**: Bright, even, front-facing light
- **Position**: Face the camera directly, 50-70cm away
- **Movement**: Natural head movement is OK, but avoid large shifts
- **Calibration**: More careful calibration = better tracking
- **Environment**: Minimal distractions, stable setup

## Technical Specifications

- **Sampling Rate**: ~60 Hz (60 samples per second)
- **Gaze Latency**: ~16-33 ms
- **Spatial Resolution**: Screen pixel coordinates
- **Calibration Time**: 1-2 minutes (20 points)
- **Model Type**: Ridge Regression (L2 regularization)
- **Feature Dimension**: 200 features (2 × 10×10 eye patches)
- **Smoothing**: 5-frame moving average

## System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.7+
- **Camera**: Webcam (640×480 minimum, 720p or better recommended)
- **Screen**: Any resolution (tested on 1920×1080)
- **RAM**: 4GB minimum, 8GB recommended
- **Processor**: Modern multi-core CPU

## Support

For issues or questions:
1. Check this guide first
2. Run `test_integration.py` to diagnose issues
3. Check console output for error messages
4. Verify all dependencies are installed correctly



