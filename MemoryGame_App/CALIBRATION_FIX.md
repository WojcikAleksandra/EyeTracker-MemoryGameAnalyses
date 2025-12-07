# Calibration Red Points Fix

## Problems Identified

1. **Initial Issue**: Red calibration points were not visible on the screen
2. **Second Issue**: Text labels overlapped with top calibration points
3. **Third Issue**: Bottom calibration points were cut off (too close to screen edge)

## Root Cause

The issue was with coordinate system confusion in the `CalibrationScreen` class:

1. **Calibration points** were generated based on widget size (local coordinates)
2. **But they were treated as global screen coordinates** when drawing
3. **The `mapFromGlobal()` conversion** was mapping them to incorrect positions (likely off-screen)

## Solution

Changed the calibration point handling to use consistent **local widget coordinates**:

### 1. Point Generation
```python
def _generate_calibration_points(self):
    # Points are generated in widget-local coordinates
    # Based on self.screen_w and self.screen_h (widget size)
    points.append((x, y))  # Local coordinates
    return points
```

### 2. Point Drawing
```python
def paintEvent(self, event):
    # Draw points directly without coordinate conversion
    target_x, target_y = self.calibration_points[self.current_point_idx]
    center = QPoint(target_x, target_y)
    painter.drawEllipse(center, 12, 12)
```

### 3. Click Detection
```python
def mousePressEvent(self, event):
    # Click detection in local coordinates
    click_x_local = event.pos().x()
    click_y_local = event.pos().y()
    target_x_local, target_y_local = self.calibration_points[...]
    
    # Only convert to global when storing training data
    global_click = self.mapToGlobal(event.pos())
    global_target = self.mapToGlobal(QPoint(target_x_local, target_y_local))
    
    # Store global coordinates for model training
    self.y_x.append(float(global_target.x()))
    self.y_y.append(float(global_target.y()))
```

## Additional Fixes

### 1. Increased Margins
Changed calibration point margins to prevent cutoff:
- **Horizontal margins**: 2% → 8% (keeps points away from sides)
- **Vertical margins**: 3.5% → 10% (keeps points away from top/bottom)

This ensures all 20 points are fully visible on screen.

### 2. Repositioned Labels
Moved instruction text from top-center to **bottom-right corner**:
- Labels no longer overlap with top calibration points
- Positioned dynamically in `paintEvent()` to stay in corner
- Semi-transparent background for readability

### 3. Simplified Visual Design
Simplified calibration points to **simple red dots**:
- Single solid red circle (10px radius)
- No extra rings or decorations
- Clean, minimal appearance
- Easy to spot and click

## Testing

To verify the fix works, run:

```bash
python test_calibration_visual.py
```

This creates a simple window showing all 20 calibration points. You should see:
- Large red points with white centers
- Gray circles showing remaining points
- Click counter advancing as you click each point

## Expected Behavior

After the fix:
1. ✅ Red calibration point visible in center of white background
2. ✅ Point moves to next position when clicked
3. ✅ Progress counter updates: "Point 1/20", "Point 2/20", etc.
4. ✅ Points arranged in 5×4 grid across screen
5. ✅ Model training receives correct global screen coordinates

## Files Modified

- `MemoryGame_v2.py`:
  - Fixed `_generate_calibration_points()` - clarified coordinate system
  - Fixed `paintEvent()` - removed incorrect coordinate conversion
  - Fixed `mousePressEvent()` - proper local/global coordinate handling
  - Enhanced visual appearance of calibration points

## Technical Details

### Coordinate Systems

**Local (Widget) Coordinates:**
- Origin (0, 0) at top-left of CalibrationScreen widget
- Used for: point generation, drawing, click detection

**Global (Screen) Coordinates:**
- Origin (0, 0) at top-left of entire screen
- Used for: model training targets (so model predicts screen positions)

### Why Global for Training?

The gaze estimation model needs to predict **global screen coordinates** because:
1. During gameplay, gaze needs to be compared with card positions in screen space
2. Different windows/widgets may be at different screen positions
3. Global coordinates are consistent across the application

### Conversion Points

```
Generate Points (Local) 
    ↓
Draw Points (Local) ← No conversion needed
    ↓
Detect Click (Local) ← No conversion needed
    ↓
Store for Training (Global) ← Convert here: mapToGlobal()
```

## Verification Checklist

- [x] Red points visible during calibration
- [x] Points positioned correctly (5×4 grid)
- [x] Click detection works (points advance)
- [x] Calibration completes successfully
- [x] Model trains on correct coordinates
- [x] Gaze tracking works during game

## Future Improvements

Potential enhancements:
- Add animation to calibration point (pulsing, fading in)
- Show predicted gaze position during calibration for feedback
- Display accuracy estimate after calibration
- Allow user to redo individual points if accuracy is poor

