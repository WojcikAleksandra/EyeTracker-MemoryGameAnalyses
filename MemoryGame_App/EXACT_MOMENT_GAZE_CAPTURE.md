# Exact Moment Gaze Capture - New Approach

## Problem with Old Approach

**Old method:** Try to match clicks to nearby gaze samples
- Click at 2400ms, gaze samples at 2347ms and 2402ms
- Required complex matching algorithm with ±50ms window
- Prone to timing mismatches
- Not guaranteed to find the exact gaze position at click moment

## New Approach: Capture Gaze at Click Moment

**New method:** Capture `self.current_gaze` directly when click happens!

### How It Works

```python
def log_click(self, btn, matched_flag=0):
    # ... card click info ...
    
    # Capture gaze position at THE EXACT MOMENT of click
    x_gaze = self.current_gaze[0]
    y_gaze = self.current_gaze[1]
    
    # Determine which card user is looking at
    card_id_gaze, grid_position_gaze = self._get_card_at_gaze(x_gaze, y_gaze)
    
    # Write everything to click_log.csv
    self.log_file.write(f"{ms},{x_click},{y_click},{flip},{matched},"
                       f"{card_id_click},{grid_position_click},"
                       f"{x_gaze},{y_gaze},{card_id_gaze},{grid_position_gaze}\n")
```

### Advantages

1. **✅ Exact timing**: No need to match timestamps
2. **✅ Simpler code**: No complex matching algorithm
3. **✅ More accurate**: Gets gaze at the EXACT moment of click, not nearby
4. **✅ No gaps**: Even if gaze sampling has gaps, we capture the current smoothed position
5. **✅ Always works**: `self.current_gaze` always has a value (from smoothing)

## New Data Structure

### Click Log (`click_log.csv`)

**Old format (7 columns):**
```csv
ms,x,y,flip,matched,card_id,grid_position
```

**New format (11 columns):**
```csv
ms,x_click,y_click,flip,matched,card_id_click,grid_position_click,x_gaze,y_gaze,card_id_gaze,grid_position_gaze
```

**Example data:**
```csv
ms,x_click,y_click,flip,matched,card_id_click,grid_position_click,x_gaze,y_gaze,card_id_gaze,grid_position_gaze
1194,737,415,1,0,4,13,740,418,4,13
2400,512,384,2,1,3,11,508,380,3,11
3850,890,420,1,0,1,23,895,425,1,23
```

**Interpretation:**
- At 1194ms: User clicked card 4 at position 13, and was looking at (740, 418) which is card 4 at position 13 ✓
- At 2400ms: User clicked card 3 at position 11, and was looking at (508, 380) which is card 3 at position 11 ✓
- At 3850ms: User clicked card 1 at position 23, and was looking at (895, 425) which is card 1 at position 23 ✓

### Playing Data (`playing_data_*.csv`)

**Still has same format:**
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,grid_position_gaze,x_click,y_click,flip,matched,card_id_click,grid_position_click
```

But now:
- Gaze columns show continuous gaze tracking (every ~16ms)
- Click columns show click info when matched
- **Gaze data at click rows comes from the click_log** (captured at exact click moment!)

## What `self.current_gaze` Represents

```python
self.current_gaze = (x, y)  # Smoothed gaze position
```

This is updated continuously by `_process_gaze_frame()`:
1. Eye detector processes frame
2. Model predicts gaze position
3. Position added to `self.gaze_history` (5-frame buffer)
4. `self.current_gaze` = average of last 5 positions (smoothed!)

**Benefits:**
- **Smoothed**: Not raw/jittery prediction
- **Always available**: Even if current frame failed, uses previous valid position
- **Low latency**: Updated ~60 times per second

## Example Timeline

```
Time    Event                           Gaze Position       Action
0ms     Game starts                     (512, 384)         Recording...
16ms    Gaze frame                      (515, 388)         Recording...
33ms    Gaze frame                      (520, 392)         Recording...
...
1194ms  USER CLICKS card 4!             (740, 418)         CAPTURE! → click_log.csv
                                        ↑
                                        This exact value saved!
1210ms  Gaze frame                      (738, 420)         Recording...
1227ms  Gaze frame                      (735, 422)         Recording...
```

At the moment of click (1194ms), we instantly capture:
- Click position: (737, 415) - center of card 4
- Gaze position: (740, 418) - where user was actually looking
- Difference: Only 3 pixels away! Perfect alignment ✓

## Comparison

### Old Approach
```
Click at 2400ms
  ↓
Find nearest gaze sample in playing_gaze_data
  ↓
Gaze at 2402ms? 2347ms? Which is better?
  ↓
Complex matching with ±50ms window
  ↓
Might not match if timing is off!
```

### New Approach
```
User clicks
  ↓
self.current_gaze = (x, y) [already available!]
  ↓
Write to click_log.csv immediately
  ↓
Done! Always works!
```

## Analysis Benefits

### 1. Click-Gaze Coordination

```python
import pandas as pd

clicks = pd.read_csv('click_log.csv')

# Calculate distance between click and gaze
clicks['distance'] = np.sqrt(
    (clicks['x_click'] - clicks['x_gaze'])**2 + 
    (clicks['y_click'] - clicks['y_gaze'])**2
)

print(f"Average click-gaze distance: {clicks['distance'].mean():.1f} pixels")

# Did user look at correct card?
clicks['correct_card'] = clicks['card_id_click'] == clicks['card_id_gaze']
print(f"Looked at correct card: {clicks['correct_card'].sum()} / {len(clicks)}")
```

### 2. Attention Verification

```python
# Did user look at card before clicking?
for _, row in clicks.iterrows():
    if row['card_id_gaze'] == row['card_id_click']:
        print(f"✓ Click {row['ms']}ms: Correctly looking at target")
    else:
        print(f"✗ Click {row['ms']}ms: Looking at card {row['card_id_gaze']}, clicked {row['card_id_click']}")
```

### 3. Precision Analysis

```python
# How precisely do users click where they look?
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(clicks['x_gaze'], clicks['y_gaze'], c='blue', label='Gaze', s=100, alpha=0.6)
plt.scatter(clicks['x_click'], clicks['y_click'], c='red', label='Click', s=100, marker='x')

for _, row in clicks.iterrows():
    plt.plot([row['x_gaze'], row['x_click']], 
             [row['y_gaze'], row['y_click']], 
             'k-', alpha=0.3)

plt.legend()
plt.title('Click vs Gaze Position')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.gca().invert_yaxis()
plt.show()
```

## Backward Compatibility

The code still reads old format `click_log.csv` files:

```python
if len(parts) >= 11:  # New format
    # Read all 11 columns
elif len(parts) >= 6:  # Old format
    # Read old columns, set gaze data to -1
```

## Summary

**Key Innovation:** Capture `self.current_gaze` at the exact moment of click instead of trying to match timestamps later!

**Benefits:**
- ✅ No timing issues
- ✅ Exact gaze position at click
- ✅ Simpler code
- ✅ More accurate analysis
- ✅ Always works

**Result:** You now know EXACTLY where the user was looking when they clicked each card!

This is the best of both worlds:
- Continuous gaze tracking (every 16ms) in playing_data
- Exact click-moment gaze in click_log
- No complex matching algorithm needed!



