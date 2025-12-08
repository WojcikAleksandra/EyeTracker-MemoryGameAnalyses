# Debug Guide - Click Matching Issues

## Your Specific Case

**Problem reported:**
- Click log has click at **2400ms**
- Playing data has gaze at **2347ms** and **2402ms**
- Click at 2400ms should match with gaze at 2402ms (only 2ms difference!)
- But they're not matching

## New Debug Output

When the game finishes, you'll now see detailed console output:

```
DEBUG: Loaded 12 click events from click_log.csv
DEBUG: First click at 1194ms, Last click at 2400ms
DEBUG: Gaze data from 0ms to 5234ms
DEBUG: Matched click at 1194ms with gaze at 1204ms (diff: 10ms)
DEBUG: Matched click at 1850ms with gaze at 1836ms (diff: 14ms)
DEBUG: Matched click at 2400ms with gaze at 2402ms (diff: 2ms)
DEBUG: Matched 12 unique clicks with gaze samples out of 12 total clicks
```

Or if there's a problem:
```
DEBUG: Matched 0 unique clicks with gaze samples out of 12 total clicks
DEBUG: Unmatched clicks at: [1194, 1850, 2400, ...]
```

## Possible Causes

### 1. No Gaze Data at That Time

**Check:** Does `playing_data_*.csv` actually have a row with `ms=2402`?

```bash
# On Windows PowerShell:
Select-String -Pattern "^2402," playing_data_*.csv

# Or open in Excel and filter for ms = 2402
```

**If missing:** Gaze tracking might have paused or eye detection failed at that moment.

**Solution:** The algorithm should still find the nearest match. If gaze samples are at 2347ms and 2419ms (both within 50ms), it should match one of them.

### 2. Gaze Data List is Empty

**Check:** Console should show:
```
DEBUG: Gaze data from Xms to Yms
```

**If you see nothing:** `self.playing_gaze_data` is empty!

**Possible causes:**
- Eye tracking not started during playing phase
- Camera thread not running
- All frames invalid (eye detection failed)

### 3. Click Data Not Loaded Properly

**Check:** Console should show:
```
DEBUG: Loaded 12 click events from click_log.csv
```

**If you see "Loaded 0":** 
- Check that `click_log.csv` exists
- Check file format (should have 7 columns after grid_position was added)
- Check file isn't corrupted

### 4. Timestamp Reference Mismatch (Should be fixed now)

Both clicks and gaze should use `game_start_time`. Verify in code:

```python
# In _start_game_timer():
self.game_start_time = QTime.currentTime()
self.playing_start_time = self.game_start_time  # Same reference!
```

## Manual Verification

### Step 1: Check Click Log

Open `click_log.csv`:
```csv
ms,x,y,flip,matched,card_id,grid_position
1194,737,415,1,0,4,13
1850,890,420,2,1,1,23
2400,512,384,1,0,3,11
...
```

Note the `ms` values for first few clicks.

### Step 2: Check Playing Data

Open `playing_data_*.csv` and look for rows with those `ms` values (±50ms):

```python
import pandas as pd

play = pd.read_csv('playing_data_20231215_143045.csv')
clicks_log = pd.read_csv('click_log.csv')

print("Click timestamps:")
print(clicks_log['ms'].head())

print("\nGaze samples near first click:")
first_click_ms = clicks_log['ms'].iloc[0]
window = play[(play['ms'] >= first_click_ms - 50) & (play['ms'] <= first_click_ms + 50)]
print(window[['ms', 'card_id_gaze', 'grid_position_gaze', 'flip']].head(10))
```

### Step 3: Verify Matching

```python
import pandas as pd

play = pd.read_csv('playing_data_20231215_143045.csv')

# Find rows where clicks matched
matched_rows = play[play['flip'] != -1]

print(f"Total gaze samples: {len(play)}")
print(f"Samples with clicks: {len(matched_rows)}")

if len(matched_rows) > 0:
    print("\nMatched clicks:")
    print(matched_rows[['ms', 'card_id_gaze', 'grid_position_gaze', 
                        'card_id_click', 'grid_position_click', 'flip']].head(10))
else:
    print("\nNo clicks matched!")
```

## Expected Behavior

For your specific case (click at 2400ms):

**What should happen:**
1. Algorithm checks gaze sample at 2347ms: `abs(2400 - 2347) = 53ms` → **Outside 50ms window** ❌
2. Algorithm checks gaze sample at 2402ms: `abs(2400 - 2402) = 2ms` → **Within 50ms window** ✓
3. Click data attached to row with `ms=2402`

**Check in CSV:**
Look for row with `ms=2402` - it should have click data, not -1s:
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,grid_position_gaze,x_click,y_click,flip,matched,card_id_click,grid_position_click
2347,512,380,1,3,11,-1,-1,-1,-1,-1,-1        ← No click (53ms away)
2402,512,384,1,3,11,512,384,1,0,3,11         ← CLICK DATA HERE! ✓
2419,515,388,1,3,11,-1,-1,-1,-1,-1,-1        ← No click
```

## If Still Not Working

### Test the algorithm directly:

```python
# Simulate your case
click_ms = 2400
gaze_samples = [2347, 2402, 2419]

for gaze_ms in gaze_samples:
    diff = abs(click_ms - gaze_ms)
    within_window = diff <= 50
    print(f"Gaze at {gaze_ms}ms: diff={diff}ms, within_window={within_window}")

# Expected output:
# Gaze at 2347ms: diff=53ms, within_window=False
# Gaze at 2402ms: diff=2ms, within_window=True  ← Should match!
# Gaze at 2419ms: diff=19ms, within_window=True
```

### Check gaze sampling rate:

```python
import pandas as pd

play = pd.read_csv('playing_data_20231215_143045.csv')

# Calculate intervals between samples
intervals = play['ms'].diff().dropna()

print(f"Average interval: {intervals.mean():.1f}ms")
print(f"Max gap: {intervals.max():.0f}ms")
print(f"Min interval: {intervals.min():.0f}ms")

# Expected: ~16-17ms average (60 FPS)
# If you see gaps > 50ms, some gaze samples are missing!
```

## Quick Fix Test

Try increasing the matching window temporarily to see if that's the issue:

In the code, change:
```python
if diff <= 50 and diff < best_diff:  # Current: 50ms window
```

To:
```python
if diff <= 100 and diff < best_diff:  # Test: 100ms window
```

If clicks start matching with a larger window, then the issue is:
- Gaze samples are too sparse
- Timestamps are slightly off
- Need to verify synchronization

## Console Output to Share

When you run the game, copy this output:
```
DEBUG: Loaded X click events from click_log.csv
DEBUG: First click at Xms, Last click at Yms
DEBUG: Gaze data from Xms to Yms
DEBUG: Matched click at Xms with gaze at Yms (diff: Zms)
DEBUG: Matched X unique clicks with gaze samples out of X total clicks
```

This will help diagnose the issue!

## Summary

The algorithm should definitely match a click at 2400ms with gaze at 2402ms (2ms difference is well within the 50ms window). If it's not matching:

1. Check console debug output
2. Verify gaze data exists at 2402ms
3. Verify click_log.csv has the click at 2400ms
4. Check if there are large gaps in gaze sampling

The enhanced debug output will show exactly what's happening!

