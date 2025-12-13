# Click Data Fix - Why Clicks Were All -1

## Problem

In the `playing_data_*.csv` file, all click-related columns were showing `-1`:
- `x_click = -1`
- `y_click = -1`
- `flip = -1`
- `matched = -1`
- `card_id_click = -1`
- `grid_position_click = -1`

Even though clicks were happening during the game!

## Root Causes

### Issue 1: Timestamp Mismatch

**The Problem:**
- **Click timestamps** used `game_start_time` (set immediately when playing phase starts)
- **Gaze timestamps** used `playing_start_time` (set on first frame)
- These two times were **different** (could be 1000-2000ms apart!)

```python
# OLD CODE - Two different time references
def _start_game_timer(self):
    self.game_start_time = QTime.currentTime()  # ← Time 1: for clicks
    self.playing_start_time = None               # ← Time 2: set later for gaze

def log_click(self):
    ms = self.game_start_time.msecsTo(now)      # ← Uses Time 1

def _process_gaze_frame(self):
    if self.playing_start_time is None:
        self.playing_start_time = QTime.currentTime()  # ← Set later!
    timestamp_ms = self.playing_start_time.msecsTo(...)  # ← Uses Time 2
```

**The Result:**
- Click at `ms=1000` (using `game_start_time`)
- Gaze at `ms=0` (using `playing_start_time` set 1000ms later)
- Matching window: ±50ms → No match! ❌

**The Fix:**
```python
# NEW CODE - Same time reference
def _start_game_timer(self):
    self.game_start_time = QTime.currentTime()
    self.playing_start_time = self.game_start_time  # ← Use SAME reference!
```

### Issue 2: CSV Format Check Too Strict

**The Problem:**
```python
# OLD CODE
if len(parts) >= 7:  # Required exactly 7 columns
    # Parse click data
```

If you had old `click_log.csv` files with only 6 columns (before grid_position was added), ALL clicks would be ignored!

**The Fix:**
```python
# NEW CODE - Backward compatible
if len(parts) >= 6:  # Need at least 6 columns
    grid_pos = int(parts[6]) if len(parts) >= 7 else -1
    # Parse click data with optional grid_position
```

## How the Matching Works

### Matching Algorithm

```python
# For each gaze sample (every ~16ms)
for gaze in self.playing_gaze_data:
    gaze_ms = gaze['ms']
    
    # Find click within ±50ms window
    for click in click_data:
        click_ms = click['ms']
        
        if abs(gaze_ms - click_ms) <= 50:
            # Match! Include click data in this row
            x_click = click['x']
            flip = click['flip']
            # etc.
```

### Example Timeline

**Working scenario (after fix):**
```
Timeline (ms since playing phase start):
0      ─── First gaze sample
16     ─── Gaze sample
33     ─── Gaze sample
...
1000   ─── Gaze sample
1005   ─── CLICK! (click_ms = 1005)
1016   ─── Gaze sample (matched with click at 1005, within 50ms!)
        └─► This row will have click data filled in
1033   ─── Gaze sample
```

**Broken scenario (before fix):**
```
Timeline:
Click timeline (uses game_start_time):
1005   ─── CLICK! (click_ms = 1005 in click log)

Gaze timeline (uses playing_start_time, set 1000ms later):
0      ─── First gaze sample (1000ms after game_start_time!)
16     ─── Gaze sample
33     ─── Gaze sample
...
No match! Click at 1005 vs gaze at 0-50 → difference > 50ms
```

## Verification

### Check Console Output

When the game finishes, you should now see:
```
Saved playing data: playing_data_20231215_143045.csv (1847 samples)
DEBUG: Loaded 12 click events from click_log.csv
DEBUG: Matched 12 clicks with gaze samples out of 12 total clicks
```

If you see `Matched 0 clicks`, there's still a timestamp synchronization issue.

### Check CSV Output

**Before fix:**
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,grid_position_gaze,x_click,y_click,flip,matched,card_id_click,grid_position_click
0,512,384,1,3,11,-1,-1,-1,-1,-1,-1
16,515,385,1,3,11,-1,-1,-1,-1,-1,-1
1005,737,415,1,4,13,-1,-1,-1,-1,-1,-1    ← Should have click data!
```

**After fix:**
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,grid_position_gaze,x_click,y_click,flip,matched,card_id_click,grid_position_click
0,512,384,1,3,11,-1,-1,-1,-1,-1,-1
16,515,385,1,3,11,-1,-1,-1,-1,-1,-1
1005,737,415,1,4,13,737,415,1,0,4,13    ← Click data present! ✓
1021,740,418,1,4,13,-1,-1,-1,-1,-1,-1
```

## Testing

### Quick Test

1. Delete old `click_log.csv` if it exists
2. Play a complete game
3. Check console output for "DEBUG: Matched X clicks"
4. Open `playing_data_*.csv`
5. Search for rows where `flip != -1`
6. You should find rows with click data!

### Verification Script

```python
import pandas as pd

# Load playing data
play = pd.read_csv('playing_data_20231215_143045.csv')

# Count rows with click data
clicks = play[play['flip'] != -1]

print(f"Total gaze samples: {len(play)}")
print(f"Samples with clicks: {len(clicks)}")
print(f"Click data percentage: {len(clicks)/len(play)*100:.2f}%")

if len(clicks) > 0:
    print("\n✓ Click data is present!")
    print(clicks[['ms', 'card_id_gaze', 'grid_position_gaze', 
                  'card_id_click', 'grid_position_click', 'flip', 'matched']].head())
else:
    print("\n✗ No click data found - timestamps may still be mismatched")
```

## Summary

**Fixed Issues:**
1. ✅ Synchronized timestamps (both use `game_start_time`)
2. ✅ Backward compatible CSV parsing (works with old files)
3. ✅ Added debug output to verify matching

**Result:**
- Clicks now properly merged with gaze data
- Click information appears in rows within ±50ms of actual click
- All click columns populated with real data instead of -1

**What to expect:**
- Most gaze rows: All click columns = -1 (no click at that moment)
- Rows near clicks: Click columns filled with actual data
- Typical: 10-20 clicks per game → 10-20 rows with click data out of ~1800 total rows

The fix ensures that when you click, the corresponding gaze sample(s) will have the click information attached!



