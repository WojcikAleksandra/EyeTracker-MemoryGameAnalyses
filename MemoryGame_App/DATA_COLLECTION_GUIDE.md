# Data Collection Guide - Two-Phase System

## Overview

The Memory Game now collects data during **BOTH** the memorization phase and the playing phase, creating two separate files with a shared unique ID for each game session.

## File Structure

After completing one game, you will see **3 files**:

### File 1: `memorization_data_<GAME_ID>.csv`
**Created**: At the end of memorization phase (after 5 seconds)
**Contains**: Gaze data during memorization

### File 2: `playing_data_<GAME_ID>.csv`
**Created**: When game finishes (all pairs found)
**Contains**: Combined gaze + click data during playing phase

### File 3: `click_log.csv` 
**Purpose**: Temporary backup (can be ignored/deleted)
**Contains**: Raw click events only

## Game ID System

Each game session gets a unique ID based on timestamp:
```
GAME_ID = YYYYMMDD_HHMMSS
```

**Example:**
- Game started at 2:30:45 PM on Dec 15, 2023
- GAME_ID = `20231215_143045`
- Files created:
  - `memorization_data_20231215_143045.csv`
  - `playing_data_20231215_143045.csv`

This allows you to match memorization and playing data from the same game!

## Data Collection Timeline

```
┌──────────────────────────────────────────────────────────────────┐
│ GAME STARTS                                                      │
│   ↓                                                              │
│ Generate Game ID: 20231215_143045                               │
│   ↓                                                              │
│ ┌─────────────────────────────────────────┐                    │
│ │ MEMORIZATION PHASE (5 seconds)           │                    │
│ │ - Cards shown face-up                    │                    │
│ │ - Timer counts down: 5...4...3...2...1   │                    │
│ │ ✅ Gaze tracking ACTIVE                  │                    │
│ │ ✅ Recording which cards user looks at   │                    │
│ │ ❌ No clicks (cards locked)              │                    │
│ └─────────────────────────────────────────┘                    │
│   ↓                                                              │
│ SAVE: memorization_data_20231215_143045.csv                     │
│   ↓                                                              │
│ ┌─────────────────────────────────────────┐                    │
│ │ PLAYING PHASE (until all pairs found)    │                    │
│ │ - Cards flipped face-down                │                    │
│ │ - User clicks to find pairs              │                    │
│ │ ✅ Gaze tracking ACTIVE                  │                    │
│ │ ✅ Recording which cards user looks at   │                    │
│ │ ✅ Recording clicks and matches          │                    │
│ └─────────────────────────────────────────┘                    │
│   ↓                                                              │
│ SAVE: playing_data_20231215_143045.csv                          │
│   ↓                                                              │
│ GAME COMPLETE                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## File Format Details

### Memorization Data File

**Filename:** `memorization_data_<GAME_ID>.csv`

**Columns:**
```csv
ms,x_gaze,y_gaze,valid,card_id
```

| Column | Description | Values |
|--------|-------------|--------|
| `ms` | Time since memorization start (milliseconds) | 0, 16, 33, ... |
| `x_gaze` | Gaze X coordinate on screen | Screen pixels |
| `y_gaze` | Gaze Y coordinate on screen | Screen pixels |
| `valid` | Eye detection quality | 1 = valid, 0 = invalid |
| `card_id` | Which card user is looking at | 1-6 or -1 (none) |

**Example:**
```csv
ms,x_gaze,y_gaze,valid,card_id
0,512,384,1,3
16,515,385,1,3
33,520,390,1,3
50,890,420,1,1
67,895,425,1,1
...
4983,300,750,1,5
5000,305,755,1,5
```

**Expected rows:** ~300 rows (5 seconds × 60 FPS)

### Playing Data File

**Filename:** `playing_data_<GAME_ID>.csv`

**Columns:**
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,x_click,y_click,flip,matched,card_id_click
```

| Column | Description | Values |
|--------|-------------|--------|
| `ms` | Time since playing phase start | 0, 16, 33, ... |
| `x_gaze` | Gaze X coordinate | Screen pixels |
| `y_gaze` | Gaze Y coordinate | Screen pixels |
| `valid` | Eye detection quality | 1 = valid, 0 = invalid |
| `card_id_gaze` | Card being looked at | 1-6 or -1 |
| `x_click` | Click X coordinate | Screen pixels or -1 |
| `y_click` | Click Y coordinate | Screen pixels or -1 |
| `flip` | Which card in pair | 1, 2, or -1 |
| `matched` | Did pair match? | 1, 0, or -1 |
| `card_id_click` | Clicked card ID | 1-6 or -1 |

**Example:**
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,x_click,y_click,flip,matched,card_id_click
0,512,384,1,3,-1,-1,-1,-1,-1
16,515,385,1,3,-1,-1,-1,-1,-1
768,737,415,1,4,737,415,1,0,4
785,740,418,1,4,-1,-1,-1,-1,-1
...
```

**Expected rows:** ~1,800 rows for 30-second game (30s × 60 FPS)

## What Gets Recorded When

### During Memorization (5 seconds)

**Recorded:**
- ✅ Gaze position (~60 times per second)
- ✅ Which card is being looked at
- ✅ Eye detection quality

**NOT recorded:**
- ❌ Clicks (cards are locked)
- ❌ Matches (not applicable)

**Why this matters:**
- Understand which cards users focus on during learning
- Analyze memorization strategies
- Compare attention during learning vs. recall

### During Playing (until completion)

**Recorded:**
- ✅ Gaze position (~60 times per second)
- ✅ Which card is being looked at
- ✅ Eye detection quality
- ✅ Click events (when they occur)
- ✅ Match results

**Why this matters:**
- See visual search patterns
- Correlate gaze with decisions
- Understand memory retrieval strategies

## Analysis Examples

### Example 1: Which cards were studied most?

```python
import pandas as pd

# Load memorization data
mem = pd.read_csv('memorization_data_20231215_143045.csv')

# Count time spent on each card (each row = ~16ms)
card_attention = mem[mem['card_id'] != -1].groupby('card_id').size() * 16 / 1000

print("Time spent memorizing each card (seconds):")
print(card_attention)
```

### Example 2: Match data across phases

```python
import pandas as pd

game_id = "20231215_143045"

# Load both phases
mem = pd.read_csv(f'memorization_data_{game_id}.csv')
play = pd.read_csv(f'playing_data_{game_id}.csv')

print(f"Memorization samples: {len(mem)}")
print(f"Playing samples: {len(play)}")
print(f"Total duration: {(mem['ms'].max() + play['ms'].max()) / 1000:.1f} seconds")
```

### Example 3: Did studying help performance?

```python
import pandas as pd

game_id = "20231215_143045"

mem = pd.read_csv(f'memorization_data_{game_id}.csv')
play = pd.read_csv(f'playing_data_{game_id}.csv')

# Cards looked at during memorization
studied_cards = mem[mem['card_id'] != -1]['card_id'].value_counts()

# Cards clicked during playing
clicks = play[play['flip'] != -1]
first_clicks = clicks[clicks['flip'] == 1]

# Compare study time with recall performance
for card_id in studied_cards.index:
    study_time = studied_cards[card_id] * 16 / 1000
    clicks_on_card = first_clicks[first_clicks['card_id_click'] == card_id]
    
    if len(clicks_on_card) > 0:
        first_click_time = clicks_on_card['ms'].min() / 1000
        print(f"Card {card_id}: studied {study_time:.1f}s, recalled at {first_click_time:.1f}s")
```

## Important Notes

### Timestamps Reset Between Phases

- Memorization phase: `ms = 0` at start of memorization
- Playing phase: `ms = 0` at start of playing (cards flip down)

They are **separate timelines**. To get total elapsed time:
```python
total_time = memorization_ms_max + playing_ms_max
```

### File Sizes

Typical 30-second game:
- `memorization_data_*.csv`: ~20 KB (300 rows)
- `playing_data_*.csv`: ~150 KB (1,800 rows)
- `click_log.csv`: ~1 KB (10-20 rows)

### Sampling Rate

- Gaze: ~60 samples/second (16ms intervals)
- Valid samples: Varies based on eye detection quality (typically 70-90%)

## Matching Files from Same Game

To find paired files:

```python
import glob
import re

# Find all memorization files
mem_files = glob.glob("memorization_data_*.csv")

for mem_file in mem_files:
    # Extract game ID
    match = re.search(r'memorization_data_(\d+_\d+)\.csv', mem_file)
    if match:
        game_id = match.group(1)
        play_file = f"playing_data_{game_id}.csv"
        
        print(f"Game {game_id}:")
        print(f"  Memorization: {mem_file}")
        print(f"  Playing: {play_file}")
```

## Best Practices

1. **Keep both files together** - They're paired by game ID
2. **Analyze both phases separately** - Different behaviors, different insights
3. **Compare across phases** - Did studying affect performance?
4. **Track game IDs** - Use them to organize your dataset
5. **Delete click_log.csv** - It's redundant (playing_data has same info)

## Console Output

When game finishes, you'll see:
```
Saved memorization data: memorization_data_20231215_143045.csv (312 samples)
Saved playing data: playing_data_20231215_143045.csv (1847 samples)
```

This confirms both files were created successfully!

## Summary

| Phase | Duration | Gaze? | Clicks? | Output File |
|-------|----------|-------|---------|-------------|
| Memorization | 5 seconds | ✅ Yes | ❌ No | `memorization_data_<ID>.csv` |
| Playing | Variable | ✅ Yes | ✅ Yes | `playing_data_<ID>.csv` |

Both files share the same `<ID>` so you can match them for comprehensive analysis!



