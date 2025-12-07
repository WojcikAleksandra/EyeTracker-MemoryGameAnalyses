# Grid Position System - Complete Guide

## Overview

The data now includes **two pieces of information** for each card:
1. **`card_id`**: Which image (1-6) - identifies the picture content
2. **`grid_position`**: Where on the grid (11-44) - identifies the location

This allows you to track both **what card** users are looking at and **where** that card is positioned!

## Grid Position Format

Grid positions use a **two-digit format: `RC`** where:
- **R** = Row number (1-based)
- **C** = Column number (1-based)

### Example: 8-Card Game (2 rows × 4 columns)

```
┌──────┬──────┬──────┬──────┐
│  11  │  12  │  13  │  14  │  ← Row 1
├──────┼──────┼──────┼──────┤
│  21  │  22  │  23  │  24  │  ← Row 2
└──────┴──────┴──────┴──────┘
  Col1   Col2   Col3   Col4
```

| Position | Meaning |
|----------|---------|
| **11** | Row 1, Column 1 (top-left) |
| **12** | Row 1, Column 2 |
| **13** | Row 1, Column 3 |
| **14** | Row 1, Column 4 (top-right) |
| **21** | Row 2, Column 1 (bottom-left) |
| **22** | Row 2, Column 2 |
| **23** | Row 2, Column 3 |
| **24** | Row 2, Column 4 (bottom-right) |

### Example: 12-Card Game (2 rows × 6 columns)

```
┌──────┬──────┬──────┬──────┬──────┬──────┐
│  11  │  12  │  13  │  14  │  15  │  16  │  ← Row 1
├──────┼──────┼──────┼──────┼──────┼──────┤
│  21  │  22  │  23  │  24  │  25  │  26  │  ← Row 2
└──────┴──────┴──────┴──────┴──────┴──────┘
```

Positions range from **11 to 26**.

### Example: 12-Card Game Alternative (3 rows × 4 columns)

```
┌──────┬──────┬──────┬──────┐
│  11  │  12  │  13  │  14  │  ← Row 1
├──────┼──────┼──────┼──────┤
│  21  │  22  │  23  │  24  │  ← Row 2
├──────┼──────┼──────┼──────┤
│  31  │  32  │  33  │  34  │  ← Row 3
└──────┴──────┴──────┴──────┘
```

Positions range from **11 to 34**.

## Special Value: -1

**grid_position = -1** means:
- User is NOT looking at any card
- Looking at blank space between cards
- Looking outside the game board
- Eye detection failed

## Complete Example: 8-Card Game

Let's say you have an 8-card game with this layout:

```
Grid:
┌──────┬──────┬──────┬──────┐
│  11  │  12  │  13  │  14  │
│ img3 │ img1 │ img4 │ img2 │
├──────┼──────┼──────┼──────┤
│  21  │  22  │  23  │  24  │
│ img2 │ img4 │ img1 │ img3 │
└──────┴──────┴──────┴──────┘
```

**Card pairs:**
- Image 1: at positions **12** and **23** → `card_id=1`
- Image 2: at positions **14** and **21** → `card_id=2`
- Image 3: at positions **11** and **24** → `card_id=3`
- Image 4: at positions **13** and **22** → `card_id=4`

## Data File Formats

### Memorization Data

**Filename:** `memorization_data_<GAME_ID>.csv`

**Columns:**
```csv
ms,x_gaze,y_gaze,valid,card_id,grid_position
```

**Example data:**
```csv
ms,x_gaze,y_gaze,valid,card_id,grid_position
0,320,240,1,3,11        ← Looking at image 3 at position 11 (top-left)
16,325,245,1,3,11       ← Still looking at same card
33,640,240,1,1,12       ← Now looking at image 1 at position 12
50,960,240,1,4,13       ← Looking at image 4 at position 13
67,1280,240,1,2,14      ← Looking at image 2 at position 14
83,320,480,1,2,21       ← Looking at image 2 at position 21 (bottom-left)
100,640,480,1,4,22      ← Looking at image 4 at position 22
116,960,480,1,1,23      ← Looking at image 1 at position 23
133,1280,480,1,3,24     ← Looking at image 3 at position 24 (bottom-right)
150,800,360,1,-1,-1     ← Looking at blank space
```

### Playing Data

**Filename:** `playing_data_<GAME_ID>.csv`

**Columns:**
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,grid_position_gaze,x_click,y_click,flip,matched,card_id_click,grid_position_click
```

**Example data:**
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,grid_position_gaze,x_click,y_click,flip,matched,card_id_click,grid_position_click
0,320,240,1,3,11,-1,-1,-1,-1,-1,-1           ← Looking at position 11
16,325,245,1,3,11,-1,-1,-1,-1,-1,-1          ← Still looking at position 11
768,320,240,1,3,11,320,240,1,0,3,11          ← Looking at & clicking position 11 (image 3)
785,1280,480,1,3,24,-1,-1,-1,-1,-1,-1        ← Now looking at position 24
2353,1280,480,1,3,24,1280,480,2,1,3,24       ← Looking at & clicking position 24 (image 3) - MATCH!
```

### Click Log (Temporary File)

**Filename:** `click_log.csv`

**Columns:**
```csv
ms,x,y,flip,matched,card_id,grid_position
```

**Example:**
```csv
ms,x,y,flip,matched,card_id,grid_position
768,320,240,1,0,3,11
2353,1280,480,2,1,3,24
```

## Analysis Examples

### 1. Which grid positions got most attention during memorization?

```python
import pandas as pd

mem = pd.read_csv('memorization_data_20231215_143045.csv')

# Count frames for each position
position_attention = mem[mem['grid_position'] != -1].groupby('grid_position').size()

print("Attention by grid position:")
print(position_attention)

# Output:
# grid_position
# 11    45
# 12    67
# 13    89
# 14    34
# 21    56
# 22    23
# 23    78
# 24    12
```

### 2. Did user look at corners vs. center?

```python
import pandas as pd

mem = pd.read_csv('memorization_data_20231215_143045.csv')

# Define corner positions (for 2x4 grid)
corners = [11, 14, 21, 24]
center = [12, 13, 22, 23]

corner_frames = len(mem[mem['grid_position'].isin(corners)])
center_frames = len(mem[mem['grid_position'].isin(center)])

print(f"Corner attention: {corner_frames} frames")
print(f"Center attention: {center_frames} frames")
```

### 3. Trace search pattern by position

```python
import pandas as pd

play = pd.read_csv('playing_data_20231215_143045.csv')

# Get sequence of positions looked at (remove duplicates)
gaze_sequence = play[play['grid_position_gaze'] != -1].copy()
gaze_sequence = gaze_sequence[gaze_sequence['grid_position_gaze'].diff() != 0]

print("Gaze position sequence:")
print(gaze_sequence[['ms', 'grid_position_gaze', 'card_id_gaze']].head(20))

# Example output:
#     ms  grid_position_gaze  card_id_gaze
# 0    0                  11             3
# 1  250                  12             1
# 2  500                  13             4
# 3  750                  11             3  ← Returned to position 11
```

### 4. Find matching pair locations

```python
import pandas as pd

mem = pd.read_csv('memorization_data_20231215_143045.csv')

# Get unique (card_id, grid_position) pairs
card_positions = mem[mem['card_id'] != -1][['card_id', 'grid_position']].drop_duplicates()

# Group by card_id to find pairs
pairs = card_positions.groupby('card_id')['grid_position'].apply(list)

print("Card pairs (image ID : positions):")
for card_id, positions in pairs.items():
    if len(positions) == 2:
        print(f"Image {card_id}: positions {positions[0]} and {positions[1]}")
```

### 5. Click accuracy: Did user look at correct position?

```python
import pandas as pd

play = pd.read_csv('playing_data_20231215_143045.csv')

# Get click events
clicks = play[play['flip'] != -1].copy()

# Check if gaze position matched click position
correct_gaze = clicks['grid_position_gaze'] == clicks['grid_position_click']

print(f"Clicks where user looked at target: {correct_gaze.sum()} / {len(clicks)}")
print(f"Accuracy: {correct_gaze.mean() * 100:.1f}%")
```

## Visual Analysis

### Heatmap by Grid Position

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mem = pd.read_csv('memorization_data_20231215_143045.csv')

# Count attention per position
attention = mem[mem['grid_position'] != -1].groupby('grid_position').size()

# Create 2D heatmap (for 2x4 grid)
heatmap = np.zeros((2, 4))
for pos, count in attention.items():
    if pos != -1:
        row = int(str(pos)[0]) - 1  # First digit = row (0-indexed)
        col = int(str(pos)[1]) - 1  # Second digit = col (0-indexed)
        heatmap[row, col] = count

plt.figure(figsize=(10, 5))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar(label='Frames')
plt.title('Attention Heatmap by Grid Position')
plt.xlabel('Column')
plt.ylabel('Row')
plt.xticks(range(4), ['Col 1', 'Col 2', 'Col 3', 'Col 4'])
plt.yticks(range(2), ['Row 1', 'Row 2'])

# Add position labels
for i in range(2):
    for j in range(4):
        position = int(f"{i+1}{j+1}")
        plt.text(j, i, f'{position}\n{int(heatmap[i,j])}', 
                ha='center', va='center', color='white', fontsize=12)

plt.tight_layout()
plt.savefig('position_heatmap.png', dpi=150)
plt.show()
```

## Decoding Grid Position

To extract row and column from grid_position:

```python
def decode_position(grid_position):
    """Convert grid position to (row, col) tuple."""
    if grid_position == -1:
        return (-1, -1)
    
    pos_str = str(grid_position)
    row = int(pos_str[0])
    col = int(pos_str[1])
    return (row, col)

# Example
position = 23
row, col = decode_position(position)
print(f"Position {position}: Row {row}, Column {col}")
# Output: Position 23: Row 2, Column 3
```

## Summary Table

| Data Type | card_id | grid_position | Meaning |
|-----------|---------|---------------|---------|
| **Gaze** | 1-6 | 11-44 | Looking at specific card image at specific position |
| **Gaze** | 1-6 | -1 | Looking at card but position unknown (shouldn't happen) |
| **Gaze** | -1 | -1 | Looking at blank space or detection failed |
| **Click** | 1-6 | 11-44 | Clicked specific card at specific position |

## Key Insights You Can Now Derive

1. **Spatial Patterns**: Which positions get attention (edges vs. center, top vs. bottom)
2. **Pair Discovery**: Track when user finds both positions of matching pair
3. **Search Strategy**: Analyze systematic vs. random search patterns
4. **Position Memory**: Do users remember card positions or just images?
5. **Click-Gaze Coordination**: Do users look where they click?

## Quick Reference

**Format:** `RC` (Row-Column)
- R = 1, 2, 3, ... (rows, top to bottom)
- C = 1, 2, 3, 4, ... (columns, left to right)

**Examples:**
- `11` = Top-left corner
- `14` = Top-right corner (for 4-column layout)
- `21` = Bottom-left (for 2-row layout)
- `34` = Row 3, Column 4
- `-1` = Not on any card

With this system, you can now perform **detailed spatial analysis** of player behavior!

