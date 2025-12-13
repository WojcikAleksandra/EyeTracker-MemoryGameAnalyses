# Card ID Mapping Guide

## What is `card_id`?

`card_id` in your data files represents **which unique card image** the user is looking at or clicking on.

## Card ID Values

### Valid Card IDs: 1-6
Each ID corresponds to a specific image file:

| card_id | Image File | Description |
|---------|------------|-------------|
| **1** | `images/1.png` | Card image 1 |
| **2** | `images/2.png` | Card image 2 |
| **3** | `images/3.png` | Card image 3 |
| **4** | `images/4.png` | Card image 4 |
| **5** | `images/5.png` | Card image 5 |
| **6** | `images/6.png` | Card image 6 |
| **-1** | (none) | User looking at blank space (not on any card) |

### Special Value: -1
- **Meaning**: User is NOT looking at any card
- **When it appears**:
  - Looking at blank space between cards
  - Looking at the border/frame
  - Looking outside the game board
  - Eye detection failed (invalid gaze position)

## How It Works

### Card Setup

When the game starts:

```python
# For 12-card game (6 unique images):
self.front_images = [f"images/{i}.png" for i in range(1, 7)] * 2
# Creates: ["images/1.png", "images/2.png", ..., "images/6.png", 
#           "images/1.png", "images/2.png", ..., "images/6.png"]

random.shuffle(self.front_images)
# Randomizes card positions
```

### Example Game Layout

For a 12-card game (6 pairs), you might have:

```
Grid Layout (2 rows × 6 columns):

┌─────┬─────┬─────┬─────┬─────┬─────┐
│  3  │  1  │  5  │  2  │  6  │  4  │
├─────┼─────┼─────┼─────┼─────┼─────┤
│  2  │  4  │  1  │  6  │  3  │  5  │
└─────┴─────┴─────┴─────┴─────┴─────┘
```

- Position (0,0) shows image 3 → `card_id = 3`
- Position (0,1) shows image 1 → `card_id = 1`
- Position (1,2) shows image 1 → `card_id = 1` (the matching pair!)

### Gaze Detection

When tracking your gaze:

```python
def _get_card_at_gaze(self, gaze_x, gaze_y):
    for btn in self.cards:
        rect = self.card_rects_screen.get(btn)
        if rect and rect.contains(gaze_x, gaze_y):
            # Extract number from image filename
            image_path = btn.image_path  # e.g., "images/3.png"
            filename = image_path.rsplit("/", 1)[-1]  # "3.png"
            name = filename.split(".", 1)[0]  # "3"
            return int(name)  # Returns: 3
    return -1  # Not looking at any card
```

## Data File Examples

### Memorization Data
```csv
ms,x_gaze,y_gaze,valid,card_id
0,512,384,1,3         ← Looking at card 3 (images/3.png)
16,515,385,1,3        ← Still looking at card 3
33,890,420,1,1        ← Now looking at card 1 (images/1.png)
50,450,200,1,-1       ← Looking at blank space
67,1200,750,1,5       ← Looking at card 5 (images/5.png)
```

### Playing Data
```csv
ms,x_gaze,y_gaze,valid,card_id_gaze,x_click,y_click,flip,matched,card_id_click
0,512,384,1,3,-1,-1,-1,-1,-1           ← Looking at card 3
768,737,415,1,4,737,415,1,0,4          ← Looking at & clicking card 4
1200,890,420,1,1,-1,-1,-1,-1,-1        ← Looking at card 1
2353,890,415,1,1,890,415,2,0,1         ← Looking at & clicking card 1 (no match)
```

## Important Distinctions

### `card_id_gaze` vs `card_id_click`

In **playing_data** files, there are TWO card_id columns:

1. **`card_id_gaze`**: Which card user is **looking at** (every frame)
2. **`card_id_click`**: Which card user **clicked** (only when click happens)

**Example scenario:**
```csv
ms,card_id_gaze,card_id_click
1000,3,-1          ← Looking at card 3, no click
1016,3,-1          ← Still looking at card 3
1033,4,-1          ← Looking at card 4 now
1050,4,4           ← Still looking at card 4, AND clicked it!
1067,4,-1          ← Looking at card 4, click processed
```

## Card Pairs

Since it's a memory matching game:
- Each image appears **exactly twice** on the board
- Two cards with the same `card_id` are a matching pair

### Example: Finding Pairs

If your game has:
- Card at position (0,1) = `card_id: 3`
- Card at position (1,5) = `card_id: 3`

These are a matching pair! The user needs to find both cards with `card_id = 3`.

## Analysis Examples

### 1. Which card was looked at most during memorization?

```python
import pandas as pd

mem = pd.read_csv('memorization_data_20231215_143045.csv')

# Count frames looking at each card
card_attention = mem[mem['card_id'] != -1].groupby('card_id').size()

print("Frames spent on each card:")
print(card_attention)

# Output:
# card_id
# 1    45
# 2    67
# 3    89
# 4    34
# 5    56
# 6    23
```

### 2. Did user look at correct card before clicking?

```python
import pandas as pd

play = pd.read_csv('playing_data_20231215_143045.csv')

# Get click events
clicks = play[play['flip'] != -1].copy()

# Check if gaze matched click
for _, click in clicks.iterrows():
    gaze_card = click['card_id_gaze']
    click_card = click['card_id_click']
    
    if gaze_card == click_card:
        print(f"✓ Looking at card {click_card} when clicking it")
    else:
        print(f"✗ Looking at card {gaze_card} but clicked card {click_card}")
```

### 3. Track search pattern for specific card

```python
import pandas as pd

play = pd.read_csv('playing_data_20231215_143045.csv')

# Find when user looked at card 3
looking_at_3 = play[play['card_id_gaze'] == 3]

print(f"User looked at card 3 in {len(looking_at_3)} frames")
print(f"First looked at: {looking_at_3['ms'].min()} ms")
print(f"Last looked at: {looking_at_3['ms'].max()} ms")

# Did they click it?
clicked_3 = play[play['card_id_click'] == 3]
if len(clicked_3) > 0:
    print(f"Clicked card 3 at: {clicked_3['ms'].values} ms")
```

## Edge Cases

### Multiple Cards at Same Position?
- **No**: Each position has exactly one card
- **But**: Two positions can have the same `card_id` (matching pairs!)

### Can `card_id` Change During Game?
- **Memorization phase**: No, cards stay in same positions
- **Playing phase**: No, cards stay in same positions
- **Between games**: Yes! Cards are shuffled for each new game

### What if Eye Tracking Fails?
```csv
ms,x_gaze,y_gaze,valid,card_id
100,-1,-1,0,-1        ← Eye detection failed: all values = -1
```

## Quick Reference

### For Memorization Data
- `card_id = 1-6`: User looking at specific card image
- `card_id = -1`: User looking elsewhere or detection failed

### For Playing Data  
- `card_id_gaze = 1-6`: User looking at specific card
- `card_id_gaze = -1`: User looking elsewhere
- `card_id_click = 1-6`: User clicked specific card (only when click happens)
- `card_id_click = -1`: No click at this timestamp

## Summary

**`card_id` = which unique card image (1-6), or -1 for none**

- Based on the **image filename** (`images/1.png` → `card_id = 1`)
- Each image appears **twice** on the board (matching pairs)
- Positions are **randomized** at game start
- Consistent throughout each game session
- Can be used to track which specific cards users study and recall



