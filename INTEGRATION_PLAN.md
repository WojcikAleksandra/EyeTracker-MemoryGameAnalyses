# Gaze Localization Integration Plan

## Overview
Integration of `GazeEngine` from `gaze_localizator.py` into `MemoryGame_v2.py` to collect comprehensive gaze data during gameplay.

---

## 1. Architecture & Data Flow

### 1.1 Components Integration
- **GazeEngine**: Manages camera, eye detection, calibration, and gaze prediction
- **MemoryGameWindow**: Main window with navigation flow
- **MemoryGameBoard**: Game board with cards and gameplay logic
- **New: CalibrationScreen**: Dedicated calibration UI before game starts

### 1.2 Data Collection Points
1. **Calibration Phase**: Collect calibration samples (target points)
2. **Memorization Phase** (5 seconds): Continuous gaze tracking
3. **Active Play Phase**: Continuous gaze tracking + click events

---

## 2. Implementation Steps

### Step 1: Create Gaze Data Logger
**File**: `MemoryGame_App/gaze_data_logger.py`

**Purpose**: Centralized logging system for all gaze data

**Features**:
- Generate unique session ID (format: `YYYYMMDD_HHMMSS`)
- Log to CSV with comprehensive fields
- Handle multiple data types (gaze samples, clicks, phase transitions)

**Data Schema**:
```csv
session_id,timestamp_ms,phase,event_type,gaze_x,gaze_y,click_x,click_y,element_type,card_row,card_col,card_id,card_image_name,matched,game_time_ms
```

**Fields**:
- `session_id`: Unique identifier (e.g., "20241215_143022")
- `timestamp_ms`: Milliseconds since game start (or calibration start)
- `phase`: "calibration", "memorization", "play"
- `event_type`: "gaze_sample", "click", "phase_start", "phase_end"
- `gaze_x`, `gaze_y`: Gaze coordinates (screen space)
- `click_x`, `click_y`: Click coordinates (if event_type="click")
- `element_type`: "card", "status_label", "timer_label", "grid_frame", "other"
- `card_row`, `card_col`: Card position (1-indexed, e.g., row=2, col=3 → "23")
- `card_id`: Numeric card ID from image name (e.g., "3" from "images/3.png")
- `card_image_name`: Full image filename (e.g., "3.png")
- `matched`: 0/1 if card was correctly matched (only for clicks)
- `game_time_ms`: Time since game start (0 during calibration)

---

### Step 2: Add Calibration Screen
**Location**: `MemoryGameWindow.show_calibration_screen()`

**Flow**:
1. Display calibration instructions
2. Show 9 calibration points in 3x3 grid (or configurable)
3. For each point:
   - Display point on screen (circle/crosshair)
   - Collect samples for ~1-2 seconds
   - Show progress indicator
4. After all points: train models (`fit_models()`)
5. Show calibration success/failure message
6. Auto-advance to countdown or allow manual proceed

**UI Elements**:
- Title: "Eye Tracking Calibration"
- Instructions text
- Calibration point indicator (animated)
- Progress bar/counter
- Status message

**Integration**:
- Insert between "Play" button click and countdown
- Replace direct `show_countdown()` call with `show_calibration_screen()`

---

### Step 3: Integrate GazeEngine into MemoryGameWindow
**Location**: `MemoryGameWindow.__init__()`

**Changes**:
- Initialize `GazeEngine` with window size
- Store reference: `self.gaze_engine`
- Initialize `GazeDataLogger`: `self.gaze_logger`
- Handle camera initialization errors gracefully

**Window Size Handling**:
- Get actual window size after first show
- Pass to `GazeEngine(screen_size=(width, height))`
- Update on resize (if allowed)

---

### Step 4: Add Gaze Tracking to MemoryGameBoard
**Location**: `MemoryGameBoard` class

**New Attributes**:
- `gaze_engine`: Reference to GazeEngine (passed from window)
- `gaze_logger`: Reference to GazeDataLogger
- `gaze_timer`: QTimer for periodic gaze sampling (e.g., 30-60ms = ~30-16 FPS)
- `tracking_active`: Boolean flag

**New Methods**:
- `start_gaze_tracking()`: Start gaze sampling timer
- `stop_gaze_tracking()`: Stop timer and cleanup
- `_sample_gaze()`: Called by timer, predicts gaze and logs
- `_get_element_at_point(x, y)`: Determines what UI element is at gaze point
- `_get_card_at_point(x, y)`: Returns card button if gaze is over a card
- `_get_card_info(btn)`: Extracts card position (row, col), ID, image name

**Integration Points**:
1. **Memorization Phase**:
   - Start tracking in `start_memorize_phase()`
   - Log phase_start event
   - Continue tracking during preview
   - Log phase_end when preview ends

2. **Active Play Phase**:
   - Continue tracking (already started)
   - Log phase_start when play begins
   - Track during entire gameplay
   - Stop tracking in `_finish_game()` or `stop_all_timers()`

3. **Click Events**:
   - Enhance `log_click()` to also log gaze data
   - Add element detection for click location
   - Include matched status

---

### Step 5: Element Detection System
**Location**: `MemoryGameBoard._get_element_at_point()`

**Logic**:
1. Check if point is within `board_rect_screen` (card grid area)
2. If yes, iterate through `card_rects_screen` to find matching card
3. If no card matches, check other UI elements:
   - `status_label` geometry
   - `timer_label` geometry
   - `grid_frame` (but not a card)
   - Return "other" if none match

**Card Position Calculation**:
- From card button index: `i // self.cols` = row, `i % self.cols` = col
- Format as string: `f"{row+1}{col+1}"` (1-indexed, e.g., "23" for row 2, col 3)

**Card Info Extraction**:
- From `btn.image_path`: extract filename and numeric ID
- Store row/col in button attribute during creation

---

### Step 6: Enhanced Click Logging
**Location**: `MemoryGameBoard.log_click()`

**Enhancements**:
- Get current gaze position at click time
- Detect element at click location
- Extract card info if applicable
- Log to gaze logger with full context
- Keep existing CSV format for backward compatibility (optional)

---

### Step 7: Data Collection During Phases

**Memorization Phase** (5 seconds):
- Sample rate: ~30-60ms (16-33 FPS)
- Log each sample with:
  - `phase="memorization"`
  - `event_type="gaze_sample"`
  - `gaze_x`, `gaze_y`
  - `element_type`, `card_row`, `card_col`, `card_id`, `card_image_name` (if over card)
  - `game_time_ms` (relative to game start, but memorization starts at 0)

**Active Play Phase**:
- Same continuous sampling
- `phase="play"`
- Include click events with full context
- Track matched status for card clicks

---

### Step 8: Session Management

**Session ID Generation**:
- Format: `YYYYMMDD_HHMMSS` (e.g., "20241215_143022")
- Generated once per game session
- Stored in `GazeDataLogger`
- Included in every log entry

**File Naming**:
- `gaze_data_YYYYMMDD_HHMMSS.csv`
- Or: `gaze_data_{session_id}.csv`

**Session Lifecycle**:
1. User clicks "Play" → Generate session ID
2. Calibration → Log calibration events
3. Countdown → Optional logging
4. Memorization → Start game timer, log phase
5. Play → Continue logging
6. Game end → Close log file, save session metadata

---

## 3. Data Schema Details

### Event Types
- `gaze_sample`: Regular gaze tracking sample
- `click`: Mouse click event
- `phase_start`: Phase transition (memorization/play start)
- `phase_end`: Phase transition (memorization/play end)
- `calibration_point`: Calibration sample collection
- `calibration_complete`: Calibration finished

### Element Types
- `card`: A game card
- `status_label`: Status text at top
- `timer_label`: Timer/moves display
- `grid_frame`: Card grid area (but not a specific card)
- `other`: Any other UI element
- `calibration_point`: During calibration

### Phase Values
- `calibration`: Eye tracking calibration
- `countdown`: 3-2-1 countdown
- `memorization`: 5-second card preview
- `play`: Active gameplay

---

## 4. Implementation Order

1. ✅ **Create GazeDataLogger** (`gaze_data_logger.py`)
   - Session ID generation
   - CSV writing with full schema
   - Helper methods for different event types

2. ✅ **Add Calibration Screen** (`MemoryGameWindow`)
   - UI layout
   - Calibration point display
   - Sample collection loop
   - Integration with GazeEngine

3. ✅ **Initialize GazeEngine in Window** (`MemoryGameWindow.__init__`)
   - Camera setup
   - Error handling
   - Pass to board

4. ✅ **Add Gaze Tracking to Board** (`MemoryGameBoard`)
   - Timer setup
   - Sampling method
   - Element detection
   - Phase logging

5. ✅ **Enhance Click Logging**
   - Integrate with gaze logger
   - Element detection
   - Full context

6. ✅ **Test & Validate**
   - Calibration flow
   - Data collection during phases
   - File output verification

---

## 5. Technical Considerations

### Performance
- Gaze sampling: 30-60ms intervals (16-33 FPS) to balance accuracy and performance
- Use QTimer for non-blocking sampling
- Batch file writes if needed (flush periodically)

### Error Handling
- Camera initialization failure → Show error, allow game without gaze tracking
- Calibration failure → Allow retry or skip
- Missing eye detection → Skip sample, don't crash

### Coordinate Systems
- All coordinates in screen/global space (from `mapToGlobal()`)
- GazeEngine uses screen_size passed during initialization
- Ensure window size is captured correctly

### Threading
- GazeEngine camera operations may block
- Consider moving camera operations to separate thread (future enhancement)
- For now, use QTimer with reasonable intervals

---

## 6. File Structure

```
MemoryGame_App/
├── MemoryGame_v2.py          (modified: add gaze integration)
├── gaze_data_logger.py       (new: logging system)
├── click_log.csv             (existing: may be deprecated or merged)
└── gaze_data_*.csv           (new: comprehensive gaze logs)
```

---

## 7. Testing Checklist

- [ ] Calibration screen displays and collects samples
- [ ] Calibration trains models successfully
- [ ] Gaze tracking works during memorization
- [ ] Gaze tracking works during play
- [ ] Element detection correctly identifies cards
- [ ] Card position calculation is correct (row/col format)
- [ ] Click events include gaze context
- [ ] Session ID is unique and formatted correctly
- [ ] CSV file contains all required fields
- [ ] No performance issues during gameplay
- [ ] Camera errors are handled gracefully

---

## 8. Future Enhancements (Optional)

- Real-time gaze visualization overlay
- Gaze heatmap generation
- Statistical analysis of gaze patterns
- Export to JSON for easier analysis
- Multiple calibration profiles
- Calibration quality metrics


