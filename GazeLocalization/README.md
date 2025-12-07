### Gaze Localization Module

A webcam-based eye-tracking engine that:
- uses the external `EyeDetector` class for face and eye detection,
- extracts features from detected eye rectangles (patches),
- trains two regression models (X and Y) based on calibration samples,
- returns real-time gaze point coordinates within the application window layout.

The module consists of:
- `gaze_localizator.py` – the `GazeEngine` class (calibration and tracking logic, no GUI),
- `gaze_loc_example_usage.py` – a demo with calibration on clickable points, gaze tracking, and camera preview with eye detection.

Requirements:
- Python 3.9+
- `eye_detector.EyeDetector` module installed (in a separate project folder),
- working webcam.

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run full demo (calibration + gaze tracking + camera preview)
python gaze_loc_example_usage.py
```

---

## 📦 What You Get

**Input** (user-side / GUI):
- Video stream from the camera (handled within `GazeEngine`),
- List of calibration points in the window coordinate system (e.g., (x, y) in pixels),
- Information when the user "confirms" looking at a point (click / timer).

**Output:**
- Trained models `model_x`, `model_y` mapping eye features → `(gx, gy)`,
- `predict_gaze()` method returning the predicted gaze point in real time: `(gx, gy)`.

---

## 💻 Usage

A minimal example of using `GazeEngine` in the application (excluding the entire checkpoint demo):

```python
from gaze_localizator import GazeEngine

screen_size = (1280, 720)
engine = GazeEngine(
    screen_size=screen_size,
    model_type="ridge",
    patch_height=8,
    patch_width=9,
    min_samples=60,
)

engine.start_calibration()

# In the app GUI:
# - display the calibration points (target_x, target_y) one by one,
# - each time the user looks at a point (e.g., clicks on it),
# COLLECT several samples from a time window – as in gaze_loc_example_usage.py.
# For simplicity – here's a single call:

target_x, target_y = 640, 360
ok = engine.add_calibration_sample(target_x, target_y)

if engine.fit_models():
    print("Calibration complete, gaze can be tracked.")

while True:
    gaze = engine.predict_gaze()
    if gaze is not None:
        gx, gy = gaze
        print("Gaze:", gx, gy)

    break

engine.close()
```

## 🎯 Output Format

The most important items that `GazeEngine` returns/stores:

```python
from gaze_localizator import GazeEngine
import numpy as np

engine = GazeEngine(screen_size=(1280, 720))

# After calibration:
engine.calib_X   # list[np.ndarray] -> after fit_models: np.ndarray [n_samples, n_features]
engine.calib_yx  # list[float]      -> after fit_models: X values for calibration points
engine.calib_yy  # list[float]      -> after fit_models: Y values for calibration points

# Tracking:
gaze = engine.predict_gaze()  # -> (gx, gy) or None
if gaze is not None:
    gx, gy = gaze
    # gx, gy are in pixels, in the screen_size specified when creating GazeEngine.
```
All implementation details (`validator`, `feature_extractor`, ML model selection, predictions history smoothing) are hidden inside `GazeEngine` so that the module can be easily hooked into any GUI/app.
