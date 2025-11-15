### Eye Detection Module - Production Ready

**Detects eyes and iris in real-time video. Returns structured data for gaze estimation.**

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Quick test (10 frames)
python quick_test.py

# Live visualization (shows iris as red dots on video)
python live_iris_test.py

# Full examples
python example_usage.py
```

---

## 📦 What You Get

**Input:** Video frame (640×480 BGR image)

**Output:** JSON structure with:
- Face bounding box
- Left/right eye bounding boxes
- 30×30 eye patches (as per paper)
- Iris centers (absolute + relative coordinates)

---

## 💻 Usage

```python
from eye_detector import EyeDetector
import cv2

detector = EyeDetector()
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
result = detector.detect(frame)

if result['status'] == 'ok':
    print(result['left_eye']['iris_center'])   # [x, y] in pixels
    print(result['left_eye']['iris_center_rel'])  # [x, y] in 0-1
    print(result['left_eye']['eye_patch_30x30'])  # 30×30 array
```

## 🎯 Output Format

```python
{
    "frame_id": 1234,
    "timestamp_ms": 1730553600123,
    "frame_size": {"width": 640, "height": 480},
    "status": "ok",  # or "no_face", "no_eyes", "partial"
    "face_bbox": [x, y, w, h],
    "left_eye": {
        "bbox": [x, y, w, h],
        "eye_patch_30x30": [[...], ...],  # 30×30 uint8 array
        "iris_bbox": [x, y, w, h],
        "iris_center": [cx, cy],          # absolute pixels
        "iris_center_rel": [rx, ry],      # 0.0 to 1.0
        "quality": {...}
    },
    "right_eye": { ... }
}
```

