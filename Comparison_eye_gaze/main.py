"""
Gaze Localization Comparison: Production (Appearance) vs Demo2_v4 (Geometric)
Super simple - calibration then test dots, saves results to CSV.
"""

import sys
import os
import time
import random
import csv
import math
from datetime import datetime

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QImage, QPixmap

import cv2
import numpy as np
from sklearn.linear_model import Ridge

# Use shared modules from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GazeLocalization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eye-detection-final'))

from eye_detector import EyeDetector


# ==============================================================================
# ALGORITHM 1: Production - Appearance-based (eye patch pixels)
# ==============================================================================

class AppearanceFeatureExtractor:
    """Extracts raw eye patch pixels as features (from GazeLocalization)."""
    
    def __init__(self, patch_height=8, patch_width=9):
        self.patch_height = patch_height
        self.patch_width = patch_width

    def _extract_eye_features(self, frame, eye_bbox):
        x, y, w, h = eye_bbox
        eye_region = frame[y:y+h, x:x+w]
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        patch = cv2.resize(eye_gray, (self.patch_width, self.patch_height), interpolation=cv2.INTER_AREA)
        return (patch.astype(np.float32) / 255.0).flatten()

    def __call__(self, frame, result):
        left = result["left_eye"]
        right = result["right_eye"]
        left_feats = self._extract_eye_features(frame, left["bbox"])
        right_feats = self._extract_eye_features(frame, right["bbox"])
        return np.concatenate([left_feats, right_feats]).astype(np.float32)


# ==============================================================================
# ALGORITHM 2: Demo2_v4 - Geometric features (pupil positions)
# ==============================================================================

class GeometricFeatureExtractor:
    """Extracts geometric features: relative pupil positions + face position (from demo2_v4)."""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def __call__(self, frame, result):
        left = result["left_eye"]
        right = result["right_eye"]
        face_x, face_y, face_w, face_h = result["face_bbox"]

        # Normalized pupil positions within eye bbox
        xL, yL = left["iris_center_rel"]
        xR, yR = right["iris_center_rel"]

        # Angle of line connecting pupils
        ixL, iyL = left["iris_center"]
        ixR, iyR = right["iris_center"]
        theta = float(np.arctan2(iyR - iyL, ixR - ixL))

        # Face center position normalized to [0,1]
        face_cx = face_x + face_w / 2.0
        face_cy = face_y + face_h / 2.0
        x_face = face_cx / self.frame_width
        y_face = face_cy / self.frame_height

        return np.array([xL, yL, xR, yR, theta, x_face, y_face], dtype=np.float32)


# ==============================================================================
# Frame Validator
# ==============================================================================

class FrameValidator:
    """Validates frames for both algorithms."""
    
    def __init__(self, symmetry_tolerance=0.4, angle_tolerance_deg=40.0):
        self.symmetry_tolerance = symmetry_tolerance
        self.angle_tolerance_rad = math.radians(angle_tolerance_deg)

    def is_valid(self, result):
        if result is None:
            return False
        if result.get("status") != "ok":
            return False
        if result.get("face_bbox") is None:
            return False
        
        left = result.get("left_eye")
        right = result.get("right_eye")
        if left is None or right is None:
            return False
        
        # Check symmetry
        face_x, face_y, face_w, face_h = result["face_bbox"]
        face_cx = face_x + face_w / 2.0
        ixL, _ = left["iris_center"]
        ixR, _ = right["iris_center"]
        midpoint_x = (ixL + ixR) / 2.0
        if abs(midpoint_x - face_cx) / face_w > self.symmetry_tolerance:
            return False
        
        # Check horizontal alignment
        iyL = left["iris_center"][1]
        iyR = right["iris_center"][1]
        dx = ixR - ixL
        dy = iyR - iyL
        if dx == 0:
            return False
        angle = math.atan2(dy, dx)
        if abs(angle) > self.angle_tolerance_rad:
            return False
        
        return True


# ==============================================================================
# Dual Gaze Engine - trains both algorithms with same calibration data
# ==============================================================================

class DualGazeEngine:
    """Manages both algorithms with shared calibration."""
    
    def __init__(self, screen_size):
        self.screen_w, self.screen_h = screen_size
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Get frame size
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera")
        self.frame_h, self.frame_w = frame.shape[:2]
        
        # Shared components
        self.detector = EyeDetector()
        self.validator = FrameValidator()
        
        # Algorithm 1: Appearance-based (production)
        self.appearance_extractor = AppearanceFeatureExtractor(patch_height=8, patch_width=9)
        self.appearance_model_x = Ridge(alpha=1.0)
        self.appearance_model_y = Ridge(alpha=1.0)
        
        # Algorithm 2: Geometric (demo2_v4)
        self.geometric_extractor = GeometricFeatureExtractor(self.frame_w, self.frame_h)
        self.geometric_model_x = Ridge(alpha=1.0)
        self.geometric_model_y = Ridge(alpha=1.0)
        
        # Calibration data (shared)
        self.calib_frames = []  # (frame, result, target_x, target_y)
        self.min_samples = 60
        
        self._calibrated = False
    
    def capture_and_detect(self):
        if self.cap is None:
            return None, None
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        result = self.detector.detect(frame)
        return frame, result
    
    def add_calibration_sample(self, target_x, target_y, frame, result):
        """Add calibration sample if valid."""
        if not self.validator.is_valid(result):
            return False
        self.calib_frames.append((frame.copy(), result, target_x, target_y))
        return True
    
    def fit_models(self):
        """Train both algorithms on the same calibration data."""
        if len(self.calib_frames) < self.min_samples:
            return False
        
        # Extract features for both algorithms
        appearance_X = []
        geometric_X = []
        y_x = []
        y_y = []
        
        for frame, result, tx, ty in self.calib_frames:
            try:
                app_feats = self.appearance_extractor(frame, result)
                geo_feats = self.geometric_extractor(frame, result)
                appearance_X.append(app_feats)
                geometric_X.append(geo_feats)
                y_x.append(float(tx))
                y_y.append(float(ty))
            except Exception:
                continue
        
        if len(appearance_X) < self.min_samples:
            return False
        
        # Train appearance models
        X_app = np.asarray(appearance_X, dtype=np.float32)
        self.appearance_model_x.fit(X_app, np.array(y_x))
        self.appearance_model_y.fit(X_app, np.array(y_y))
        
        # Train geometric models
        X_geo = np.asarray(geometric_X, dtype=np.float32)
        self.geometric_model_x.fit(X_geo, np.array(y_x))
        self.geometric_model_y.fit(X_geo, np.array(y_y))
        
        self._calibrated = True
        return True
    
    def predict_appearance(self, frame, result):
        """Get prediction from appearance-based algorithm."""
        if not self._calibrated or not self.validator.is_valid(result):
            return None
        try:
            feats = self.appearance_extractor(frame, result).reshape(1, -1)
            gx = int(self.appearance_model_x.predict(feats)[0])
            gy = int(self.appearance_model_y.predict(feats)[0])
            return (gx, gy)
        except Exception:
            return None
    
    def predict_geometric(self, frame, result):
        """Get prediction from geometric algorithm."""
        if not self._calibrated or not self.validator.is_valid(result):
            return None
        try:
            feats = self.geometric_extractor(frame, result).reshape(1, -1)
            gx = int(self.geometric_model_x.predict(feats)[0])
            gy = int(self.geometric_model_y.predict(feats)[0])
            return (gx, gy)
        except Exception:
            return None
    
    def is_calibrated(self):
        return self._calibrated
    
    def get_sample_count(self):
        return len(self.calib_frames)
    
    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# ==============================================================================
# Camera Debug Window (for --dev mode)
# ==============================================================================

class CameraDebugWindow(QWidget):
    """Separate window showing camera feed with eye detection visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Eye Detection Debug")
        self.setFixedSize(680, 540)
        self.setStyleSheet("background-color: #1a1a1a;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.camera_view = QLabel()
        self.camera_view.setFixedSize(640, 480)
        self.camera_view.setStyleSheet("background-color: #333; border: 2px solid #666;")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setText("Waiting for camera...")
        layout.addWidget(self.camera_view, alignment=Qt.AlignCenter)
        
        self.status_label = QLabel("Status: ---")
        self.status_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
    
    def update_frame(self, frame, result):
        """Update camera view with eye detection visualization."""
        vis_frame = frame.copy()
        
        # Draw face bbox (yellow)
        if result and result.get('face_bbox') is not None:
            fx, fy, fw, fh = result['face_bbox']
            cv2.rectangle(vis_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
        
        # Draw eyes and iris
        if result:
            for eye_key, color in [('left_eye', (0, 255, 0)), ('right_eye', (0, 255, 0))]:
                eye = result.get(eye_key)
                if eye is not None:
                    # Eye bbox (green)
                    ex, ey, ew, eh = eye['bbox']
                    cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), color, 2)
                    
                    # Iris center (red)
                    iris_cx, iris_cy = eye['iris_center']
                    cv2.circle(vis_frame, (int(iris_cx), int(iris_cy)), 5, (0, 0, 255), -1)
        
        # Convert to Qt format
        try:
            h, w, ch = vis_frame.shape
            bytes_per_line = ch * w
            rgb_frame = vis_frame[:, :, ::-1].copy()
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_view.setPixmap(scaled_pixmap)
        except Exception as e:
            pass
        
        # Update status
        status = result.get('status', 'unknown') if result else 'no_frame'
        status_colors = {'ok': '#00ff00', 'no_face': '#ff0000', 'no_eyes': '#ff9900', 'partial': '#ffff00'}
        color = status_colors.get(status, '#ffffff')
        self.status_label.setText(f"Status: <span style='color:{color};'>{status.upper()}</span>")


# ==============================================================================
# Main Comparison App
# ==============================================================================

class ComparisonApp(QWidget):
    """Super simple comparison app - calibration then test dots."""
    
    def __init__(self, dev_mode=False):
        super().__init__()
        self.dev_mode = dev_mode
        self.setWindowTitle("Gaze Comparison - Resize window, then click first dot")
        
        # Initialize state BEFORE showing window (paintEvent/resizeEvent may be called)
        self.phase = "calibration"
        self.current_calib_idx = 0
        self.calib_points = []
        self.current_test_dot = None
        self.engine = None
        self.window_locked = False  # Window can be resized until first click
        self.cols = 5
        self.rows = 4
        self.screen_w = 800
        self.screen_h = 600
        
        # Camera debug window (dev mode only)
        self.camera_window = None
        if self.dev_mode:
            self.camera_window = CameraDebugWindow()
            self.camera_window.show()
            self.camera_window.move(50, 50)
        
        # Show maximized but resizable (not fullscreen)
        self.showMaximized()
        
        self.screen_w = self.width()
        self.screen_h = self.height()
        
        # Initialize engine (will update screen size on first click)
        try:
            self.engine = DualGazeEngine((self.screen_w, self.screen_h))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Camera error: {e}")
            sys.exit(1)
        
        # Calibration settings
        self.calib_points = self._generate_grid_points()
        self.window_ms = 1000
        
        # Test settings
        self.num_test_dots = 20
        self.test_dots_completed = 0
        self.results = []
        
        # Frame buffer for calibration
        self.frame_buffer = []
        
        # Timer for camera capture
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._capture_frame)
        self.camera_timer.start(33)
        
        self.update()
    
    def _generate_grid_points(self):
        margin_x = 0.03 * self.screen_w
        margin_y = 0.05 * self.screen_h
        usable_w = self.screen_w - 2 * margin_x
        usable_h = self.screen_h - 2 * margin_y
        
        points = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = int(margin_x + c * usable_w / (self.cols - 1)) if self.cols > 1 else int(margin_x)
                y = int(margin_y + r * usable_h / (self.rows - 1)) if self.rows > 1 else int(margin_y)
                points.append((x, y))
        return points
    
    def _generate_random_dot(self):
        margin = 100
        x = random.randint(margin, self.screen_w - margin)
        y = random.randint(margin, self.screen_h - margin)
        return (x, y)
    
    def _capture_frame(self):
        if not self.engine:
            return
        frame, result = self.engine.capture_and_detect()
        if frame is not None:
            timestamp = int(time.time() * 1000)
            self.frame_buffer.append((timestamp, frame, result))
            # Keep only last 2 seconds
            cutoff = timestamp - 2000
            self.frame_buffer = [(t, f, r) for t, f, r in self.frame_buffer if t >= cutoff]
            
            # Update debug window if in dev mode
            if self.dev_mode and self.camera_window:
                self.camera_window.update_frame(frame, result)
    
    def mousePressEvent(self, event):
        if not self.engine:
            return
        click_x, click_y = event.pos().x(), event.pos().y()
        
        if self.phase == "calibration":
            self._handle_calibration_click(click_x, click_y)
        elif self.phase == "test":
            self._handle_test_click(click_x, click_y)
    
    def _lock_window(self):
        """Lock window size - disable resizing."""
        self.window_locked = True
        self.setWindowTitle("Gaze Comparison")
        # Fix the window size
        self.setFixedSize(self.width(), self.height())
        # Update screen dimensions
        self.screen_w = self.width()
        self.screen_h = self.height()
        # Update engine screen size
        self.engine.screen_w = self.screen_w
        self.engine.screen_h = self.screen_h
        # Regenerate calibration points for new size
        self.calib_points = self._generate_grid_points()
        self.update()
    
    def _handle_calibration_click(self, click_x, click_y):
        if self.current_calib_idx >= len(self.calib_points):
            return
        
        # Lock window on first click
        if not self.window_locked:
            self._lock_window()
            return  # Don't process click, just lock - user needs to click again
        
        target = self.calib_points[self.current_calib_idx]
        dist = ((click_x - target[0])**2 + (click_y - target[1])**2)**0.5
        
        if dist > 50:
            return
        
        # Collect samples from last window_ms
        click_time = int(time.time() * 1000)
        window_start = click_time - self.window_ms
        
        samples_added = 0
        for ts, frame, result in self.frame_buffer:
            if window_start <= ts <= click_time:
                if self.engine.add_calibration_sample(target[0], target[1], frame, result):
                    samples_added += 1
        
        self.current_calib_idx += 1
        
        if self.current_calib_idx >= len(self.calib_points):
            self._finish_calibration()
        else:
            self.update()
    
    def _finish_calibration(self):
        success = self.engine.fit_models()
        sample_count = self.engine.get_sample_count()
        
        if success:
            QMessageBox.information(
                self, "OK",
                f"Calibration OK ({sample_count} samples)"
            )
            self.phase = "test"
            self.current_test_dot = self._generate_random_dot()
        else:
            QMessageBox.critical(
                self, "FAIL",
                f"Calibration failed ({sample_count} samples, need {self.engine.min_samples})"
            )
            self.close()
        
        self.update()
    
    def _handle_test_click(self, click_x, click_y):
        if self.current_test_dot is None:
            return
        
        dot_x, dot_y = self.current_test_dot
        dist = ((click_x - dot_x)**2 + (click_y - dot_y)**2)**0.5
        
        if dist > 40:
            return
        
        # Get predictions from both algorithms
        frame, result = self.engine.capture_and_detect()
        
        appearance_pred = None
        geometric_pred = None
        
        if frame is not None:
            appearance_pred = self.engine.predict_appearance(frame, result)
            geometric_pred = self.engine.predict_geometric(frame, result)
        
        # Store result
        self.results.append({
            'dot_x': dot_x,
            'dot_y': dot_y,
            'appearance_x': appearance_pred[0] if appearance_pred else None,
            'appearance_y': appearance_pred[1] if appearance_pred else None,
            'geometric_x': geometric_pred[0] if geometric_pred else None,
            'geometric_y': geometric_pred[1] if geometric_pred else None,
        })
        
        self.test_dots_completed += 1
        
        if self.test_dots_completed >= self.num_test_dots:
            self._finish_test()
        else:
            self.current_test_dot = self._generate_random_dot()
            self.update()
    
    def _finish_test(self):
        self.camera_timer.stop()
        self._save_results()
        QMessageBox.information(self, "Done", "Results saved to CSV")
        self.close()
    
    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dot_x', 'dot_y',
                'appearance_x', 'appearance_y', 'appearance_error',
                'geometric_x', 'geometric_y', 'geometric_error'
            ])
            
            for r in self.results:
                app_err = None
                geo_err = None
                
                if r['appearance_x'] is not None:
                    app_err = ((r['appearance_x'] - r['dot_x'])**2 + 
                               (r['appearance_y'] - r['dot_y'])**2)**0.5
                
                if r['geometric_x'] is not None:
                    geo_err = ((r['geometric_x'] - r['dot_x'])**2 + 
                               (r['geometric_y'] - r['dot_y'])**2)**0.5
                
                writer.writerow([
                    r['dot_x'], r['dot_y'],
                    r['appearance_x'], r['appearance_y'], 
                    f"{app_err:.1f}" if app_err else "N/A",
                    r['geometric_x'], r['geometric_y'],
                    f"{geo_err:.1f}" if geo_err else "N/A"
                ])
        
        # Print summary
        app_errors = [((r['appearance_x'] - r['dot_x'])**2 + (r['appearance_y'] - r['dot_y'])**2)**0.5 
                      for r in self.results if r['appearance_x'] is not None]
        geo_errors = [((r['geometric_x'] - r['dot_x'])**2 + (r['geometric_y'] - r['dot_y'])**2)**0.5 
                      for r in self.results if r['geometric_x'] is not None]
        
        print(f"\n{'='*50}")
        print(f"Results saved to: {filename}")
        print(f"{'='*50}")
        if app_errors:
            print(f"Appearance (Production): avg={np.mean(app_errors):.1f}px, std={np.std(app_errors):.1f}px")
        if geo_errors:
            print(f"Geometric (Demo2_v4):    avg={np.mean(geo_errors):.1f}px, std={np.std(geo_errors):.1f}px")
        print(f"{'='*50}\n")
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # White background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        if self.phase == "calibration":
            if self.calib_points and self.current_calib_idx < len(self.calib_points):
                x, y = self.calib_points[self.current_calib_idx]
                # Red calibration dot
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 0, 0))
                painter.drawEllipse(x - 10, y - 10, 20, 20)
        
        elif self.phase == "test":
            if self.current_test_dot:
                x, y = self.current_test_dot
                # Green test dot
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 200, 0))
                painter.drawEllipse(x - 12, y - 12, 24, 24)
                # White center
                painter.setBrush(QColor(255, 255, 255))
                painter.drawEllipse(x - 3, y - 3, 6, 6)
    
    def resizeEvent(self, event):
        """Update calibration points when window is resized (before lock)."""
        super().resizeEvent(event)
        if not self.window_locked:
            self.screen_w = self.width()
            self.screen_h = self.height()
            if self.cols and self.rows:  # Only if already initialized
                self.calib_points = self._generate_grid_points()
            self.update()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        if hasattr(self, 'camera_timer'):
            self.camera_timer.stop()
        if self.camera_window:
            self.camera_window.close()
        if self.engine:
            self.engine.close()
        event.accept()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaze Comparison App")
    parser.add_argument("--dev", action="store_true", help="Show camera debug window")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = ComparisonApp(dev_mode=args.dev)
    sys.exit(app.exec_())

