import sys
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QColor, QImage, QPixmap

from MemoryGame.algorithms.gaze_localizator import GazeEngine


class CameraDebugWindow(QWidget):
    """Separate window showing camera feed with eye detection visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Eye Detection - DEV MODE")
        self.setFixedSize(680, 540)
        self.setStyleSheet("background-color: #1a1a1a;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Camera View - Eye Detection")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Camera image display
        self.camera_view = QLabel()
        self.camera_view.setFixedSize(640, 480)
        self.camera_view.setStyleSheet("background-color: #333; border: 2px solid #666;")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setText("Waiting for camera...")
        layout.addWidget(self.camera_view, alignment=Qt.AlignCenter)
        
        # Status info
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("color: #aaa; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Sample count
        self.sample_label = QLabel("Samples: 0")
        self.sample_label.setStyleSheet("color: #ff6600; font-size: 14px; font-weight: bold;")
        self.sample_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sample_label)
        
        # Legend
        legend = QLabel(
            "Legend: YELLOW = Face | GREEN = Eyes | RED = Iris Center"
        )
        legend.setStyleSheet("color: #888; font-size: 12px;")
        legend.setAlignment(Qt.AlignCenter)
        layout.addWidget(legend)
        
    def update_frame(self, frame, result):
        """Update camera view with eye detection visualization."""
        cv2 = self._get_cv2()
        if cv2 is None:
            return
            
        vis_frame = frame.copy()
        
        # Draw face bounding box (yellow)
        if result and result.get('face_bbox') is not None:
            fx, fy, fw, fh = result['face_bbox']
            cv2.rectangle(vis_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
            cv2.putText(vis_frame, "Face", (fx, fy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw eyes
        if result:
            for eye_key, label in [('left_eye', 'L'), ('right_eye', 'R')]:
                eye = result.get(eye_key)
                if eye is not None:
                    ex, ey, ew, eh = eye['bbox']
                    # Eye box (green)
                    cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"{label} Eye", (ex, ey - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Iris center (red)
                    iris_cx, iris_cy = eye['iris_center']
                    cv2.circle(vis_frame, (iris_cx, iris_cy), 5, (0, 0, 255), -1)
                    cv2.circle(vis_frame, (iris_cx, iris_cy), 8, (0, 0, 255), 2)
                    
                    # Iris bbox (blue)
                    ix, iy, iw, ih = eye['iris_bbox']
                    cv2.rectangle(vis_frame, (ix, iy), (ix + iw, iy + ih), (255, 0, 0), 1)
        
        # Convert to QImage for display
        try:
            h, w, ch = vis_frame.shape
            bytes_per_line = ch * w
            # OpenCV uses BGR, Qt uses RGB
            rgb_frame = vis_frame[:, :, ::-1].copy()
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_view.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error updating camera view: {e}")
    
    def update_status(self, status):
        """Update detection status display."""
        status_colors = {
            'ok': '#00ff00',
            'no_face': '#ff0000',
            'no_eyes': '#ff9900',
            'partial': '#ffff00'
        }
        color = status_colors.get(status, '#ffffff')
        self.status_label.setText(
            f"Detection Status: <span style='color:{color}; font-weight:bold;'>{status.upper()}</span>"
        )
    
    def update_sample_count(self, current_point, total_points, samples):
        """Update sample count display."""
        self.sample_label.setText(
            f"Point {current_point}/{total_points} | Last samples: {samples} | "
            f"Total collected: calculating..."
        )
    
    def update_total_samples(self, total):
        """Update total sample count."""
        self.sample_label.setText(f"Total samples collected: {total}")
    
    def _get_cv2(self):
        """Lazy import of cv2."""
        try:
            import cv2
            return cv2
        except ImportError:
            return None


class CalibrationScreen(QWidget):
    """
    Calibration screen for eye tracking.
    Displays 20 calibration points (5x4 grid) and collects samples.
    """

    def __init__(self, screen_size, parent=None, dev_mode=False):
        super().__init__(parent)
        self.dev_mode = dev_mode
        
        # Use parent window size if screen_size is not valid
        if parent and (screen_size[0] <= 0 or screen_size[1] <= 0):
            self.screen_w, self.screen_h = parent.width(), parent.height()
        else:
            self.screen_w, self.screen_h = screen_size
        
        if self.screen_w <= 0 or self.screen_h <= 0:
            self.screen_w, self.screen_h = 1920, 1080  # Default fallback
        
        self.setMinimumSize(self.screen_w, self.screen_h)

        # Calibration parameters - reduced margins for better edge coverage
        self.num_points = 20
        self.cols = 5
        self.rows = 4
        self.window_ms = 1000  # Time window for collecting samples before click
        self.min_samples = 60  # Minimum samples needed for training
        
        # Margins for calibration points (very close to edges)
        self.margin_x_ratio = 0.03  # 3% from left/right edge
        self.margin_top_ratio = 0.03  # 3% from top
        self.margin_bottom_ratio = 0.08  # 8% from bottom (higher to avoid taskbar/cutoff)

        # State
        self.calibration_points = []
        self.current_point_index = 0
        self.frame_buffer = []  # List of (timestamp_ms, frame, result)
        self.calibration_complete = False
        self.calibration_success = False
        self.last_samples_collected = 0
        
        # Dev mode camera window
        self.camera_window = None
        if self.dev_mode:
            self.camera_window = CameraDebugWindow()
            self.camera_window.show()
            # Position camera window to the side
            self.camera_window.move(50, 50)

        # GazeEngine
        try:
            self.gaze_engine = GazeEngine(
                screen_size=screen_size,
                model_type="ridge",
                patch_height=8,
                patch_width=9,
                min_samples=self.min_samples,
            )
            self.gaze_engine.start_calibration()
        except Exception as e:
            print(f"Error initializing GazeEngine: {e}")
            self.gaze_engine = None
            QMessageBox.critical(
                self,
                "Camera Error",
                f"Could not initialize camera:\n{str(e)}\n\nGame will continue without gaze tracking."
            )

        # Timers
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._capture_frame)

        self._generate_calibration_points()
        self._build_ui()

    def _generate_calibration_points(self):
        """Generate 20 calibration points in a 5x4 grid with minimal margins."""
        margin_x = self.margin_x_ratio * self.screen_w
        margin_top = self.margin_top_ratio * self.screen_h
        margin_bottom = self.margin_bottom_ratio * self.screen_h

        usable_w = self.screen_w - 2 * margin_x
        usable_h = self.screen_h - margin_top - margin_bottom

        points = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = int(margin_x + c * usable_w / (self.cols - 1) if self.cols > 1 else margin_x)
                y = int(margin_top + r * usable_h / (self.rows - 1) if self.rows > 1 else margin_top)
                points.append((x, y))

        self.calibration_points = points

    def _build_ui(self):
        """Build UI with text in center."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add stretch to push content to center
        layout.addStretch(1)

        # Container for centered text
        text_container = QVBoxLayout()
        text_container.setAlignment(Qt.AlignCenter)
        text_container.setSpacing(15)

        # Title
        self.title_label = QLabel("Eye Tracking Calibration", alignment=Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 32px; font-weight: 700; color: #4B2C82;")
        text_container.addWidget(self.title_label, alignment=Qt.AlignCenter)

        # Instructions
        self.instructions_label = QLabel(
            f"Point {self.current_point_index + 1}/{self.num_points}\n\n"
            "Look at the red dot and click on it.",
            alignment=Qt.AlignCenter
        )
        self.instructions_label.setStyleSheet("font-size: 18px; color: #333;")
        self.instructions_label.setWordWrap(True)
        text_container.addWidget(self.instructions_label, alignment=Qt.AlignCenter)

        # Progress label (only visible in dev mode)
        self.progress_label = QLabel("", alignment=Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14px; color: #666;")
        if not self.dev_mode:
            self.progress_label.hide()
        text_container.addWidget(self.progress_label, alignment=Qt.AlignCenter)

        # Add text container to layout
        layout.addLayout(text_container)
        
        # Add stretch to center vertically
        layout.addStretch(1)

    def start_calibration(self):
        """Start the calibration process."""
        if not self.gaze_engine:
            self.calibration_complete = True
            self.calibration_success = False
            return

        self.current_point_index = 0
        self.frame_buffer = []
        self.calibration_complete = False
        self.calibration_success = False

        self.camera_timer.start(33)  # ~30 FPS
        self.update()

    def _capture_frame(self):
        """Capture frame from camera and detect eyes."""
        if not self.gaze_engine or self.calibration_complete:
            return

        frame, result = self.gaze_engine.capture_and_detect()
        if frame is not None:
            timestamp_ms = int(time.time() * 1000)
            self.frame_buffer.append((timestamp_ms, frame.copy(), result))
            
            # Update dev mode camera window
            if self.dev_mode and self.camera_window:
                self.camera_window.update_frame(frame, result)
                status = result.get('status', 'unknown') if result else 'no_frame'
                self.camera_window.update_status(status)

            # Keep only recent frames (last 2 seconds)
            cutoff = timestamp_ms - 2000
            self.frame_buffer = [(ts, f, r) for ts, f, r in self.frame_buffer if ts >= cutoff]

    def mousePressEvent(self, event):
        """Handle mouse click on calibration point."""
        if self.calibration_complete or not self.gaze_engine:
            return

        click_pos = (event.pos().x(), event.pos().y())
        current_point = self.calibration_points[self.current_point_index]

        # Check if click is near the calibration point (within 50 pixels)
        dist = ((click_pos[0] - current_point[0]) ** 2 + (click_pos[1] - current_point[1]) ** 2) ** 0.5
        if dist > 50:
            if self.dev_mode:
                self.progress_label.setText("Please click on the red dot!")
            return

        # Start collecting samples from the time window
        click_time_ms = int(time.time() * 1000)
        window_start = click_time_ms - self.window_ms

        # Get frames from the time window
        window_frames = [
            (frm, r) for (ts, frm, r) in self.frame_buffer
            if window_start <= ts <= click_time_ms
        ]

        # Collect valid samples
        samples_collected = 0
        for frame, result in window_frames:
            if self.gaze_engine.add_calibration_sample(
                current_point[0], current_point[1], frame, result
            ):
                samples_collected += 1

        self.last_samples_collected = samples_collected
        
        # Update dev mode displays
        if self.dev_mode:
            self.progress_label.setText(
                f"Point {self.current_point_index + 1}/{self.num_points}: "
                f"{samples_collected} samples collected"
            )
            if self.camera_window:
                self.camera_window.update_sample_count(
                    self.current_point_index + 1, 
                    self.num_points, 
                    samples_collected
                )

        # Move to next point
        self.current_point_index += 1
        if self.current_point_index >= self.num_points:
            # All points collected, train models
            self._finish_calibration()
        else:
            # Update UI for next point
            self.instructions_label.setText(
                f"Point {self.current_point_index + 1}/{self.num_points}\n\n"
                "Look at the red dot and click on it."
            )
            self.update()

    def _finish_calibration(self):
        """Finish calibration and train models."""
        self.camera_timer.stop()
        self.calibration_complete = True

        if not self.gaze_engine:
            self.calibration_success = False
            return

        # Train models
        success = self.gaze_engine.fit_models()
        self.calibration_success = success
        
        total_samples = len(self.gaze_engine.calib_X)

        if success:
            self.title_label.setText("Calibration Complete!")
            if self.dev_mode:
                self.instructions_label.setText(
                    f"Successfully collected {total_samples} samples.\n"
                    "Models have been trained.\n\n"
                    "You can now proceed to the game."
                )
                if self.camera_window:
                    self.camera_window.update_total_samples(total_samples)
            else:
                self.instructions_label.setText(
                    "Calibration successful!\n\n"
                    "You can now proceed to the game."
                )
            self.progress_label.setText("")
        else:
            self.title_label.setText("Calibration Failed")
            if self.dev_mode:
                self.instructions_label.setText(
                    f"Only {total_samples} samples collected.\n"
                    f"Minimum required: {self.min_samples}\n\n"
                    "Please try again."
                )
                if self.camera_window:
                    self.camera_window.update_total_samples(total_samples)
            else:
                self.instructions_label.setText(
                    "Not enough data collected.\n\n"
                    "Please try again with better lighting\n"
                    "and face visibility."
                )
            self.progress_label.setText("")

        self.update()

    def _cancel_calibration(self):
        """Cancel calibration process."""
        self.camera_timer.stop()
        self.calibration_complete = True
        self.calibration_success = False

    def paintEvent(self, event):
        """Draw calibration point."""
        super().paintEvent(event)

        if self.calibration_complete:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw current calibration point - simple red dot
        if self.current_point_index < len(self.calibration_points):
            point = self.calibration_points[self.current_point_index]
            x, y = point

            # Simple red circle
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(x - 8, y - 8, 16, 16)

    def get_gaze_engine(self):
        """Get the GazeEngine instance (or None if failed)."""
        return self.gaze_engine if self.calibration_success else None

    def is_successful(self):
        """Check if calibration was successful."""
        return self.calibration_success

    def closeEvent(self, event):
        """Clean up on close."""
        self.camera_timer.stop()
        if self.camera_window:
            self.camera_window.close()
        # Don't close gaze_engine here if calibration succeeded - it gets transferred
        # to the main window for use during the game. Only close if calibration failed.
        if self.gaze_engine and not self.calibration_success:
            self.gaze_engine.close()
        event.accept()
