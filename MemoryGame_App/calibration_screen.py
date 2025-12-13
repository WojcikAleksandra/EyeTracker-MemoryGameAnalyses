import sys
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
from PyQt5.QtGui import QPainter, QPen, QColor

sys.path.append("..")
sys.path.append("../GazeLocalization")
from gaze_localizator import GazeEngine


class CalibrationScreen(QWidget):
    """
    Calibration screen for eye tracking.
    Displays 20 calibration points (5x4 grid) and collects samples.
    Based on gaze_localization_demo3_v1.py approach.
    """

    def __init__(self, screen_size, parent=None):
        super().__init__(parent)
        # Use parent window size if screen_size is not valid
        if parent and (screen_size[0] <= 0 or screen_size[1] <= 0):
            self.screen_w, self.screen_h = parent.width(), parent.height()
        else:
            self.screen_w, self.screen_h = screen_size
        
        if self.screen_w <= 0 or self.screen_h <= 0:
            self.screen_w, self.screen_h = 1920, 1080  # Default fallback
        
        self.setMinimumSize(self.screen_w, self.screen_h)

        # Calibration parameters
        self.num_points = 20
        self.cols = 5
        self.rows = 4
        self.window_ms = 1000  # Time window for collecting samples before click
        self.min_samples = 60  # Minimum samples needed for training

        # State
        self.calibration_points = []
        self.current_point_index = 0
        self.frame_buffer = []  # List of (timestamp_ms, frame, result)
        self.collection_start_time = None
        self.is_collecting = False
        self.calibration_complete = False
        self.calibration_success = False

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
        """Generate 20 calibration points in a 5x4 grid."""
        margin_x = 0.10 * self.screen_w  # Increased margin
        margin_y = 0.15 * self.screen_h  # Increased margin to prevent bottom points from being too low

        usable_w = self.screen_w - 2 * margin_x
        usable_h = self.screen_h - 2 * margin_y

        points = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = int(margin_x + c * usable_w / (self.cols - 1) if self.cols > 1 else margin_x)
                y = int(margin_y + r * usable_h / (self.rows - 1) if self.rows > 1 else margin_y)
                points.append((x, y))

        self.calibration_points = points

    def _build_ui(self):
        # Use a centered layout that won't overlap with calibration points
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
        self.title_label.setStyleSheet("font-size: 36px; font-weight: 700; color: #4B2C82;")
        text_container.addWidget(self.title_label)

        # Instructions
        self.instructions_label = QLabel(
            f"Point {self.current_point_index + 1}/{self.num_points}\n\n"
            "Look at the red dot and click on it when you are ready.\n"
            "Make sure your face is clearly visible to the camera.",
            alignment=Qt.AlignCenter
        )
        self.instructions_label.setStyleSheet("font-size: 20px; color: #333;")
        self.instructions_label.setWordWrap(True)
        text_container.addWidget(self.instructions_label)

        # Progress
        self.progress_label = QLabel("", alignment=Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 18px; color: #666;")
        text_container.addWidget(self.progress_label)

        # Add text container to main layout (centered)
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
        self.is_collecting = False
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

        self.progress_label.setText(
            f"Point {self.current_point_index + 1}/{self.num_points}: "
            f"{samples_collected} samples collected"
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
                "Look at the red dot and click on it when you are ready.\n"
                "Make sure your face is clearly visible to the camera."
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

        if success:
            self.title_label.setText("Calibration Complete!")
            self.instructions_label.setText(
                f"Successfully collected {len(self.gaze_engine.calib_X)} samples.\n"
                "Models have been trained.\n\n"
                "You can now proceed to the game."
            )
            self.progress_label.setText("")
        else:
            self.title_label.setText("Calibration Failed")
            self.instructions_label.setText(
                f"Only {len(self.gaze_engine.calib_X)} samples collected.\n"
                f"Minimum required: {self.min_samples}\n\n"
                "Please try again."
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

            # Draw simple red circle (dot)
            painter.setPen(Qt.NoPen)  # No border
            painter.setBrush(QColor(255, 0, 0))  # Red fill
            painter.drawEllipse(x - 8, y - 8, 16, 16)  # 16px diameter dot

    def get_gaze_engine(self):
        """Get the GazeEngine instance (or None if failed)."""
        return self.gaze_engine if self.calibration_success else None

    def is_successful(self):
        """Check if calibration was successful."""
        return self.calibration_success

    def closeEvent(self, event):
        """Clean up on close."""
        self.camera_timer.stop()
        if self.gaze_engine:
            self.gaze_engine.close()
        event.accept()

