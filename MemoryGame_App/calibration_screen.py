import sys
import time
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap

sys.path.append("..")
sys.path.append("../GazeLocalization")
sys.path.append("../eye-detection-final")
from gaze_localizator import GazeEngine


class CalibrationScreen(QWidget):
    """
    Calibration screen for eye tracking.
    Displays 20 calibration points (5x4 grid) and collects samples.
    Based on gaze_localization_demo3_v1.py approach.
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
        self.margin_x_ratio = 0.03  # 3% from edge
        self.margin_y_ratio = 0.03  # 3% from edge
        
        # Safe zone in center where text will be displayed (avoid overlap)
        # Text will only appear in this central region
        self.text_zone_x_start = 0.25  # 25% from left
        self.text_zone_x_end = 0.75    # 75% from left
        self.text_zone_y_start = 0.35  # 35% from top
        self.text_zone_y_end = 0.65    # 65% from top

        # State
        self.calibration_points = []
        self.current_point_index = 0
        self.frame_buffer = []  # List of (timestamp_ms, frame, result)
        self.collection_start_time = None
        self.is_collecting = False
        self.calibration_complete = False
        self.calibration_success = False
        self.last_samples_collected = 0
        
        # Camera frame for dev mode display
        self.current_frame = None
        self.current_result = None

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
        margin_y = self.margin_y_ratio * self.screen_h

        usable_w = self.screen_w - 2 * margin_x
        usable_h = self.screen_h - 2 * margin_y

        points = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = int(margin_x + c * usable_w / (self.cols - 1) if self.cols > 1 else margin_x)
                y = int(margin_y + r * usable_h / (self.rows - 1) if self.rows > 1 else margin_y)
                points.append((x, y))

        self.calibration_points = points

    def _is_point_in_text_zone(self, x, y):
        """Check if a point is in the central text zone."""
        x_ratio = x / self.screen_w
        y_ratio = y / self.screen_h
        return (self.text_zone_x_start <= x_ratio <= self.text_zone_x_end and
                self.text_zone_y_start <= y_ratio <= self.text_zone_y_end)

    def _build_ui(self):
        """Build UI with text in center, avoiding calibration points."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left spacer (for dev mode camera view)
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(0)  # Hidden by default
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        
        if self.dev_mode:
            self.left_panel.setFixedWidth(350)
            self.left_panel.setStyleSheet("background-color: #1a1a1a;")
            
            # Camera view label
            self.camera_label = QLabel("Camera View")
            self.camera_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
            self.camera_label.setAlignment(Qt.AlignCenter)
            self.left_layout.addWidget(self.camera_label)
            
            # Camera image display
            self.camera_view = QLabel()
            self.camera_view.setFixedSize(320, 240)
            self.camera_view.setStyleSheet("background-color: #333; border: 2px solid #666;")
            self.camera_view.setAlignment(Qt.AlignCenter)
            self.left_layout.addWidget(self.camera_view)
            
            # Status info
            self.status_label = QLabel("Status: Initializing...")
            self.status_label.setStyleSheet("color: #aaa; font-size: 12px;")
            self.left_layout.addWidget(self.status_label)
            
            # Dev info
            self.dev_info_label = QLabel("DEV MODE ACTIVE")
            self.dev_info_label.setStyleSheet("color: #ff6600; font-size: 14px; font-weight: bold;")
            self.dev_info_label.setAlignment(Qt.AlignCenter)
            self.left_layout.addWidget(self.dev_info_label)
            
            self.left_layout.addStretch()
        
        main_layout.addWidget(self.left_panel)
        
        # Center content area
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add stretch to push content to center
        center_layout.addStretch(1)

        # Container for centered text (will be in the safe zone)
        text_container = QVBoxLayout()
        text_container.setAlignment(Qt.AlignCenter)
        text_container.setSpacing(15)

        # Title
        self.title_label = QLabel("Eye Tracking Calibration", alignment=Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 32px; 
            font-weight: 700; 
            color: #4B2C82;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px 20px;
            border-radius: 10px;
        """)
        text_container.addWidget(self.title_label)

        # Instructions
        self.instructions_label = QLabel(
            f"Point {self.current_point_index + 1}/{self.num_points}\n\n"
            "Look at the red dot and click on it.",
            alignment=Qt.AlignCenter
        )
        self.instructions_label.setStyleSheet("""
            font-size: 18px; 
            color: #333;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 15px 25px;
            border-radius: 10px;
        """)
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setMaximumWidth(400)
        text_container.addWidget(self.instructions_label)

        # Progress label (only visible in dev mode)
        self.progress_label = QLabel("", alignment=Qt.AlignCenter)
        self.progress_label.setStyleSheet("""
            font-size: 14px; 
            color: #666;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 8px 15px;
            border-radius: 8px;
        """)
        if not self.dev_mode:
            self.progress_label.hide()
        text_container.addWidget(self.progress_label)

        # Add text container to center layout
        center_layout.addLayout(text_container)
        
        # Add stretch to center vertically
        center_layout.addStretch(1)
        
        main_layout.addWidget(center_widget, 1)
        
        layout.addLayout(main_layout)

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
            
            # Store for dev mode display
            self.current_frame = frame.copy()
            self.current_result = result
            
            # Update dev mode camera view
            if self.dev_mode:
                self._update_camera_view(frame, result)

            # Keep only recent frames (last 2 seconds)
            cutoff = timestamp_ms - 2000
            self.frame_buffer = [(ts, f, r) for ts, f, r in self.frame_buffer if ts >= cutoff]

    def _update_camera_view(self, frame, result):
        """Update camera view in dev mode with eye detection visualization."""
        if not self.dev_mode:
            return
            
        vis_frame = frame.copy()
        
        # Draw face bounding box
        if result and result.get('face_bbox') is not None:
            fx, fy, fw, fh = result['face_bbox']
            cv2 = self._get_cv2()
            if cv2:
                cv2.rectangle(vis_frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
        
        # Draw eyes
        if result:
            cv2 = self._get_cv2()
            if cv2:
                for eye_key in ['left_eye', 'right_eye']:
                    eye = result.get(eye_key)
                    if eye is not None:
                        ex, ey, ew, eh = eye['bbox']
                        cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        
                        # Iris center
                        iris_cx, iris_cy = eye['iris_center']
                        cv2.circle(vis_frame, (iris_cx, iris_cy), 4, (0, 0, 255), -1)
                        cv2.circle(vis_frame, (iris_cx, iris_cy), 6, (0, 0, 255), 2)
        
        # Convert to QImage for display
        try:
            h, w, ch = vis_frame.shape
            bytes_per_line = ch * w
            # OpenCV uses BGR, Qt uses RGB
            rgb_frame = vis_frame[:, :, ::-1].copy()
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_view.setPixmap(scaled_pixmap)
            
            # Update status
            status = result.get('status', 'unknown') if result else 'no_frame'
            status_colors = {
                'ok': '#00ff00',
                'no_face': '#ff0000',
                'no_eyes': '#ff9900',
                'partial': '#ffff00'
            }
            color = status_colors.get(status, '#ffffff')
            self.status_label.setText(f"Status: <span style='color:{color}'>{status.upper()}</span>")
            self.status_label.setStyleSheet("color: #aaa; font-size: 12px;")
        except Exception as e:
            if self.dev_mode:
                print(f"Error updating camera view: {e}")

    def _get_cv2(self):
        """Lazy import of cv2."""
        try:
            import cv2
            return cv2
        except ImportError:
            return None

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
        
        # Only show sample count in dev mode
        if self.dev_mode:
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

        if success:
            self.title_label.setText("Calibration Complete!")
            if self.dev_mode:
                self.instructions_label.setText(
                    f"Successfully collected {len(self.gaze_engine.calib_X)} samples.\n"
                    "Models have been trained.\n\n"
                    "You can now proceed to the game."
                )
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
                    f"Only {len(self.gaze_engine.calib_X)} samples collected.\n"
                    f"Minimum required: {self.min_samples}\n\n"
                    "Please try again."
                )
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

        # Draw current calibration point - larger red dot for visibility
        if self.current_point_index < len(self.calibration_points):
            point = self.calibration_points[self.current_point_index]
            x, y = point

            # Outer glow effect
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0, 80))
            painter.drawEllipse(x - 20, y - 20, 40, 40)
            
            # Middle ring
            painter.setBrush(QColor(255, 0, 0, 150))
            painter.drawEllipse(x - 12, y - 12, 24, 24)
            
            # Inner solid dot
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(x - 6, y - 6, 12, 12)
            
            # Center white dot for precision
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(x - 2, y - 2, 4, 4)

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
