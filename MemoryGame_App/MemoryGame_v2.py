import sys
import random
import os
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from sklearn.linear_model import Ridge
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QPushButton, QLabel, QFrame,
    QComboBox, QAction, QMessageBox
)
from PyQt5.QtCore import Qt, QSize, QTime, QTimer, QPoint, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPainter, QPen

# Import eye tracking modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eye-detection-final"))
from eye_detector import EyeDetector


# ========================= #
#       Style Constants     #
# ========================= #
class Styles:
    BUTTON = """
        QPushButton {
            background-color: #8549c9;
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 22px;
            padding: 12px 24px;
        }
        QPushButton:hover { background-color: #7239b5; }
        QPushButton:pressed { background-color: #5e2f99; }
    """

    MENU_BAR = """
        QMenuBar {
            background-color: #E0E0E0;
            color: #222;
            font-size: 16px;
            padding: 4px 8px;
        }
        QMenuBar::item:selected { background-color: #D0D0D0; }
    """

    TITLE = """
        font-size: 72px;
        font-weight: 800;
        color: #8549c9;
        margin-top: 10px;
        letter-spacing: 2px;
    """

    HUD = "font-size: 18px; color: #333;"
    SUBTITLE = "font-size: 26px; font-weight: 600; color: #8549c9;"

    CARD_BUTTON = """
        QPushButton {
            border: 3px solid #ceb5e7;
            border-radius: 12px;
            background-color: #ceb5e7;
            padding: 0px;
        }
        QPushButton:hover {
            background-color: #8549c9;
            border-color: #8549c9;
        }
    """

    FRAME = """
        QFrame {
            border: 4px solid #B68DDE;
            border-radius: 16px;
            background-color: #fdfcff;
            padding: 5px;
        }
    """

    LIGHT_FRAME = """
        QFrame {
            background-color: #EEEEEE;
            border-radius: 10px;
            padding: 20px;
        }
    """

    PLOT_BOX = """
        QFrame {
            background-color: #E0E0E0;
            border-radius: 12px;
            border: 2px solid #CCCCCC;
        }
    """

    CALIB_POINT = """
        QPushButton {
            background-color: #FF0000;
            border-radius: 15px;
            border: 3px solid #8B0000;
        }
        QPushButton:hover {
            background-color: #CC0000;
        }
    """


# ========================= #
#   Eye Tracking Components #
# ========================= #

class GazeFeatureExtractor:
    """Extract features from eye patches for gaze estimation."""
    
    def __init__(self, patch_height: int = 10, patch_width: int = 10):
        self.patch_height = patch_height
        self.patch_width = patch_width
    
    def _extract_eye_features(self, frame: np.ndarray, eye_bbox) -> np.ndarray:
        x_global, y_global, eye_w, eye_h = eye_bbox
        eye_region = frame[y_global:y_global + eye_h, x_global:x_global + eye_w]
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        patch_resized = cv2.resize(eye_gray, (self.patch_width, self.patch_height), interpolation=cv2.INTER_AREA)
        patch_norm = patch_resized.astype(np.float32) / 255.0
        return patch_norm.flatten()
    
    def __call__(self, frame: np.ndarray, result: dict) -> np.ndarray:
        left = result["left_eye"]
        right = result["right_eye"]
        left_feats = self._extract_eye_features(frame, left["bbox"])
        right_feats = self._extract_eye_features(frame, right["bbox"])
        features = np.concatenate([left_feats, right_feats], axis=0)
        return features.astype(np.float32)


class EyeFrameValidator:
    """Validate if frame has valid eye detection."""
    
    def is_valid_frame(self, result: dict) -> bool:
        if result is None:
            return False
        if result.get("face_bbox") is None:
            return False
        left = result.get("left_eye")
        right = result.get("right_eye")
        if left is None or right is None:
            return False
        return True


class CameraThread(QThread):
    """Background thread for camera capture."""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
    
    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(16)  # ~60 FPS
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ========================= #
#     Calibration Screen    #
# ========================= #

class CalibrationScreen(QWidget):
    """PyQt5-based calibration screen for eye tracking."""
    calibration_complete = pyqtSignal(object, object)  # model_x, model_y
    calibration_failed = pyqtSignal()
    
    def __init__(self, detector, feature_extractor, validator, screen_size):
        super().__init__()
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.validator = validator
        self.screen_w, self.screen_h = screen_size
        
        self.window_ms = 1000
        self.cols = 5
        self.rows = 4
        self.min_samples = 60
        
        self.current_frame = None
        self.frame_buffer = []
        self.calibration_points = []
        self.current_point_idx = 0
        self.waiting_for_click = False
        
        self.X = []
        self.y_x = []
        self.y_y = []
        
        self.model_x = Ridge(alpha=1.0)
        self.model_y = Ridge(alpha=1.0)
        
        self._build_ui()
        
        # Camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.on_frame)
        
        # Timer for frame buffer cleanup
        self.buffer_timer = QTimer()
        self.buffer_timer.timeout.connect(self._cleanup_buffer)
        self.buffer_timer.start(100)
    
    def _build_ui(self):
        self.setStyleSheet("background-color: white;")
        
        # Create labels positioned at bottom-right corner (won't overlap with points)
        self.info_label = QLabel(self)
        self.info_label.setText("Calibration: Click on red points")
        self.info_label.setStyleSheet("""
            font-size: 16px; 
            color: #333; 
            padding: 10px 15px;
            background-color: rgba(240, 240, 240, 230);
            border-radius: 5px;
        """)
        self.info_label.setAlignment(Qt.AlignCenter)
        
        self.point_label = QLabel(self)
        self.point_label.setText("0/20")
        self.point_label.setStyleSheet("""
            font-size: 20px; 
            color: #8549c9;
            font-weight: bold;
            padding: 8px 12px;
            background-color: rgba(240, 240, 240, 230);
            border-radius: 5px;
        """)
        self.point_label.setAlignment(Qt.AlignCenter)
    
    def _generate_calibration_points(self):
        """Generate calibration points in widget-local coordinates."""
        points = []
        # Increased margins to keep points away from edges
        margin_x = 0.04 * self.screen_w  # 8% margin on sides
        margin_y = 0.10 * self.screen_h  # 10% margin on top/bottom
        usable_w = self.screen_w - 2 * margin_x
        usable_h = self.screen_h - 2 * margin_y
        
        for r in range(self.rows):
            for c in range(self.cols):
                x = int(margin_x + c * usable_w / (self.cols - 1))
                y = int(margin_y + r * usable_h / (self.rows - 1))
                points.append((x, y))
        return points
    
    def start_calibration(self):
        self.calibration_points = self._generate_calibration_points()
        self.current_point_idx = 0
        self.waiting_for_click = True
        self.X = []
        self.y_x = []
        self.y_y = []
        self.frame_buffer = []
        
        self.camera_thread.start()
        self._update_info()
    
    def on_frame(self, frame):
        self.current_frame = frame.copy()
        timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
        result = self.detector.detect(frame)
        self.frame_buffer.append((timestamp_ms, frame.copy(), result))
    
    def _cleanup_buffer(self):
        """Keep only recent frames in buffer."""
        if len(self.frame_buffer) > 100:
            self.frame_buffer = self.frame_buffer[-100:]
    
    def _update_info(self):
        if self.current_point_idx < len(self.calibration_points):
            self.info_label.setText(f"Look at the red point and click on it")
            self.point_label.setText(f"Point {self.current_point_idx + 1}/{len(self.calibration_points)}")
        self.update()
    
    def mousePressEvent(self, event):
        if not self.waiting_for_click:
            return
        
        if self.current_point_idx >= len(self.calibration_points):
            return
        
        # Local coordinates (widget coordinates)
        click_x_local = event.pos().x()
        click_y_local = event.pos().y()
        target_x_local, target_y_local = self.calibration_points[self.current_point_idx]
        
        # Check if click is close to target (in local coordinates)
        dist = np.hypot(click_x_local - target_x_local, click_y_local - target_y_local)
        
        if dist > 50:
            return  # Click too far from point
        
        # Convert to global screen coordinates for training data
        global_click = self.mapToGlobal(event.pos())
        global_target = self.mapToGlobal(QPoint(target_x_local, target_y_local))
        
        click_time_ms = int(QTime.currentTime().msecsSinceStartOfDay())
        window_start = click_time_ms - self.window_ms
        
        # Collect frames from window
        window_results = [
            (frm, r) for (ts, frm, r) in self.frame_buffer
            if window_start <= ts <= click_time_ms
        ]
        
        # Extract valid features
        valid_features = []
        for frm, r in window_results:
            if self.validator.is_valid_frame(r):
                feats = self.feature_extractor(frm, r)
                valid_features.append(feats)
        
        if len(valid_features) > 0:
            for feats in valid_features:
                self.X.append(feats)
                # Store global screen coordinates for training
                self.y_x.append(float(global_target.x()))
                self.y_y.append(float(global_target.y()))
        
        self.current_point_idx += 1
        
        if self.current_point_idx >= len(self.calibration_points):
            self._finish_calibration()
        else:
            self._update_info()
    
    def _finish_calibration(self):
        self.waiting_for_click = False
        self.camera_thread.stop()
        self.camera_thread.wait()
        
        if len(self.X) < self.min_samples:
            self.calibration_failed.emit()
            return
        
        X = np.asarray(self.X, dtype=np.float32)
        y_x = np.asarray(self.y_x, dtype=np.float32)
        y_y = np.asarray(self.y_y, dtype=np.float32)
        
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)
        
        self.calibration_complete.emit(self.model_x, self.model_y)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_point_idx >= len(self.calibration_points):
            return
        
        if len(self.calibration_points) == 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw calibration point - simple red dot
        target_x, target_y = self.calibration_points[self.current_point_idx]
        center = QPoint(target_x, target_y)
        
        # Simple solid red circle
        painter.setBrush(Qt.red)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, 10, 10)
        
        # Position labels at center of screen (both horizontally and vertically)
        info_width = self.info_label.sizeHint().width()
        info_height = self.info_label.sizeHint().height()
        point_width = self.point_label.sizeHint().width()
        point_height = self.point_label.sizeHint().height()
        
        # Calculate total height of both labels
        label_spacing = 10
        total_height = info_height + label_spacing + point_height
        
        # Center both horizontally and vertically
        # Position info label
        self.info_label.setGeometry(
            (self.width() - info_width) // 2,
            (self.height() - total_height) // 2,
            info_width,
            info_height
        )
        
        # Position point counter below info label
        self.point_label.setGeometry(
            (self.width() - point_width) // 2,
            (self.height() - total_height) // 2 + info_height + label_spacing,
            point_width,
            point_height
        )
    
    def closeEvent(self, event):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        super().closeEvent(event)


# ========================= #
#     Memory Game Board     #
# ========================= #
class MemoryGameBoard(QWidget):
    GRID_SPACING = 8
    PREVIEW_MS = 5000
    FLIP_CHECK_DELAY_MS = 800

    def __init__(self, num_cards=8, detector=None, feature_extractor=None, validator=None, model_x=None, model_y=None):
        super().__init__()
        self.num_cards = num_cards
        self.rows, self.cols = ((3, num_cards // 3) if num_cards % 3 == 0 else (2, num_cards // 2))

        # state
        self.cards = []
        self.flipped = []
        self.matched = []
        self.locked = True
        self.elapsed = 0
        self.moves = 0

        # współrzędne planszy i kart na ekranie
        self.board_rect_screen = None
        self.card_rects_screen = {}

        # flaga do debugowania hitboxów
        self.debug_hitboxes = False

        # timers
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self._update_timer)

        # --- Generate unique game ID ---
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- logging click data ---
        self.log_file_path = "click_log.csv"
        self.log_file = None
        self.click_counter = 2
        self.game_start_time = None
        
        # --- Phase tracking ---
        self.current_phase = None  # 'memorization' or 'playing'
        self.memorization_start_time = None
        self.playing_start_time = None

        # --- Eye tracking ---
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.validator = validator
        self.model_x = model_x
        self.model_y = model_y
        
        # Separate data storage for each phase
        self.memorization_gaze_data = []
        self.playing_gaze_data = []
        self.gaze_history = deque(maxlen=5)
        self.current_gaze = (0, 0)
        
        self.camera_thread = None
        if self.detector:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self._process_gaze_frame)

        self._build_ui()
        self._create_cards()

    # ---------- UI ----------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(30)

        self.status_label = QLabel("Memorize the cards! 5", alignment=Qt.AlignCenter)
        self.status_label.setStyleSheet(Styles.SUBTITLE)
        layout.addWidget(self.status_label)

        self.timer_label = QLabel("Time: 0s | Moves: 0", alignment=Qt.AlignCenter)
        self.timer_label.setStyleSheet(Styles.HUD)
        layout.addWidget(self.timer_label)

        self.grid_frame = QFrame()
        self.grid_frame.setStyleSheet(Styles.FRAME)
        self.grid = QGridLayout(self.grid_frame)
        self.grid.setSpacing(self.GRID_SPACING)
        for r in range(self.rows):
            self.grid.setRowStretch(r, 1)
        for c in range(self.cols):
            self.grid.setColumnStretch(c, 1)
        layout.addWidget(self.grid_frame)

        # images
        self.front_images = [f"images/{i}.png" for i in range(1, self.num_cards // 2 + 1)] * 2
        random.shuffle(self.front_images)
        self.back_image = "images/backOfCard.png"

    def _create_cards(self):
        for i, img in enumerate(self.front_images):
            btn = QPushButton()
            btn.setStyleSheet(Styles.CARD_BUTTON)
            btn.image_path = img
            
            # Store grid position (row, col) - using 1-based indexing
            row = (i // self.cols) + 1
            col = (i % self.cols) + 1
            btn.grid_row = row
            btn.grid_col = col
            btn.grid_position = int(f"{row}{col}")  # e.g., 11, 12, 23, etc.
            
            btn.clicked.connect(self.on_card_click)
            btn.setIcon(QIcon(img))  # show face-up during preview
            self.grid.addWidget(btn, i // self.cols, i % self.cols)
            self.cards.append(btn)

    def update_hitboxes(self):
        """
        Przelicza prostokąt planszy i poszczególnych kart
        na współrzędne ekranu (do porównywania z eye-trackerem).
        """
        # prostokąt ramki z kartami
        top_left_board = self.grid_frame.mapToGlobal(QPoint(0, 0))
        self.board_rect_screen = QRect(top_left_board, self.grid_frame.size())

        # prostokąty każdej karty
        self.card_rects_screen.clear()
        for btn in self.cards:
            top_left = btn.mapToGlobal(QPoint(0, 0))
            rect = QRect(top_left, btn.size())
            self.card_rects_screen[btn] = rect

        self.update()


    @property
    def elapsed_seconds(self):
        return self.elapsed

    @property
    def move_count(self):
        return self.moves

    # ---------- resize: keep square cards ----------
    def resizeEvent(self, event):
        # size inside grid_frame
        w = self.grid_frame.width()
        h = self.grid_frame.height()
        if self.cols and self.rows:
            cell_w = (w - (self.cols - 1) * self.GRID_SPACING) / self.cols
            cell_h = (h - (self.rows - 1) * self.GRID_SPACING) / self.rows
            size = int(max(0, min(cell_w, cell_h)) * 0.90)
        else:
            size = 0

        icon_size = max(0, size - 10)
        for btn in self.cards:
            btn.setFixedSize(QSize(size, size))
            btn.setIconSize(QSize(icon_size, icon_size))

        self.update_hitboxes()

        super().resizeEvent(event)

    # ---------- gameplay flow ----------
    def _process_gaze_frame(self, frame):
        """Process frame for gaze tracking during both memorization and playing phases."""
        if self.current_phase is None:
            return
        
        # Calculate timestamp based on phase start time
        if self.current_phase == 'memorization':
            if self.memorization_start_time is None:
                self.memorization_start_time = QTime.currentTime()
            timestamp_ms = self.memorization_start_time.msecsTo(QTime.currentTime())
        else:  # playing phase
            # Use game_start_time (same as click logging) for consistency
            if self.playing_start_time is None:
                # Shouldn't happen now, but safety check
                self.playing_start_time = QTime.currentTime()
            timestamp_ms = self.playing_start_time.msecsTo(QTime.currentTime())
        
        result = self.detector.detect(frame)
        
        gaze_x, gaze_y = -1, -1
        valid = 0
        
        if self.validator.is_valid_frame(result):
            features = self.feature_extractor(frame, result).reshape(1, -1)
            gaze_x = float(self.model_x.predict(features)[0])
            gaze_y = float(self.model_y.predict(features)[0])
            self.gaze_history.append((gaze_x, gaze_y))
            valid = 1
        
        # Smooth gaze
        if len(self.gaze_history) > 0:
            hx, hy = np.mean(np.array(self.gaze_history), axis=0)
            self.current_gaze = (int(hx), int(hy))
        
        # Determine which card user is looking at (returns tuple: card_id, grid_position)
        card_id, grid_position = self._get_card_at_gaze(self.current_gaze[0], self.current_gaze[1])
        
        # Log gaze data to appropriate phase storage
        gaze_entry = {
            'ms': timestamp_ms,
            'x_gaze': self.current_gaze[0],
            'y_gaze': self.current_gaze[1],
            'valid': valid,
            'card_id': card_id,
            'grid_position': grid_position
        }
        
        if self.current_phase == 'memorization':
            self.memorization_gaze_data.append(gaze_entry)
        else:  # playing phase
            self.playing_gaze_data.append(gaze_entry)
    
    def _get_card_at_gaze(self, gaze_x, gaze_y):
        """Determine which card the user is looking at.
        Returns: (card_id, grid_position) tuple"""
        for btn in self.cards:
            rect = self.card_rects_screen.get(btn)
            if rect and rect.contains(gaze_x, gaze_y):
                # Extract card ID from image path
                image_path = getattr(btn, "image_path", "")
                card_id = -1
                if image_path:
                    filename = image_path.rsplit("/", 1)[-1]
                    name = filename.split(".", 1)[0]
                    if name.isdigit():
                        card_id = int(name)
                
                # Get grid position
                grid_pos = getattr(btn, "grid_position", -1)
                
                return (card_id, grid_pos)
        return (-1, -1)

    def start_memorize_phase(self):
        self.moves = 0
        self.elapsed = 0
        self._update_hud()

        QTimer.singleShot(0, self.update_hitboxes)

        self._preview_deadline_ms = self.PREVIEW_MS
        self._preview_start = QTime.currentTime()
        self.locked = True
        self.preview_timer.start(100)
        
        # Start memorization phase tracking
        # Note: start_time will be set on first frame to ensure ms=0 at first sample
        self.current_phase = 'memorization'
        self.memorization_start_time = None  # Will be set on first frame
        self.memorization_gaze_data = []
        
        # Start gaze tracking for memorization
        if self.camera_thread and not self.camera_thread.isRunning():
            self.camera_thread.start()

    def _update_preview(self):
        remaining = max(0, self._preview_deadline_ms - self._preview_start.msecsTo(QTime.currentTime()))
        seconds = remaining // 1000 + 1
        self.status_label.setText(f"Memorize the cards! {seconds}")
        if remaining <= 0:
            self.preview_timer.stop()
            
            # Save memorization data before transitioning to playing phase
            self._save_memorization_data()
            
            self._flip_all_to_back()
            self.status_label.setText("Now find the pairs!")
            self.locked = False
            self._start_game_timer()

    def _start_game_timer(self):
        self.elapsed = 0
        self._update_hud()
        self.game_timer.start(1000)

        # Switch to playing phase
        # Note: We set game_start_time immediately so it can be used for click logging
        # The playing_start_time will be set on first gaze frame, but we'll use 
        # game_start_time for both to keep timestamps synchronized
        self.current_phase = 'playing'
        self.game_start_time = QTime.currentTime()
        self.playing_start_time = self.game_start_time  # Use same reference!
        self.playing_gaze_data = []
        
        # start logging clicks
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        self.log_file.write("ms,x_click,y_click,flip,matched,card_id_click,grid_position_click,x_gaze,y_gaze,card_id_gaze,grid_position_gaze\n")
        
        # Camera thread should already be running from memorization phase
        # If not (safety check), start it
        if self.camera_thread and not self.camera_thread.isRunning():
            self.camera_thread.start()

    def _update_timer(self):
        self.elapsed += 1
        self._update_hud()

    def _update_hud(self):
        self.timer_label.setText(f"Time: {self.elapsed}s | Moves: {self.moves}")

    def _flip_all_to_back(self):
        for c in self.cards:
            c.setIcon(QIcon(self.back_image))

    # ---------- interactions ----------
    def on_card_click(self):
        if self.locked:
            return

        btn = self.sender()
        if btn in self.matched or btn in self.flipped:
            return

        btn.setIcon(QIcon(btn.image_path))
        self.flipped.append(btn)

        # jeśli to PIERWSZA karta z pary -> logujemy tu (flip=1, matched=0)
        if len(self.flipped) == 1:
            self.log_click(btn, matched_flag=0)

        # jeśli są dwie karty, odpalamy sprawdzanie po krótkim czasie
        if len(self.flipped) == 2:
            self.locked = True
            QTimer.singleShot(self.FLIP_CHECK_DELAY_MS, self._check_match)

    def _check_match(self):
        a, b = self.flipped
        self.moves += 1
        self._update_hud()

        if a.image_path == b.image_path:
            self.matched += [a, b]
            self.status_label.setText("Nice! You found a pair!")
            # LOG CLICK (flip=2, matched=1)
            self.log_click(b, matched_flag=1)

        else:
            self.status_label.setText("Try again!")
            for btn in self.flipped:
                btn.setIcon(QIcon(self.back_image))
            # LOG CLICK (flip=2, matched=0)
            self.log_click(b, matched_flag=0)

        self.flipped.clear()
        self.locked = False

        if len(self.matched) == len(self.cards):
            self._finish_game()

    def _finish_game(self):
        self.locked = True
        if self.game_timer.isActive():
            self.game_timer.stop()
        
        # Stop gaze tracking
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        self.status_label.setText("You found all pairs! Great job!")
        
        # Save playing phase data
        self._save_playing_data()
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        
        QTimer.singleShot(1500, lambda: getattr(self.window(), "show_win_page", lambda: None)())


    def _save_memorization_data(self):
        """Save memorization phase gaze data to CSV."""
        output_file = f"memorization_data_{self.game_id}.csv"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("ms,x_gaze,y_gaze,valid,card_id,grid_position\n")
                
                for gaze in self.memorization_gaze_data:
                    ms = gaze['ms']
                    x_gaze = gaze['x_gaze']
                    y_gaze = gaze['y_gaze']
                    valid = gaze['valid']
                    card_id = gaze['card_id']
                    grid_position = gaze['grid_position']
                    
                    f.write(f"{ms},{x_gaze},{y_gaze},{valid},{card_id},{grid_position}\n")
            
            print(f"Saved memorization data: {output_file} ({len(self.memorization_gaze_data)} samples)")
        
        except Exception as e:
            print(f"Error saving memorization data: {e}")
    
    def _save_playing_data(self):
        """Save playing phase combined gaze and click data to CSV."""
        output_file = f"playing_data_{self.game_id}.csv"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("ms,x_gaze,y_gaze,valid,card_id_gaze,grid_position_gaze,x_click,y_click,flip,matched,card_id_click,grid_position_click\n")
                
                # Read click data (now includes gaze data at moment of click)
                click_data = []
                if os.path.exists(self.log_file_path):
                    with open(self.log_file_path, 'r', encoding='utf-8') as cf:
                        lines = cf.readlines()[1:]  # Skip header
                        for line in lines:
                            if line.strip():
                                parts = line.strip().split(',')
                                if len(parts) >= 11:  # New format with gaze data
                                    click_data.append({
                                        'ms': int(parts[0]),
                                        'x_click': int(parts[1]),
                                        'y_click': int(parts[2]),
                                        'flip': int(parts[3]),
                                        'matched': int(parts[4]),
                                        'card_id_click': int(parts[5]),
                                        'grid_position_click': int(parts[6]),
                                        'x_gaze': int(parts[7]),
                                        'y_gaze': int(parts[8]),
                                        'card_id_gaze': int(parts[9]),
                                        'grid_position_gaze': int(parts[10])
                                    })
                                elif len(parts) >= 6:  # Old format (backward compatible)
                                    grid_pos = int(parts[6]) if len(parts) >= 7 else -1
                                    click_data.append({
                                        'ms': int(parts[0]),
                                        'x_click': int(parts[1]),
                                        'y_click': int(parts[2]),
                                        'flip': int(parts[3]),
                                        'matched': int(parts[4]),
                                        'card_id_click': int(parts[5]),
                                        'grid_position_click': grid_pos,
                                        'x_gaze': -1,
                                        'y_gaze': -1,
                                        'card_id_gaze': -1,
                                        'grid_position_gaze': -1
                                    })
                
                print(f"DEBUG: Loaded {len(click_data)} click events from {self.log_file_path}")
                if len(click_data) > 0:
                    print(f"DEBUG: First click at {click_data[0]['ms']}ms, Last click at {click_data[-1]['ms']}ms")
                
                # Create a mapping of clicks by timestamp for efficient lookup
                # Use a dict where key is click timestamp, value is click data
                clicks_by_time = {click['ms']: click for click in click_data}
                clicks_matched = set()  # Track which clicks we've matched
                
                if len(self.playing_gaze_data) > 0:
                    print(f"DEBUG: Gaze data from {self.playing_gaze_data[0]['ms']}ms to {self.playing_gaze_data[-1]['ms']}ms")
                
                # Merge gaze and click data for playing phase
                for gaze in self.playing_gaze_data:
                    ms = gaze['ms']
                    x_gaze = gaze['x_gaze']
                    y_gaze = gaze['y_gaze']
                    valid = gaze['valid']
                    card_id_gaze = gaze['card_id']
                    grid_position_gaze = gaze['grid_position']
                    
                    # Check if there's a click within ±50ms window
                    x_click = y_click = flip = matched = card_id_click = grid_position_click = -1
                    
                    # Look for click within window
                    best_click = None
                    best_diff = float('inf')
                    
                    for click_ms, click in clicks_by_time.items():
                        diff = abs(click_ms - ms)
                        if diff <= 50 and diff < best_diff:
                            best_click = click
                            best_diff = diff
                    
                    if best_click is not None:
                        x_click = best_click['x_click']
                        y_click = best_click['y_click']
                        flip = best_click['flip']
                        matched = best_click['matched']
                        card_id_click = best_click['card_id_click']
                        grid_position_click = best_click['grid_position_click']
                        
                        # Note: We now use gaze data from click_log (captured at click moment)
                        # instead of gaze data at this timestamp
                        # Override gaze data with the one from the click moment
                        if best_click['x_gaze'] != -1:  # If gaze was captured at click
                            x_gaze = best_click['x_gaze']
                            y_gaze = best_click['y_gaze']
                            card_id_gaze = best_click['card_id_gaze']
                            grid_position_gaze = best_click['grid_position_gaze']
                        
                        # Add to matched set and log first few matches
                        if best_click['ms'] not in clicks_matched:
                            clicks_matched.add(best_click['ms'])
                            if len(clicks_matched) <= 3:  # Log first 3 matches
                                print(f"DEBUG: Matched click at {best_click['ms']}ms with gaze at {ms}ms (diff: {best_diff}ms)")
                    
                    f.write(f"{ms},{x_gaze},{y_gaze},{valid},{card_id_gaze},{grid_position_gaze},{x_click},{y_click},{flip},{matched},{card_id_click},{grid_position_click}\n")
                
                print(f"DEBUG: Matched {len(clicks_matched)} unique clicks with gaze samples out of {len(click_data)} total clicks")
                
                # Report unmatched clicks
                if len(clicks_matched) < len(click_data):
                    unmatched = [c['ms'] for c in click_data if c['ms'] not in clicks_matched]
                    print(f"DEBUG: Unmatched clicks at: {unmatched[:5]}")  # Show first 5
            
            print(f"Saved playing data: {output_file} ({len(self.playing_gaze_data)} samples)")
        
        except Exception as e:
            print(f"Error saving playing data: {e}")

    def stop_all_timers(self):
        self.preview_timer.stop()
        self.game_timer.stop()
        self.locked = True
        
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None


    def paintEvent(self, event):
        # najpierw normalne rysowanie
        super().paintEvent(event)

        if not self.debug_hitboxes:
            return
        if self.board_rect_screen is None:
            return

        painter = QPainter(self)
        pen = QPen(Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)

        # 1) narysuj ramkę całej planszy (grid_frame)
        top_left_local = self.mapFromGlobal(self.board_rect_screen.topLeft())
        board_rect_local = QRect(top_left_local, self.board_rect_screen.size())
        painter.drawRect(board_rect_local)

        # 2) narysuj prostokąty po każdej karcie
        for rect_screen in self.card_rects_screen.values():
            tl_local = self.mapFromGlobal(rect_screen.topLeft())
            card_rect_local = QRect(tl_local, rect_screen.size())
            painter.drawRect(card_rect_local)

    def log_click(self, btn, matched_flag=0):
        """Loguje: czas od startu gry, współrzędne klikniętej karty,
        flip 1/2, matched_flag 0/1, card_id, grid_position, 
        oraz pozycję wzroku w momencie kliknięcia"""
        if self.log_file is None:
            return

        now = QTime.currentTime()
        ms = self.game_start_time.msecsTo(now)

        # globalne współrzędne kliknięcia: bierzemy środek karty
        rect = self.card_rects_screen.get(btn)
        if rect:
            x_click = rect.center().x()
            y_click = rect.center().y()
        else:
            x_click = y_click = -1

        # wyciągamy numer karty z image_path, np. "images/3.png" -> "3"
        card_id = -1
        image_path = getattr(btn, "image_path", "")
        if image_path:
            filename = image_path.rsplit("/", 1)[-1]  # "3.png"
            name = filename.split(".", 1)[0]  # "3"
            if name.isdigit():
                card_id = int(name)
        
        # Get grid position
        grid_position = getattr(btn, "grid_position", -1)
        
        # Capture gaze position at the exact moment of click
        # Check if eye tracking is available
        if self.detector is None or self.model_x is None or self.model_y is None:
            # No eye tracking - set all gaze values to -1
            x_gaze = y_gaze = -1
            card_id_gaze = grid_position_gaze = -1
            print(f"INFO: Eye tracking not available at click {ms}ms")
        else:
            x_gaze = self.current_gaze[0] if self.current_gaze else 0
            y_gaze = self.current_gaze[1] if self.current_gaze else 0
            
            # Debug: Check if gaze is being tracked
            if x_gaze == 0 and y_gaze == 0:
                print(f"WARNING: No gaze data at click time {ms}ms! current_gaze = {self.current_gaze}")
                print(f"  Camera thread running: {self.camera_thread.isRunning() if self.camera_thread else 'No thread'}")
                print(f"  Gaze history size: {len(self.gaze_history)}")
            
            # Determine which card user is looking at when clicking
            card_id_gaze, grid_position_gaze = self._get_card_at_gaze(x_gaze, y_gaze)

        # flip 1/2/1/2...
        self.click_counter = 1 if self.click_counter == 2 else 2

        # Write: ms, x_click, y_click, flip, matched, card_id_click, grid_position_click,
        #        x_gaze, y_gaze, card_id_gaze, grid_position_gaze
        self.log_file.write(
            f"{ms},{x_click},{y_click},{self.click_counter},{matched_flag},{card_id},{grid_position},"
            f"{x_gaze},{y_gaze},{card_id_gaze},{grid_position_gaze}\n"
        )


# ========================= #
#      Main Game Window     #
# ========================= #
class MemoryGameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Memory Game")

        self._countdown_timer = None
        self._countdown_page = None
        self._countdown_cancelled = False
        self._auto_nav_enabled = True
        self.board_page = None

        # flagi do blokady rozmiaru po rozpoczeciu gry
        self._resize_locked = False
        self._old_min_size = QSize(0, 0)
        self._old_max_size = QSize(16777215, 16777215)

        # Eye tracking components
        self.detector = None
        self.feature_extractor = None
        self.validator = None
        self.model_x = None
        self.model_y = None
        self.calibrated = False
        
        self._init_eye_tracking()
        self._build_ui()
    
    def _init_eye_tracking(self):
        """Initialize eye tracking components."""
        try:
            self.detector = EyeDetector()
            self.feature_extractor = GazeFeatureExtractor(patch_height=10, patch_width=10)
            self.validator = EyeFrameValidator()
        except Exception as e:
            QMessageBox.warning(self, "Eye Tracking Error", 
                              f"Failed to initialize eye tracking: {e}\n\nThe game will run without eye tracking.")
            self.detector = None

    # ---- small helpers ----
    def _add_menu_action(self, menu, text, slot):
        act = QAction(text, self, triggered=slot)
        menu.addAction(act)
        return act

    def _lock_window_resize(self):
        """Zablokuj zmianę rozmiaru okna (na czas gry)."""
        if self._resize_locked:
            return
        self._resize_locked = True

        # zapamiętujemy stare ograniczenia
        self._old_min_size = self.minimumSize()
        self._old_max_size = self.maximumSize()

        # ustawiamy aktualny rozmiar jako stały
        current_size = self.size()
        self.setMinimumSize(current_size)
        self.setMaximumSize(current_size)

    def _unlock_window_resize(self):
        """Odblokuj zmianę rozmiaru okna (po grze / na innych zakładkach)."""
        if not self._resize_locked:
            return
        self._resize_locked = False

        # przywracamy poprzednie ograniczenia
        self.setMinimumSize(self._old_min_size)
        self.setMaximumSize(self._old_max_size)
        # ⛔ NIE MA JUŻ showMaximized() – rozmiar zostaje taki, jaki był

    # ---- UI ----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        menu = self.menuBar()
        menu.setStyleSheet(Styles.MENU_BAR)
        self._add_menu_action(menu, "Home", self.show_home_page)
        self._add_menu_action(menu, "Statistics", self.show_stats_page)
        settings = menu.addMenu("Settings")
        self._add_menu_action(settings, "Set file directory", lambda: None)
        self._add_menu_action(settings, "Restart data", lambda: None)
        self._add_menu_action(settings, "Recalibrate eye-tracking", self.start_recalibration)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.show_home_page()

    def show_home_page(self):
        self._abort_activity()

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(80, 20, 80, 60)
        layout.setSpacing(30)

        title = QLabel("Memory Game", alignment=Qt.AlignCenter)
        title.setStyleSheet(Styles.TITLE)
        layout.addWidget(title)

        content = QHBoxLayout()
        content.setSpacing(60)
        content.setAlignment(Qt.AlignCenter)

        # Instructions
        instructions_frame = QFrame()
        instructions_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #B68DDE;
                border-radius: 12px;
                background-color: #FFFFFF;
                padding: 20px;
            }
        """)
        instructions = QLabel(
            "How to play:\n\n"
            "You will see a grid of cards.\n"
            "They will be face up for 5 seconds and your task is to remember as many as you can.\n"
            "After that, they will flip face down and you can start the game.\n\n"
            "Next:\n"
            "1. Find all matching pairs of cards.\n"
            "2. You can flip two cards at a time.\n"
            "3️. If they match, they stay revealed.\n"
            "4️. If not, they flip back.\n"
            "5️. Try to finish with as few moves and as fast as possible!\n\n"
            "Good luck and have fun!\n\n"
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 22px; color: #333; line-height: 1.4;")
        QVBoxLayout(instructions_frame).addWidget(instructions)

        # Right side: difficulty + play
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignCenter)
        right.setSpacing(20)

        cards_label = QLabel("Select Number of Cards:", alignment=Qt.AlignCenter)
        cards_label.setStyleSheet("font-size: 24px; color: #4B2C82; font-weight: 600;")

        card_box = QComboBox()
        card_box.addItems(["8", "10", "12"])
        card_box.setFixedSize(160, 60)
        card_box.setStyleSheet("""
            font-size: 22px;
            border-radius: 10px;
            padding: 8px 14px;
            background-color: white;
            border: 2px solid #B68DDE;
        """)

        play_btn = QPushButton("▶ Play")
        play_btn.setStyleSheet(Styles.BUTTON)
        play_btn.clicked.connect(lambda: self._on_play_clicked(int(card_box.currentText())))

        note = QLabel(
            "Note: after you click 'Play', you won't be able to resize the window until the game ends."
        )
        note.setAlignment(Qt.AlignCenter)
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 14px; color: #666; font-style: italic;")

        right.addWidget(cards_label)
        right.addWidget(card_box)
        right.addWidget(play_btn)
        right.addWidget(note)

        content.addWidget(instructions_frame)
        content.addLayout(right)

        layout.addStretch(1)
        layout.addLayout(content)
        layout.addStretch(1)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)
    
    def _on_play_clicked(self, num_cards: int):
        """Handle play button click - check calibration first."""
        if not self.detector:
            # No eye tracking, proceed directly
            self.show_countdown(num_cards)
            return
        
        if not self.calibrated:
            # Show calibration info and start calibration
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Eye Tracking Calibration")
            msg.setText("Before starting the game, we need to calibrate the eye tracker.")
            msg.setInformativeText("You will see 20 red points on the screen. Look at each point and click on it when ready.\n\nMake sure you're in a well-lit environment and the camera can see your face.")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            
            if msg.exec_() == QMessageBox.Ok:
                self.start_calibration(num_cards)
            return
        else:
            # Already calibrated, proceed to game
            self.show_countdown(num_cards)
    
    def start_calibration(self, num_cards: int):
        """Start calibration process."""
        self._lock_window_resize()
        
        screen_size = (self.width(), self.height())
        calib_screen = CalibrationScreen(self.detector, self.feature_extractor, 
                                        self.validator, screen_size)
        calib_screen.calibration_complete.connect(lambda mx, my: self._on_calibration_complete(mx, my, num_cards))
        calib_screen.calibration_failed.connect(self._on_calibration_failed)
        
        self.stack.addWidget(calib_screen)
        self.stack.setCurrentWidget(calib_screen)
        
        QTimer.singleShot(500, calib_screen.start_calibration)
    
    def start_recalibration(self):
        """Start recalibration from menu."""
        if not self.detector:
            QMessageBox.warning(self, "Eye Tracking", "Eye tracking is not available.")
            return
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Recalibrate Eye Tracking")
        msg.setText("Do you want to recalibrate the eye tracker?")
        msg.setInformativeText("This will replace your current calibration.")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        if msg.exec_() == QMessageBox.Yes:
            self.calibrated = False
            self.model_x = None
            self.model_y = None
            self.start_calibration(num_cards=8)  # Default to 8 cards for recalibration
    
    def _on_calibration_complete(self, model_x, model_y, num_cards):
        """Handle successful calibration."""
        self.model_x = model_x
        self.model_y = model_y
        self.calibrated = True
        
        QMessageBox.information(self, "Calibration Complete", 
                              "Eye tracking calibration successful!\n\nStarting game...")
        
        self._unlock_window_resize()
        self.show_countdown(num_cards)
    
    def _on_calibration_failed(self):
        """Handle failed calibration."""
        self._unlock_window_resize()
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Calibration Failed")
        msg.setText("Eye tracking calibration failed.")
        msg.setInformativeText("Not enough valid eye detection frames were collected.\n\nDo you want to:\n- Retry calibration\n- Continue without eye tracking")
        retry_btn = msg.addButton("Retry", QMessageBox.AcceptRole)
        continue_btn = msg.addButton("Continue Without Tracking", QMessageBox.RejectRole)
        msg.exec_()
        
        if msg.clickedButton() == retry_btn:
            self.start_calibration(num_cards=8)
        else:
            self.detector = None  # Disable eye tracking
            self.show_home_page()

    def show_countdown(self, num_cards: int):
        self._lock_window_resize()
        self._auto_nav_enabled = True
        self._countdown_cancelled = False

        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Get Ready", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-size: 36px; color: #4B2C82; font-weight: 700;")

        count_label = QLabel("3", alignment=Qt.AlignCenter)
        count_label.setStyleSheet("font-size: 100px; font-weight: bold; color: #8549c9;")

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(count_label)
        layout.addStretch(1)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

        total_ms = 3000
        start_time = QTime.currentTime()
        timer = QTimer(page)  # parented to page → auto-cleanup

        self._countdown_timer = timer
        self._countdown_page = page

        def tick():
            if self._countdown_cancelled:
                timer.stop()
                return
            elapsed = start_time.msecsTo(QTime.currentTime())
            remaining = max(0, total_ms - elapsed)
            seconds = remaining // 1000 + 1
            count_label.setText(str(int(seconds)))

            if remaining <= 0:
                timer.stop()
                if self._countdown_cancelled or not self._auto_nav_enabled:
                    return
                if self.stack.currentWidget() is page:
                    self._countdown_timer = None
                    self._countdown_page = None
                    self.start_game(num_cards)

        timer.timeout.connect(tick)
        timer.start(50)

    def start_game(self, num_cards):
        self._auto_nav_enabled = True
        self.board_page = MemoryGameBoard(
            num_cards=num_cards,
            detector=self.detector,
            feature_extractor=self.feature_extractor,
            validator=self.validator,
            model_x=self.model_x,
            model_y=self.model_y
        )
        self.stack.addWidget(self.board_page)
        self.stack.setCurrentWidget(self.board_page)
        self.board_page.start_memorize_phase()

    def show_win_page(self):
        if not self._auto_nav_enabled:
            return

        self._unlock_window_resize()

        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("🎉 You Win! 🎉", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-size: 52px; font-weight: bold; color: #8549c9;")

        subtitle = QLabel("Congratulations, you found all pairs!", alignment=Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 24px; margin-bottom: 30px;")

        time_taken = getattr(self.board_page, "elapsed_seconds", 0)
        moves = getattr(self.board_page, "move_count", 0)

        stats_label = QLabel(f"Time: {time_taken}s\nMoves: {moves}", alignment=Qt.AlignCenter)
        stats_label.setStyleSheet("font-size: 22px;")

        stats_btn = QPushButton("See Statistics")
        stats_btn.setFixedSize(250, 80)
        stats_btn.setStyleSheet(Styles.BUTTON)
        stats_btn.clicked.connect(self.show_stats_page)

        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(stats_label)
        layout.addSpacing(40)
        layout.addWidget(stats_btn, alignment=Qt.AlignCenter)
        layout.addStretch()

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def show_stats_page(self):
        self._abort_activity()

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Top bar
        top = QHBoxLayout()
        title = QLabel("Your Statistics")
        title.setStyleSheet("font-size: 36px; font-weight: 700; color: #4B2C82;")
        back = QPushButton("Back to Home")
        back.setFixedSize(200, 50)
        back.setStyleSheet(Styles.BUTTON)
        back.clicked.connect(self.show_home_page)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(back)
        layout.addLayout(top)

        # Full-width description
        desc_box = QFrame()
        desc_box.setStyleSheet(Styles.LIGHT_FRAME)
        desc_label = QLabel("Description of a played game and a summary of performance statistics.")
        desc_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        desc_label.setStyleSheet("font-size: 20px; color: #333;")
        QVBoxLayout(desc_box).addWidget(desc_label)
        layout.addWidget(desc_box)

        # Grid of plots
        grid = QGridLayout()
        grid.setSpacing(20)
        for i in range(2):
            for j in range(3):
                box = QFrame()
                box.setFixedSize(250, 250)
                box.setStyleSheet(Styles.PLOT_BOX)
                lbl = QLabel("Plot", alignment=Qt.AlignCenter)
                lbl.setStyleSheet("font-size: 20px; color: #555; font-weight: bold;")
                vb = QVBoxLayout(box)
                vb.addWidget(lbl)
                grid.addWidget(box, i, j, alignment=Qt.AlignCenter)
        layout.addLayout(grid)
        layout.addStretch(1)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    # ---- control / cleanup ----
    def _abort_activity(self):
        self._auto_nav_enabled = False
        self.cancel_countdown()
        if self.board_page:
            self.board_page.stop_all_timers()

        self._unlock_window_resize()

    def cancel_countdown(self):
        if self._countdown_timer is not None:
            self._countdown_timer.stop()
            self._countdown_timer = None
        self._countdown_cancelled = True

        page = self._countdown_page
        if page is not None:
            idx = self.stack.indexOf(page)
            if idx != -1:
                self.stack.removeWidget(page)
            page.deleteLater()
            self._countdown_page = None


# ========================= #
#           Run App         #
# ========================= #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-family: 'Segoe UI'; }")
    win = MemoryGameWindow()
    win.showMaximized()
    sys.exit(app.exec_())
