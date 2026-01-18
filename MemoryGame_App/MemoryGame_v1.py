import shutil
import sys
import random
import os
import re
import csv
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QPushButton, QLabel, QFrame,
    QComboBox, QAction, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QButtonGroup, QRadioButton, QGroupBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QTime, QTimer, QPoint, QRect
from PyQt5.QtGui import QIcon, QPainter, QPen, QCursor

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

sys.path.append("..")
sys.path.append("../GazeLocalization")
sys.path.append("/pages")
try:
    from MemoryGame_App.pages.calibration_screen import CalibrationScreen
    from gaze_data_logger import GazeDataLogger
    from MemoryGame_App.pages.heatmap_view import HeatmapWindow
    from app_data_paths import (
        get_app_data_dir, get_images_dir, get_haar_cascade_path,
        get_game_history_path, get_gaze_data_dir, get_click_log_path,
        get_latest_archived_gaze_file_path
    )
except ImportError as e:
    print(f"ERROR: Required gaze tracking modules not available: {e}")
    print("\nPlease install required dependencies:")
    print("  pip install opencv-python scikit-learn numpy")
    print("\nApplication cannot start without gaze tracking support.")
    sys.exit(1)


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


def extract_card_id_from_filename(image_path: str):
    """Extracts card_id from filename: "3.png" -> 3, "easy (7).jpg" -> 7"""
    if not image_path:
        return None
    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    if name.isdigit():
        return int(name)
    match = re.search(r'\((\d+)\)', name)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', name)
    if match:
        return int(match.group(1))
    return None


class MemoryGameBoard(QWidget):
    GRID_SPACING = 8
    PREVIEW_MS = 5000
    FLIP_CHECK_DELAY_MS = 300

    def __init__(self, num_cards=8, difficulty="easy", gaze_engine=None, gaze_logger=None):
        super().__init__()
        self.num_cards = num_cards
        self.difficulty = difficulty
        # 3 rows for counts divisible by 3, otherwise 2 rows
        self.rows, self.cols = ((3, num_cards // 3) if num_cards % 3 == 0 else (2, num_cards // 2))

        self.cards = []
        self.flipped = []
        self.matched = []
        self.locked = True
        self.elapsed = 0
        self.moves = 0
        self.game_finished = False

        # Screen-global coordinates for gaze mapping
        self.board_rect_screen = None
        self.card_rects_screen = {}
        self.debug_hitboxes = False

        self.gaze_engine = gaze_engine
        self.gaze_logger = gaze_logger
        self.gaze_timer = QTimer(self)
        self.gaze_timer.timeout.connect(self._sample_gaze)
        self.tracking_active = False
        self.current_phase = None

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self._update_timer)

        self.log_file_path = get_click_log_path()
        self.log_file = None
        self.click_counter = 2
        self.game_start_time = None

        self._build_ui()
        self._create_cards()

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

        images_dir = get_images_dir()
        difficulty_dir = os.path.join(images_dir, self.difficulty)
        available_images = []
        if os.path.isdir(difficulty_dir):
            for f in os.listdir(difficulty_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    available_images.append(os.path.join(difficulty_dir, f))

        num_pairs = self.num_cards // 2
        if len(available_images) < num_pairs:
            print(f"Warning: Only {len(available_images)} images in {difficulty_dir}, need {num_pairs}")
            selected_images = random.choices(available_images, k=num_pairs) if available_images else []
        else:
            selected_images = random.sample(available_images, num_pairs)

        self.front_images = selected_images * 2
        random.shuffle(self.front_images)
        self.back_image = os.path.join(images_dir, "backOfCard.png")

    def _create_cards(self):
        for i, img in enumerate(self.front_images):
            btn = QPushButton()
            btn.setStyleSheet(Styles.CARD_BUTTON)
            btn.image_path = img
            btn.card_index = i
            btn.card_row = i // self.cols
            btn.card_col = i % self.cols
            btn.mousePressEvent = lambda event, b=btn: self._on_card_mouse_press(b, event)
            btn.setIcon(QIcon(img))
            self.grid.addWidget(btn, btn.card_row, btn.card_col)
            self.cards.append(btn)

    def update_hitboxes(self):
        top_left_board = self.grid_frame.mapToGlobal(QPoint(0, 0))
        self.board_rect_screen = QRect(top_left_board, self.grid_frame.size())
        self.card_rects_screen.clear()
        for btn in self.cards:
            top_left = btn.mapToGlobal(QPoint(0, 0))
            rect = QRect(top_left, btn.size())
            self.card_rects_screen[btn] = rect

    @property
    def elapsed_seconds(self):
        return self.elapsed

    @property
    def move_count(self):
        return self.moves

    def resizeEvent(self, event):
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

    def start_memorize_phase(self):
        self.moves = 0
        self.elapsed = 0
        self._update_hud()

        QTimer.singleShot(0, self.update_hitboxes)

        self._preview_deadline_ms = self.PREVIEW_MS
        self._preview_start = QTime.currentTime()

        self.locked = True
        self.preview_timer.start(100)

        self.current_phase = "memorization"
        self._start_gaze_tracking()
        if self.gaze_logger:
            timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
            self.gaze_logger.log_phase_event(timestamp_ms, "memorization", "phase_start", 0)

    def _update_preview(self):
        remaining = max(0, self._preview_deadline_ms - self._preview_start.msecsTo(QTime.currentTime()))
        seconds = remaining // 1000 + 1

        self.status_label.setText(f"Memorize the cards! {seconds}")

        if remaining <= 0:
            self.preview_timer.stop()
            self._flip_all_to_back()
            self.status_label.setText("Now find the pairs!")
            self.locked = False
            self._start_game_timer()
            if self.gaze_logger:
                timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
                self.gaze_logger.log_phase_event(timestamp_ms, "memorization", "phase_end", 0)
                self.current_phase = "play"
                self.gaze_logger.log_phase_event(timestamp_ms, "play", "phase_start", 0)

    def _start_game_timer(self):
        self.elapsed = 0
        self._update_hud()
        self.game_timer.start(1000)

        self.game_start_time = QTime.currentTime()
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        self.log_file.write("ms,x,y,flip,matched,card_id\n")

    def _update_timer(self):
        self.elapsed += 1
        self._update_hud()

    def _update_hud(self):
        self.timer_label.setText(f"Time: {self.elapsed}s | Moves: {self.moves}")

    def _flip_all_to_back(self):
        for c in self.cards:
            c.setIcon(QIcon(self.back_image))

    def _on_card_mouse_press(self, btn, event):
        if self.locked:
            QPushButton.mousePressEvent(btn, event)
            return
        if btn in self.matched or btn in self.flipped:
            QPushButton.mousePressEvent(btn, event)
            return

        click_pos_global = btn.mapToGlobal(event.pos())
        click_x = click_pos_global.x()
        click_y = click_pos_global.y()
        QPushButton.mousePressEvent(btn, event)

        btn.setIcon(QIcon(btn.image_path))
        self.flipped.append(btn)

        if len(self.flipped) == 1:
            self.log_click(btn, matched_flag=0, click_x=click_x, click_y=click_y)
        if len(self.flipped) == 2:
            self.locked = True
            QTimer.singleShot(self.FLIP_CHECK_DELAY_MS, lambda: self._check_match(click_x, click_y))

    def _check_match(self, click_x=None, click_y=None):
        a, b = self.flipped
        self.moves += 1
        self._update_hud()

        if a.image_path == b.image_path:
            self.matched += [a, b]
            self.status_label.setText("Nice! You found a pair!")
            if click_x is None or click_y is None:
                cursor_pos = QCursor.pos()
                click_x, click_y = cursor_pos.x(), cursor_pos.y()
            self.log_click(b, matched_flag=1, click_x=click_x, click_y=click_y)
        else:
            self.status_label.setText("Try again!")
            for btn in self.flipped:
                btn.setIcon(QIcon(self.back_image))
            if click_x is None or click_y is None:
                cursor_pos = QCursor.pos()
                click_x, click_y = cursor_pos.x(), cursor_pos.y()
            self.log_click(b, matched_flag=0, click_x=click_x, click_y=click_y)

        self.flipped.clear()
        self.locked = False
        if len(self.matched) == len(self.cards):
            self._finish_game()

    def _finish_game(self):
        self.game_finished = True
        self.locked = True
        if self.game_timer.isActive():
            self.game_timer.stop()
        self.status_label.setText("You found all pairs! Great job!")
        self._stop_gaze_tracking()

        if self.gaze_logger:
            timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
            game_time_ms = self.game_start_time.msecsTo(QTime.currentTime()) if self.game_start_time else 0
            self.gaze_logger.log_phase_event(timestamp_ms, "play", "phase_end", game_time_ms)
            main_window = self.window()
            if main_window and hasattr(main_window, 'last_game_info'):
                log_path = self.gaze_logger.get_log_file_path()
                main_window.last_game_info["gaze_log_path"] = log_path
                print(f"Gaze data saved to: {log_path}")
            self.gaze_logger.stop_logging()
            print(f"Gaze logging stopped. File should be saved at: {self.gaze_logger.get_log_file_path()}")

        main_window = self.window()
        if main_window and hasattr(main_window, 'save_game_result'):
            main_window.save_game_result(self.elapsed, self.moves, self.num_cards, self.difficulty)

        # Keep calibration for multiple games - don't close gaze engine here
        QTimer.singleShot(1500, lambda: getattr(self.window(), "show_win_page", lambda: None)())

        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def stop_all_timers(self):
        self.preview_timer.stop()
        self.game_timer.stop()
        self._stop_gaze_tracking()
        self.locked = True

        if self.log_file:
            self.log_file.close()
            self.log_file = None

        if not self.game_finished and self.log_file_path and os.path.exists(self.log_file_path):
            if self.gaze_logger:
                gaze_path = self.gaze_logger.get_log_file_path()
                self.gaze_logger.stop_logging()
                if not self.game_finished and gaze_path and os.path.exists(gaze_path):
                    try:
                        os.remove(gaze_path)
                    except Exception as e:
                        print("Could not delete gaze log:", e)
                self.gaze_logger = None

        if self.gaze_logger:
            self.gaze_logger.stop_logging()
        # Don't close gaze engine - keep calibration for multiple games

    def _start_gaze_tracking(self):
        if self.gaze_engine and self.gaze_engine.is_calibrated():
            self.tracking_active = True
            self.gaze_timer.start(50)

    def _stop_gaze_tracking(self):
        self.tracking_active = False
        self.gaze_timer.stop()

    def _sample_gaze(self):
        if not self.tracking_active or not self.gaze_engine or not self.gaze_logger:
            return

        gaze = self.gaze_engine.predict_gaze()
        if gaze is None:
            return

        gaze_x, gaze_y = gaze
        timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
        game_time_ms = self.game_start_time.msecsTo(QTime.currentTime()) if self.game_start_time else 0

        if self.board_rect_screen is None:
            self.update_hitboxes()

        # Gaze engine predicts in window coords, hitboxes are in screen coords
        window = self.window()
        if window:
            window_pos = window.mapToGlobal(QPoint(0, 0))
            gaze_screen_x = gaze_x + window_pos.x()
            gaze_screen_y = gaze_y + window_pos.y()
        else:
            gaze_screen_x, gaze_screen_y = gaze_x, gaze_y

        element_info = self._get_element_at_point(gaze_screen_x, gaze_screen_y)
        self.gaze_logger.log_gaze_sample(
            timestamp_ms=timestamp_ms,
            phase=self.current_phase or "",
            gaze_x=gaze_screen_x,
            gaze_y=gaze_screen_y,
            element_type=element_info.get("element_type") or "other",
            card_row=element_info.get("card_row"),
            card_col=element_info.get("card_col"),
            card_id=element_info.get("card_id"),
            card_image_name=element_info.get("card_image_name"),
            game_time_ms=game_time_ms,
        )

    def _get_element_at_point(self, x, y):
        point = QPoint(x, y)
        result = {
            "element_type": "other",
            "card_row": None,
            "card_col": None,
            "card_id": None,
            "card_image_name": None,
        }

        if self.board_rect_screen and self.board_rect_screen.contains(point):
            for btn, rect in self.card_rects_screen.items():
                if rect.contains(point):
                    result["element_type"] = "card"
                    result["card_row"] = btn.card_row + 1
                    result["card_col"] = btn.card_col + 1
                    image_path = getattr(btn, "image_path", "")
                    if image_path:
                        filename = image_path.rsplit("/", 1)[-1]
                        result["card_image_name"] = filename
                        result["card_id"] = extract_card_id_from_filename(image_path)
                    return result
            result["element_type"] = "grid_frame"
            return result

        local_point = self.mapFromGlobal(point)

        if self.status_label:
            label_rect = QRect(
                self.status_label.mapToGlobal(QPoint(0, 0)),
                self.status_label.size()
            )
            if label_rect.contains(point):
                result["element_type"] = "status_label"
                return result

        if self.timer_label:
            label_rect = QRect(
                self.timer_label.mapToGlobal(QPoint(0, 0)),
                self.timer_label.size()
            )
            if label_rect.contains(point):
                result["element_type"] = "timer_label"
                return result

        return result

    def _get_card_info(self, btn):
        result = {
            "card_row": None,
            "card_col": None,
            "card_id": None,
            "card_image_name": None,
        }

        if hasattr(btn, "card_row") and hasattr(btn, "card_col"):
            result["card_row"] = btn.card_row + 1
            result["card_col"] = btn.card_col + 1
        image_path = getattr(btn, "image_path", "")
        if image_path:
            filename = image_path.rsplit("/", 1)[-1]
            result["card_image_name"] = filename
            result["card_id"] = extract_card_id_from_filename(image_path)
        return result

    def log_click(self, btn, matched_flag=0, click_x=None, click_y=None):
        now = QTime.currentTime()
        ms = self.game_start_time.msecsTo(now) if self.game_start_time else 0
        if click_x is not None and click_y is not None:
            x = click_x
            y = click_y
        else:
            rect = self.card_rects_screen.get(btn)
            if rect:
                x = rect.center().x()
                y = rect.center().y()
            else:
                x = y = -1

        image_path = getattr(btn, "image_path", "")
        card_id = extract_card_id_from_filename(image_path)
        if card_id is None:
            card_id = -1

        self.click_counter = 1 if self.click_counter == 2 else 2

        if self.log_file:
            self.log_file.write(
                f"{ms},{x},{y},{self.click_counter},{matched_flag},{card_id}\n"
            )

        if self.gaze_logger:
            gaze_screen_x, gaze_screen_y = None, None
            if self.gaze_engine:
                gaze = self.gaze_engine.predict_gaze()
                if gaze:
                    gaze_x, gaze_y = gaze
                    window = self.window()
                    if window:
                        window_pos = window.mapToGlobal(QPoint(0, 0))
                        gaze_screen_x = gaze_x + window_pos.x()
                        gaze_screen_y = gaze_y + window_pos.y()
                    else:
                        gaze_screen_x, gaze_screen_y = gaze_x, gaze_y

            card_info = self._get_card_info(btn)
            timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
            game_time_ms = ms
            self.gaze_logger.log_click(
                timestamp_ms=timestamp_ms,
                phase=self.current_phase or "play",
                click_x=x,
                click_y=y,
                gaze_x=gaze_screen_x,
                gaze_y=gaze_screen_y,
                element_type="card",
                card_row=card_info["card_row"],
                card_col=card_info["card_col"],
                card_id=card_info["card_id"],
                card_image_name=card_info["card_image_name"],
                matched=matched_flag,
                game_time_ms=game_time_ms,
            )


class MemoryGameWindow(QMainWindow):
    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    GAME_HISTORY_FILE = get_game_history_path()

    def __init__(self, dev_mode=False):
        super().__init__()
        self.setWindowTitle("Memory Game")
        self.dev_mode = dev_mode
        self.session_start = datetime.now()

        self._countdown_timer = None
        self._countdown_page = None
        self._countdown_cancelled = False
        self._auto_nav_enabled = True
        self.board_page = None

        self._resize_locked = False
        self._old_min_size = QSize(0, 0)
        self._old_max_size = QSize(16777215, 16777215)

        self.gaze_engine = None
        self.gaze_logger = None
        self.calibration_done = False
        self.pending_num_cards = None

        self.last_game_info = {
            "num_cards": 8,
            "front_images": [],
            "gaze_log_path": None,
            "board_size": None,
        }
        self.heatmap_window = None
        self.game_history = self._load_game_history()

        # Archive previous gaze data on startup
        gaze_data_dir = get_gaze_data_dir()
        for f in os.listdir(gaze_data_dir):
            path = os.path.join(gaze_data_dir, f)
            if os.path.isfile(path):
                shutil.move(path, gaze_data_dir + "/archived/" + f)

        if self.dev_mode:
            self.setWindowTitle("Memory Game [DEV MODE]")
        self._build_ui()

    def _add_menu_action(self, menu, text, slot):
        act = QAction(text, self, triggered=slot)
        menu.addAction(act)
        return act

    def _lock_window_resize(self):
        if self._resize_locked:
            return
        self._resize_locked = True
        self._old_min_size = self.minimumSize()
        self._old_max_size = self.maximumSize()
        current_size = self.size()
        self.setMinimumSize(current_size)
        self.setMaximumSize(current_size)

    def _unlock_window_resize(self):
        if not self._resize_locked:
            return
        self._resize_locked = False
        self.setMinimumSize(self._old_min_size)
        self.setMaximumSize(self._old_max_size)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        menu = self.menuBar()
        menu.setStyleSheet(Styles.MENU_BAR)
        self._add_menu_action(menu, "Home", self.show_home_page)
        self._add_menu_action(menu, "Statistics", self.show_stats_page)

        settings = menu.addMenu("Settings")
        self._add_menu_action(settings, "Recalibrate Eye-Tracking", self._recalibrate)
        self._add_menu_action(settings, "Clear Game History", self._clear_game_history)
        self._add_menu_action(settings, "Restore Game History", self._restore_game_history)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        self.show_home_page()

    def show_home_page(self):
        self._abort_activity()
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(80, 20, 80, 60)
        layout.setSpacing(30)

        content_widget.setMinimumWidth(1100)

        title = QLabel("Memory Game", alignment=Qt.AlignCenter)
        title.setStyleSheet(Styles.TITLE)
        layout.addWidget(title)

        content = QHBoxLayout()
        content.setSpacing(60)
        content.setAlignment(Qt.AlignCenter)

        # Instructions box
        instructions_frame = QFrame()
        instructions_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #B68DDE;
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

        # Right side: difficulty + play button
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

        difficulty_label = QLabel("Select Difficulty:", alignment=Qt.AlignCenter)
        difficulty_label.setStyleSheet("font-size: 24px; color: #4B2C82; font-weight: 600;")

        difficulty_box = QComboBox()
        difficulty_box.addItems(["easy", "medium", "hard"])
        difficulty_box.setFixedSize(160, 60)
        difficulty_box.setStyleSheet("""
            font-size: 22px;
            border-radius: 10px;
            padding: 8px 14px;
            background-color: white;
            border: 2px solid #B68DDE;
        """)

        play_btn = QPushButton("Play")
        play_btn.setStyleSheet(Styles.BUTTON)
        play_btn.clicked.connect(lambda: self._on_play_clicked(int(card_box.currentText()), difficulty_box.currentText()))

        note = QLabel(
            "Note: after you click 'Play', you won't be able to resize the window until the game ends."
        )
        note.setAlignment(Qt.AlignCenter)
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 14px; color: #666; font-style: italic;")

        right.addWidget(cards_label)
        right.addWidget(card_box)
        right.addWidget(difficulty_label)
        right.addWidget(difficulty_box)
        right.addWidget(play_btn)
        right.addWidget(note)

        content.addWidget(instructions_frame)
        content.addLayout(right)

        layout.addStretch(1)
        layout.addLayout(content)
        layout.addStretch(1)

        scroll.setWidget(content_widget)
        page_layout.addWidget(scroll)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def _on_play_clicked(self, num_cards: int, difficulty: str = "easy"):
        self.pending_num_cards = num_cards
        self.pending_difficulty = difficulty

        if not self.calibration_done:
            reply = QMessageBox.question(
                self,
                "Eye Tracking Calibration Required",
                "Eye tracking calibration is required before playing.\n\n"
                "You will need to look at 20 calibration points and click on each one.\n"
                "This helps the system learn how your eyes move.\n\n"
                "Do you want to proceed with calibration?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self.show_calibration_screen()
            else:
                self.calibration_done = True
                self.gaze_engine = None
                self.show_countdown(num_cards, difficulty)
        else:
            self.show_countdown(num_cards, difficulty)

    def show_calibration_screen(self):
        self._lock_window_resize()
        self._auto_nav_enabled = True
        window_size = (self.geometry().width(), self.geometry().height())
        if window_size[0] <= 0 or window_size[1] <= 0:
            window_size = (self.width(), self.height())

        calibration_page = CalibrationScreen(window_size, self, dev_mode=self.dev_mode)
        self.stack.addWidget(calibration_page)
        self.stack.setCurrentWidget(calibration_page)
        QTimer.singleShot(100, calibration_page.start_calibration)

        self._calibration_check_timer = QTimer(self)
        self._calibration_check_timer.timeout.connect(
            lambda: self._check_calibration_status(calibration_page)
        )
        self._calibration_check_timer.start(500)

    def _check_calibration_status(self, calibration_page):
        if not calibration_page.calibration_complete:
            return
        if hasattr(self, '_calibration_check_timer'):
            self._calibration_check_timer.stop()

        success = calibration_page.is_successful()
        self.gaze_engine = calibration_page.get_gaze_engine()

        msg = QMessageBox(self)
        msg.setWindowTitle("Calibration Result")

        if success:
            msg.setIcon(QMessageBox.Information)
            msg.setText("Calibration Successful!")
            msg.setInformativeText(
                "Eye tracking has been calibrated successfully.\n"
                "You can now proceed to the game."
            )
            self.calibration_done = True
        else:
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Calibration Failed")
            msg.setInformativeText(
                "Not enough samples were collected during calibration.\n\n"
                "Please try again. Make sure:\n"
                "- Your face is clearly visible to the camera\n"
                "- You have good lighting\n"
                "- You click on each calibration point accurately"
            )
            msg.addButton("Retry", QMessageBox.AcceptRole)
            msg.addButton("Skip Calibration", QMessageBox.RejectRole)

        msg.exec_()

        # close() triggers closeEvent which closes camera_window
        idx = self.stack.indexOf(calibration_page)
        if idx != -1:
            self.stack.removeWidget(calibration_page)
            calibration_page.close()
            calibration_page.deleteLater()

        if success:
            if self.pending_num_cards:
                num_cards = self.pending_num_cards
                difficulty = getattr(self, 'pending_difficulty', 'easy')
                self.pending_num_cards = None
                self.pending_difficulty = None
                self.show_countdown(num_cards, difficulty)
        else:
            clicked_btn = msg.clickedButton()
            if clicked_btn and clicked_btn.text() == "Retry":
                self.show_calibration_screen()
            else:
                self.calibration_done = True
                self.gaze_engine = None
                if self.pending_num_cards:
                    num_cards = self.pending_num_cards
                    difficulty = getattr(self, 'pending_difficulty', 'easy')
                    self.pending_num_cards = None
                    self.pending_difficulty = None
                    self.show_countdown(num_cards, difficulty)

    def show_countdown(self, num_cards: int, difficulty: str = "easy"):
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
        timer = QTimer(page)
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
                    self.start_game(num_cards, difficulty)

        timer.timeout.connect(tick)
        timer.start(50)

    def start_game(self, num_cards, difficulty="easy"):
        self._auto_nav_enabled = True

        gaze_logger = None
        if self.gaze_engine:
            try:
                gaze_data_dir = get_gaze_data_dir()
                gaze_logger = GazeDataLogger(output_dir=gaze_data_dir, app_session_id=self.SESSION_ID)
                gaze_logger.start_logging()
                log_path = gaze_logger.get_log_file_path()
                self.last_game_info["gaze_log_path"] = log_path
                print(f"Gaze data will be saved to: {log_path}")
            except Exception as e:
                print(f"Warning: Could not initialize gaze logger: {e}")
                import traceback
                traceback.print_exc()

        self.board_page = MemoryGameBoard(
            num_cards,
            difficulty=difficulty,
            gaze_engine=self.gaze_engine,
            gaze_logger=gaze_logger
        )
        self.last_game_info["num_cards"] = num_cards
        self.last_game_info["difficulty"] = difficulty
        self.last_game_info["front_images"] = self.board_page.front_images.copy()
        self.last_game_info["board_size"] = (self.width(), self.height())

        self.stack.addWidget(self.board_page)
        self.stack.setCurrentWidget(self.board_page)
        self.board_page.start_memorize_phase()

    def show_win_page(self):
        if not self._auto_nav_enabled:
            return
        self._unlock_window_resize()

        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("You Win!", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-size: 52px; font-weight: bold; color: #8549c9;")

        subtitle = QLabel("Congratulations, you found all pairs!", alignment=Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 24px; margin-bottom: 30px;")

        time_taken = getattr(self.board_page, "elapsed_seconds", 0)
        moves = getattr(self.board_page, "move_count", 0)

        stats_label = QLabel(f"Time: {time_taken}s\nMoves: {moves}", alignment=Qt.AlignCenter)
        stats_label.setStyleSheet("font-size: 22px;")

        stats_btn = QPushButton("See Gaze Statistics")
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

    def _create_leaderboard_widget(self, limit=10, top_n=5):
        leaderboard_box = QFrame()
        leaderboard_box.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: none;
                padding: 10px;
            }
        """)
        layout = QVBoxLayout(leaderboard_box)

        header_widget = QWidget()
        header_widget.setFixedHeight(60)
        header_row = QHBoxLayout(header_widget)
        header_row.setSpacing(12)
        header_row.setContentsMargins(0, 0, 0, 0)

        title_box = QFrame()
        title_box.setStyleSheet("""
            QFrame {
                border: 2px solid #B68DDE;
                border-radius: 10px;
                padding: 6px 14px;
                background: #FFFFFF;
            }
        """)

        title_layout = QHBoxLayout(title_box)
        title_layout.setContentsMargins(10, 4, 10, 4)

        title = QLabel(f"Leaderboard (first {top_n} fastest)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #4B2C82; border: none;"
        )
        title_layout.addWidget(title)

        filter_box = QFrame()
        filter_box.setStyleSheet("""
            QFrame {
                border: 2px solid #B68DDE;
                border-radius: 10px;
                padding: 6px 10px;
                background: #FFFFFF;
            }
        """)

        filter_layout = QHBoxLayout(filter_box)
        filter_layout.setContentsMargins(8, 4, 8, 4)
        filter_layout.setSpacing(6)

        filter_lbl = QLabel("Cards:")
        filter_lbl.setStyleSheet("font-size: 18px; color: #333; border: none;")

        cards_combo = QComboBox()
        cards_combo.addItems(["All", "8", "10", "12"])
        cards_combo.setCurrentText("All")
        cards_combo.setFixedWidth(90)
        cards_combo.setStyleSheet("QComboBox { padding: 2px 6px; font-size: 16px; }")

        filter_layout.addWidget(filter_lbl)
        filter_layout.addWidget(cards_combo)
        header_row.addWidget(title_box, 1)
        header_row.addWidget(filter_box, 0)
        layout.addWidget(header_widget)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Cards", "Difficulty", "Time", "Moves", "Date"])
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setFixedHeight(50)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                font-size: 13px;
                gridline-color: #ddd;
                alternate-background-color: #f5f0fa;
            }
            QHeaderView::section {
                background-color: #8549c9;
                color: white;
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #6f37b1;
            }
        """)

        layout.addWidget(table)

        def refresh_table(selected):
            game_history_leaderboard = self.game_history
            if selected == '8':
                game_history_leaderboard = [g for g in self.game_history if g.get("num_cards") == 8]
            elif selected == '10':
                game_history_leaderboard = [g for g in self.game_history if g.get("num_cards") == 10]
            elif selected == '12':
                game_history_leaderboard = [g for g in self.game_history if g.get("num_cards") == 12]

            sorted_history = sorted(
                game_history_leaderboard,
                key=lambda x: x.get("time_seconds", 9999)
            )[:limit]

            table.setRowCount(len(sorted_history))

            for i, game in enumerate(sorted_history):
                table.setItem(i, 0, QTableWidgetItem(str(game.get("num_cards", "?"))))
                table.setItem(i, 1, QTableWidgetItem(game.get("difficulty", "?")))
                table.setItem(i, 2, QTableWidgetItem(f"{game.get('time_seconds', '?')}s"))
                table.setItem(i, 3, QTableWidgetItem(str(game.get("moves", "?"))))

                timestamp = game.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        date_str = dt.strftime("%m/%d %H:%M")
                    except Exception:
                        date_str = "?"
                else:
                    date_str = "?"
                table.setItem(i, 4, QTableWidgetItem(date_str))

        refresh_table("All")
        cards_combo.currentTextChanged.connect(refresh_table)
        return leaderboard_box

    def show_stats_page(self):
        self._abort_activity()
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setLineWidth(0)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        content.setMinimumWidth(1200)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        top = QHBoxLayout()
        title = QLabel("Your Gaze Statistics")
        title.setStyleSheet("font-size: 36px; font-weight: 700; color: #4B2C82;")
        back = QPushButton("Back to Home")
        back.setFixedSize(200, 50)
        back.setStyleSheet(Styles.BUTTON)
        back.clicked.connect(self.show_home_page)

        analysis_box = QGroupBox("Analysis Type")
        analysis_box_layout = QHBoxLayout(analysis_box)
        single = QRadioButton("Last Game")
        single.setChecked(True)
        multiple = QRadioButton("All Games")
        analysis_box_layout.addWidget(single)
        analysis_box_layout.addWidget(multiple)
        analysis_type = QButtonGroup(self)
        analysis_type.addButton(single, 1)
        analysis_type.addButton(multiple, 2)
        analysis_box.setStyleSheet("""
            QGroupBox { font-size: 20px; font-weight: bold; color: #4B2C82; }
            QRadioButton { font-size: 17px; }
        """)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(analysis_box)
        top.addWidget(back)
        layout.addLayout(top)

        mode_stack = QStackedWidget()
        layout.addWidget(mode_stack, 1)

        no_data_container = QWidget()
        no_data_layout = QVBoxLayout(no_data_container)
        no_data_layout.setContentsMargins(0, 80, 0, 0)
        no_data_layout.setSpacing(10)

        no_data_title = QLabel("There is no data to perform analysis on")
        no_data_title.setAlignment(Qt.AlignCenter)
        no_data_title.setStyleSheet("font-size: 28px; font-weight: 700; color: #4B2C82;")

        no_data_sub = QLabel("Play a game or try restoring data.")
        no_data_sub.setAlignment(Qt.AlignCenter)
        no_data_sub.setStyleSheet("font-size: 16px; color: #666;")

        no_data_layout.addWidget(no_data_title)
        no_data_layout.addWidget(no_data_sub)
        no_data_layout.addStretch(1)

        last_game_container = QWidget()
        last_game_layout = QVBoxLayout(last_game_container)
        last_game_layout.setSpacing(15)

        gaze_log_path = get_latest_archived_gaze_file_path(archived=False)
        if not gaze_log_path:
            gaze_log_path = get_latest_archived_gaze_file_path(archived=True)

        gaze_log_path = self._resolve_gaze_log_path(gaze_log_path)

        try:
            ts = os.path.getmtime(gaze_log_path) if gaze_log_path else None
            dt = datetime.fromtimestamp(ts)
            gaze_time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            gaze_time_str = "N/A"

        gaze_note = QLabel(f"Showing statistics based on gaze log from game played on {gaze_time_str}")
        gaze_note.setStyleSheet("font-size: 15px; color: #666; font-style: italic;")
        gaze_note.setAlignment(Qt.AlignLeft)
        last_game_layout.addWidget(gaze_note)

        top_row = QHBoxLayout()
        top_row.setSpacing(20)

        metrics_box = QFrame()
        metrics_box.setStyleSheet(Styles.LIGHT_FRAME)
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setSpacing(20)

        matched_game = self._find_game_record_for_gaze_file(gaze_log_path) if gaze_log_path else None
        if matched_game:
            time_taken = matched_game.get("time_seconds", "N/A")
            moves = matched_game.get("moves", "N/A")
            num_cards = matched_game.get("num_cards", "N/A")
            difficulty = matched_game.get("difficulty", "N/A")
        else:
            time_taken = "N/A"
            moves = "N/A"
            num_cards = "N/A"
            difficulty = "N/A"
            print("WARNING: No game_history entry matches gaze file:", gaze_log_path)

        gaze_stats = self._compute_gaze_statistics(gaze_log_path)
        print(f"Computing gaze stats from: {gaze_log_path}")

        total_samples = gaze_stats.get("memorization_samples", 0) + gaze_stats.get("play_samples", 0)

        numbers_row = QHBoxLayout()
        numbers_row.setSpacing(18)

        num_cards_lbl = QLabel(f"Number of cards\n{num_cards}")
        difficulty_lbl = QLabel(f"Difficulty\n{difficulty}")
        time_lbl = QLabel(f"Time to finish\n{time_taken}s")
        moves_lbl = QLabel(f"Moves\n{moves}")
        samples_lbl = QLabel(f"Gaze samples\n{total_samples}")

        for w in (num_cards_lbl, difficulty_lbl, time_lbl, moves_lbl, samples_lbl):
            w.setAlignment(Qt.AlignCenter)
            w.setStyleSheet("font-size: 20px; color: #333; font-weight: 600;")
            w.setMinimumWidth(0)
            w.setWordWrap(True)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        numbers_wrap = QWidget()
        numbers_wrap.setLayout(numbers_row)
        numbers_row.addWidget(num_cards_lbl)
        numbers_row.addWidget(difficulty_lbl)
        numbers_row.addWidget(time_lbl)
        numbers_row.addWidget(moves_lbl)
        numbers_row.addWidget(samples_lbl)

        gaze_distribution_row = QHBoxLayout()
        gaze_distribution_row.setSpacing(8)

        gaze_dist_title = QLabel("Gaze \nDistribution:")
        gaze_dist_title.setStyleSheet("font-size: 18px; font-weight: 500; color: #333;")
        gaze_dist_title.setAlignment(Qt.AlignTop)
        gaze_distribution_row.addWidget(gaze_dist_title)

        gaze_dist_memorization = QLabel()
        gaze_dist_memorization.setStyleSheet("font-size: 17px; color: #333;")
        gaze_dist_memorization.setWordWrap(True)
        gaze_dist_memorization.setText(
            "Memorization: \n"
            f"- Cards {gaze_stats.get('memorization_card_percentage', 0):.1f}%\n"
            f"- Grid frame {gaze_stats.get('memorization_gridFrame_percentage', 0):.1f}%\n"
            f"- Timer label and status label {gaze_stats.get('memorization_labels_percentage', 0):.1f}%\n"
            f"- Other {gaze_stats.get('memorization_other_percentage', 0):.1f}%\n"
        )

        gaze_dist_play = QLabel()
        gaze_dist_play.setStyleSheet("font-size: 17px; color: #333;")
        gaze_dist_play.setWordWrap(True)
        gaze_dist_play.setText(
            "Play: \n"
            f"- Cards {gaze_stats.get('play_card_percentage', 0):.1f}%\n"
            f"- Grid frame {gaze_stats.get('play_gridFrame_percentage', 0):.1f}%\n"
            f"- Timer label and status label {gaze_stats.get('play_labels_percentage', 0):.1f}%\n"
            f"- Other {gaze_stats.get('play_other_percentage', 0):.1f}%\n"
        )

        for lbl in (gaze_dist_memorization, gaze_dist_play):
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        gaze_distribution_row.addWidget(gaze_dist_memorization)
        gaze_distribution_row.addWidget(gaze_dist_play)

        metrics_layout.addWidget(numbers_wrap)
        metrics_layout.addLayout(gaze_distribution_row)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        small_leaderboard = self._create_leaderboard_widget(limit=5)
        small_leaderboard.setFixedHeight(370)
        right_col.addWidget(small_leaderboard)

        has_heatmap_data = bool(self.last_game_info.get("gaze_log_path"))

        heatmap_btn = QPushButton("View Gaze Heatmap")
        heatmap_btn.setFixedHeight(45)
        heatmap_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #d35400; }
            QPushButton:pressed { background-color: #c0392b; }
            QPushButton:disabled {
                background-color: #cfcfcf;
                color: #7a7a7a;
            }
        """)

        if not has_heatmap_data:
            heatmap_btn.setEnabled(False)
            heatmap_btn.setToolTip(
                "Heatmap unavailable.\nHeatmap is available only for the last game if you played with active eye-tracking.")
        else:
            heatmap_btn.setEnabled(True)
            heatmap_btn.setToolTip("Show heatmap for the most recent gaze log.")

        heatmap_btn.clicked.connect(self._show_heatmap)
        right_col.addWidget(heatmap_btn)

        right_col.addStretch(1)

        right_wrap = QWidget()
        right_wrap.setLayout(right_col)
        right_wrap.setMinimumWidth(420)
        metrics_box.setMinimumHeight(405)
        right_wrap.setMinimumHeight(405)

        top_row.addWidget(metrics_box, 2)
        top_row.addWidget(right_wrap, 2)

        last_game_layout.addLayout(top_row)

        plots_grid = QGridLayout()
        plots_grid.setSpacing(15)

        def plot_frame(title_text, content_widget: QWidget):
            box = QFrame()
            box.setStyleSheet("""
                QFrame {
                    background-color: #FFFFFF;
                    border-radius: 12px;
                    border: 2px solid #B68DDE;
                    padding: 10px;
                }
            """)
            box.setMinimumHeight(420)
            v = QVBoxLayout(box)

            t = QLabel(title_text)
            t.setStyleSheet("font-size: 16px; font-weight: bold; color: #4B2C82;")
            v.addWidget(t)

            content_widget.setMinimumHeight(340)
            v.addWidget(content_widget, 1)

            return box

        canvas1 = self._plot_gaze_per_card(gaze_log_path)
        canvas2 = self._plot_gaze_before_matched(gaze_log_path, window_ms=3000)
        canvas3 = self._plot_correct_vs_incorrect(gaze_log_path, window_ms=1000)
        canvas4 = self._plot_gaze_over_time(gaze_log_path, bin_ms=1000)

        plots_grid.addWidget(plot_frame("Gaze time % per card (Memorization vs Play)", canvas1), 0, 0)
        plots_grid.addWidget(plot_frame("Gaze time on card before it was matched", canvas2), 0, 1)
        plots_grid.addWidget(plot_frame("Correct vs incorrect gaze comparison (before click)", canvas3), 1, 0)
        plots_grid.addWidget(plot_frame("Gaze on cards over time (Play)", canvas4), 1, 1)

        last_game_layout.addLayout(plots_grid, 1)

        all_games_container = QWidget()
        all_games_layout = QVBoxLayout(all_games_container)
        all_games_layout.setSpacing(15)

        all_history_games = list(self.game_history)

        if not all_history_games:
            msg = QLabel("No games played yet.")
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("font-size: 18px; color: #666;")
            all_games_layout.addStretch(1)
            all_games_layout.addWidget(msg)
            all_games_layout.addStretch(1)
        else:
            # session_start_str = self.session_start.strftime("%Y-%m-%d %H:%M:%S")
            all_games_gaze_note = QLabel(
                f"Showing statistics for all games played."
            )
            all_games_gaze_note.setStyleSheet("font-size: 15px; color: #666; font-style: italic;")
            all_games_gaze_note.setAlignment(Qt.AlignLeft)

            all_games_layout.addWidget(all_games_gaze_note)

            top_row_all = QHBoxLayout()
            top_row_all.setSpacing(20)

            all_games_box = QFrame()
            all_games_box.setStyleSheet(Styles.LIGHT_FRAME)
            all_games_metrics = QVBoxLayout(all_games_box)
            all_games_metrics.setSpacing(20)

            games_played = len(all_history_games)
            times = [(g.get("time_seconds") or 0) for g in all_history_games]
            avg_time = (sum(times) / games_played) if games_played else 0.0
            valid_times = [t for t in times if t > 0]
            best_time = min(valid_times) if valid_times else 0.0

            efficiencies = []
            for g in all_history_games:
                num_cards = g.get("num_cards") or 0
                moves = g.get("moves") or 0
                if num_cards > 0 and moves > 0:
                    pairs = num_cards / 2.0
                    efficiencies.append(pairs / float(moves))

            avg_eff = (sum(efficiencies) / len(efficiencies)) if efficiencies else 0.0
            best_eff = max(efficiencies) if efficiencies else 0.0

            numbers_wrap_all = QWidget()
            grid = QGridLayout(numbers_wrap_all)
            grid.setSpacing(25)

            games_lbl = QLabel(f"Games played\n{games_played}")
            avg_time_lbl = QLabel(f"Average game time\n{int(round(avg_time))}s")
            best_time_lbl = QLabel(f"Best game time\n{int(round(best_time))}s")
            avg_eff_lbl = QLabel(f"Average efficiency\n{avg_eff:.2f}")
            best_eff_lbl = QLabel(f"Best efficiency\n{best_eff:.2f}")

            for w in (games_lbl, avg_time_lbl, best_time_lbl,
                      avg_eff_lbl, best_eff_lbl):
                w.setAlignment(Qt.AlignCenter)
                w.setStyleSheet("font-size: 20px; color: #333; font-weight: 600;")
                w.setMinimumWidth(0)
                w.setWordWrap(True)
                w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            grid.addWidget(games_lbl, 0, 0, 2, 1, alignment=Qt.AlignCenter)

            grid.addWidget(avg_time_lbl, 0, 1)
            grid.addWidget(best_time_lbl, 0, 2)

            grid.addWidget(avg_eff_lbl, 1, 1)
            grid.addWidget(best_eff_lbl, 1, 2)

            all_games_metrics.addWidget(numbers_wrap_all)

            top_row_all.addWidget(all_games_box, 2)

            right_col_all = QVBoxLayout()
            right_col_all.setSpacing(12)

            all_games_leaderboard = self._create_leaderboard_widget(limit=5)
            all_games_leaderboard.setFixedHeight(370)
            right_col_all.addWidget(all_games_leaderboard)

            right_col_all.addStretch(1)

            right_wrap_all = QWidget()
            right_wrap_all.setLayout(right_col_all)
            right_wrap_all.setMinimumWidth(420)

            all_games_box.setMinimumHeight(405)
            right_wrap_all.setMinimumHeight(405)

            top_row_all.addWidget(right_wrap_all, 2)

            all_games_layout.addLayout(top_row_all)

            plots_grid_all = QGridLayout()
            plots_grid_all.setSpacing(15)
            plots_grid_all.setRowStretch(0, 1)
            plots_grid_all.setRowStretch(1, 1)
            plots_grid_all.setColumnStretch(0, 1)
            plots_grid_all.setColumnStretch(1, 1)

            eff_canvas = self._plot_session_efficiency(all_history_games)
            plots_grid_all.addWidget(
                plot_frame("Efficiency (pairs / moves) and pace index over games", eff_canvas),
                0, 0
            )
            gaze_dist_widget = self._create_session_gaze_distribution_widget(all_history_games)
            plots_grid_all.addWidget(
                plot_frame("Gaze distribution over games", gaze_dist_widget),
                0, 1
            )
            fix_canvas = self._plot_session_fixation_boxplot(all_history_games)
            plots_grid_all.addWidget(
                plot_frame("Fixation duration on cards per game", fix_canvas),
                1, 0
            )
            exploration_canvas = self._plot_session_exploration_scatter(all_history_games)
            plots_grid_all.addWidget(
                plot_frame("Exploration pace vs median fixation duration per game",
                           exploration_canvas),
                1, 1
            )

            all_games_layout.addLayout(plots_grid_all, 1)

        mode_stack.addWidget(no_data_container)
        mode_stack.addWidget(last_game_container)
        mode_stack.addWidget(all_games_container)

        def on_mode_changed():
            if multiple.isChecked():
                mode_stack.setCurrentIndex(2)
            else:
                mode_stack.setCurrentIndex(0 if (total_samples == 0) else 1)

        single.toggled.connect(on_mode_changed)
        multiple.toggled.connect(on_mode_changed)
        on_mode_changed()

        scroll.setWidget(content)
        page_layout.addWidget(scroll)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def _show_heatmap(self):
        gaze_log_path = self.last_game_info.get("gaze_log_path")
        if not gaze_log_path or not os.path.exists(gaze_log_path):
            QMessageBox.information(self, "Heatmap unavailable", "No gaze data found for the last game.")
            return

        images_dir = get_images_dir()
        default_images = [os.path.join(images_dir, f"{i}.png") for i in range(1, 5)] * 2
        game_config = {
            "num_cards": self.last_game_info.get("num_cards", 8),
            "front_images": self.last_game_info.get("front_images", default_images),
            "board_size": self.last_game_info.get("board_size", (self.width(), self.height())),
        }

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(20, 20, 20, 20)
        page_layout.setSpacing(10)

        def go_back():
            idx = self.stack.indexOf(page)
            if idx != -1:
                self.stack.removeWidget(page)
            page.deleteLater()
            self.show_stats_page()

        heatmap_widget = HeatmapWindow(
            gaze_data_path=gaze_log_path,
            game_config=game_config,
            parent=page,
            on_back=go_back
        )
        page_layout.addWidget(heatmap_widget, 1)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def closeEvent(self, event):
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if hasattr(widget, 'camera_window') and widget.camera_window:
                widget.camera_window.close()
            if hasattr(widget, 'camera_timer'):
                widget.camera_timer.stop()
        if self.gaze_engine:
            self.gaze_engine.close()
            self.gaze_engine = None
        if self.board_page:
            self.board_page.stop_all_timers()
        event.accept()

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

    def _load_game_history(self):
        try:
            if os.path.exists(self.GAME_HISTORY_FILE):
                with open(self.GAME_HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading game history: {e}")
        return []

    def _save_game_history(self):
        try:
            with open(self.GAME_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.game_history, f, indent=2)
        except Exception as e:
            print(f"Error saving game history: {e}")

    def save_game_result(self, time_seconds, moves, num_cards, difficulty="easy"):
        result = {
            "session_id": self.SESSION_ID,
            "timestamp": datetime.now().isoformat(),
            "time_seconds": time_seconds,
            "moves": moves,
            "num_cards": num_cards,
            "difficulty": difficulty,
            "gaze_log_path": self.last_game_info.get("gaze_log_path"),
        }
        self.game_history.append(result)
        self._save_game_history()

    def _clear_game_history(self):
        reply = QMessageBox.question(
            self, "Clear Game History",
            "Are you sure you want to clear all game history?\n",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            gaze_data_dir = get_gaze_data_dir()
            app_data_dir = get_app_data_dir()
            for f in os.listdir(gaze_data_dir):
                path = os.path.join(gaze_data_dir, f)
                if os.path.isfile(path):
                    shutil.move(path, str(Path(app_data_dir) / 'data_cleared' / f))
                if os.path.isdir(path):
                    for fa in os.listdir(path):
                        path_a = os.path.join(path, fa)
                        shutil.move(path_a, str(Path(app_data_dir) / 'data_cleared' / "archived" / fa))
            history_data_path = get_game_history_path()
            if os.path.isfile(history_data_path):
                shutil.move(history_data_path, str(Path(app_data_dir) / 'data_cleared' / 'game_history.json'))
            self.game_history = []
            self._save_game_history()
            QMessageBox.information(self, "Cleared", "Game history has been cleared.")

    def _restore_game_history(self):
        cleared_data_dir = Path(get_app_data_dir()) / 'data_cleared'
        for f in os.listdir(cleared_data_dir):
            src_path = cleared_data_dir / f
            if os.path.isfile(src_path):
                if f == 'game_history.json':
                    dest_path = Path(get_game_history_path())
                else:
                    dest_path = Path(get_gaze_data_dir()) / f
                shutil.move(str(src_path), str(dest_path))
            elif os.path.isdir(src_path):
                for fa in os.listdir(src_path):
                    src_path_a = src_path / fa
                    dest_path_a = Path(get_gaze_data_dir()) / "archived" / fa
                    shutil.move(str(src_path_a), str(dest_path_a))
        self.game_history = self._load_game_history()
        QMessageBox.information(self, "Restored", "Game history has been restored.")

    def _recalibrate(self):
        if self.gaze_engine:
            self.gaze_engine.close()
            self.gaze_engine = None
        self.calibration_done = False

        QMessageBox.information(
            self, "Recalibration",
            "Eye tracking calibration has been reset.\nYou will be prompted to recalibrate when you start the next game."
        )

    def _find_game_record_for_gaze_file(self, gaze_log_path: str):
        """Match by basename so it works even if files moved to /archived."""
        if not gaze_log_path or not self.game_history:
            return None
        target_name = os.path.basename(gaze_log_path)
        for g in reversed(self.game_history):
            p = g.get("gaze_log_path")
            if not p:
                continue
            if os.path.basename(p) == target_name:
                return g

        return None


    def _resolve_gaze_log_path(self, p: str):
        """If gaze path was archived, find it in archived/ by filename."""
        if not p:
            return None
        if os.path.exists(p):
            return p

        # try archived location by basename
        fname = os.path.basename(p)
        archived = os.path.join(get_gaze_data_dir(), "archived", fname)
        if os.path.exists(archived):
            return archived

        # try to find same filename anywhere in gaze dirs
        for base in (get_gaze_data_dir(), os.path.join(get_gaze_data_dir(), "archived")):
            cand = os.path.join(base, fname)
            if os.path.exists(cand):
                return cand

        return None


    def _compute_gaze_statistics(self, gaze_log_path):
        stats = {
            "memorization_samples": 0,
            "memorization_on_cards": 0,
            "memorization_on_gridFrame": 0,
            "memorization_on_labels": 0,
            "memorization_on_other": 0,
            "memorization_card_percentage": 0,
            "memorization_gridFrame_percentage": 0,
            "memorization_labels_percentage": 0,
            "memorization_other_percentage": 0,
            "play_samples": 0,
            "play_on_cards": 0,
            "play_on_gridFrame": 0,
            "play_on_labels": 0,
            "play_on_other": 0,
            "play_card_percentage": 0,
            "play_gridFrame_percentage": 0,
            "play_labels_percentage": 0,
            "play_other_percentage": 0
        }

        if not gaze_log_path or not os.path.exists(gaze_log_path):
            return stats

        try:
            with open(gaze_log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("event_type") != "gaze_sample":
                        continue

                    phase = row.get("phase", "")
                    element_type = row.get("element_type", "")

                    if phase == "memorization":
                        stats["memorization_samples"] += 1
                        if element_type == "card":
                            stats["memorization_on_cards"] += 1
                        elif element_type == "grid_frame":
                            stats["memorization_on_gridFrame"] += 1
                        elif element_type in ("status_label", "timer_label"):
                            stats["memorization_on_labels"] += 1
                        else:
                            stats["memorization_on_other"] += 1
                    elif phase == "play":
                        stats["play_samples"] += 1
                        if element_type == "card":
                            stats["play_on_cards"] += 1
                        elif element_type == "grid_frame":
                            stats["play_on_gridFrame"] += 1
                        elif element_type in ("status_label", "timer_label"):
                            stats["play_on_labels"] += 1
                        else:
                            stats["play_on_other"] += 1

            m = stats["memorization_samples"]
            if m > 0:
                stats["memorization_card_percentage"] = stats["memorization_on_cards"] / m * 100
                stats["memorization_gridFrame_percentage"] = stats["memorization_on_gridFrame"] / m * 100
                stats["memorization_labels_percentage"] = stats["memorization_on_labels"] / m * 100
                stats["memorization_other_percentage"] = stats["memorization_on_other"] / m * 100

            p = stats["play_samples"]
            if p > 0:
                stats["play_card_percentage"] = stats["play_on_cards"] / p * 100
                stats["play_gridFrame_percentage"] = stats["play_on_gridFrame"] / p * 100
                stats["play_labels_percentage"] = stats["play_on_labels"] / p * 100
                stats["play_other_percentage"] = stats["play_on_other"] / p * 100

        except Exception as e:
            print(f"Error computing gaze statistics: {e}")

        return stats

    def _load_gaze_log_rows(self, gaze_log_path: str):
        if not gaze_log_path or not os.path.exists(gaze_log_path):
            return []

        rows = []
        with open(gaze_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                for k in ("game_time_ms", "timestamp_ms", "card_id", "matched", "card_row", "card_col"):
                    if k in r and r[k] not in (None, "", "None"):
                        try:
                            r[k] = int(float(r[k]))
                        except Exception:
                            pass
                rows.append(r)
        return rows

    def _detect_card_fixation_durations(self, gaze_log_path: str, min_duration_ms: int = 300):
        """Detects fixations on cards (play phase). Removes outliers via IQR."""
        rows = self._load_gaze_log_rows(gaze_log_path)
        if not rows:
            return []

        samples = [
            r for r in rows
            if r.get("event_type") == "gaze_sample"
               and r.get("element_type") == "card"
               and (r.get("phase") or "").strip() == "play"
               and r.get("card_id") not in (None, "", "None")
        ]
        if not samples:
            return []

        def get_time(r):
            return r.get("game_time_ms") or r.get("timestamp_ms") or 0

        samples.sort(key=get_time)

        durations = []
        current_card = None
        start_t = None
        last_t = None

        for r in samples:
            t = get_time(r)
            cid = r.get("card_id")

            if current_card is None:
                current_card = cid
                start_t = t
                last_t = t
                continue

            if cid == current_card:
                last_t = t
            else:
                if start_t is not None and last_t is not None:
                    duration = last_t - start_t
                    if duration >= min_duration_ms:
                        durations.append(duration)
                current_card = cid
                start_t = t
                last_t = t

        if current_card is not None and start_t is not None and last_t is not None:
            duration = last_t - start_t
            if duration >= min_duration_ms:
                durations.append(duration)

        if not durations:
            return durations

        durations_arr = np.array(durations, dtype=float)
        q1 = np.percentile(durations_arr, 25)
        q3 = np.percentile(durations_arr, 75)
        iqr = q3 - q1

        upper = q3 + 1.5 * iqr
        lower = max(min_duration_ms, q1 - 1.5 * iqr)

        durations_filtered = [d for d in durations if lower <= d <= upper]

        return durations_filtered

    def _plot_session_fixation_boxplot(self, session_games) -> FigureCanvas:
        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        all_durations = []
        labels = []

        for idx, g in enumerate(session_games, start=1):
            gaze_log_path = self._resolve_gaze_log_path(g.get("gaze_log_path"))
            if not gaze_log_path or not os.path.exists(gaze_log_path):
                continue

            durations_ms = self._detect_card_fixation_durations(
                gaze_log_path,
                min_duration_ms=300,
            )
            if not durations_ms:
                continue

            durations_s = [d / 1000.0 for d in durations_ms]
            all_durations.append(durations_s)
            labels.append(str(len(labels) + 1))

        if not all_durations:
            ax.text(
                0.5,
                0.5,
                "No fixation data\navailable.",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        ax.boxplot(all_durations, labels=labels, showmeans=True)
        ax.set_xlabel("Game number")
        ax.set_ylabel("Fixation duration on cards (s)")

        return FigureCanvas(fig)

    def _plot_session_exploration_scatter(self, session_games) -> FigureCanvas:
        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        x_pace = []
        y_median_fix = []
        labels = []

        for idx, g in enumerate(session_games, start=1):
            time_s = g.get("time_seconds") or 0
            gaze_log_path = self._resolve_gaze_log_path(g.get("gaze_log_path"))

            if not gaze_log_path or not os.path.exists(gaze_log_path) or time_s <= 0:
                continue

            rows = self._load_gaze_log_rows(gaze_log_path)
            play_card_samples = [
                r for r in rows
                if r.get("event_type") == "gaze_sample"
                   and (r.get("phase") or "").strip() == "play"
                   and r.get("element_type") == "card"
                   and r.get("game_time_ms") is not None
            ]
            if not play_card_samples:
                continue

            pace = len(play_card_samples) / float(time_s)

            durations_ms = self._detect_card_fixation_durations(
                gaze_log_path,
                min_duration_ms=300,
            )
            if not durations_ms:
                continue

            durations_s = sorted(d / 1000.0 for d in durations_ms)

            n = len(durations_s)
            if n % 2 == 1:
                median_fix = durations_s[n // 2]
            else:
                median_fix = 0.5 * (durations_s[n // 2 - 1] + durations_s[n // 2])

            x_pace.append(pace)
            y_median_fix.append(median_fix)
            labels.append(str(len(labels) + 1))

        if not x_pace:
            ax.text(
                0.5, 0.5,
                "No fixation data\navailable.",
                ha="center", va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        ax.scatter(x_pace, y_median_fix)

        for x, y, label in zip(x_pace, y_median_fix, labels):
            ax.text(x, y, label, ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Exploration pace (fixations on cards per second)")
        ax.set_ylabel("Median fixation duration on cards (s)")

        return FigureCanvas(fig)

    def _create_session_gaze_distribution_widget(self, session_games) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        phase_box = QFrame()
        phase_box.setStyleSheet("""
            QFrame {
                border: 2px solid #B68DDE;
                border-radius: 8px;
                padding: 2px 6px;
                background: #FFFFFF;
            }
        """)

        phase_layout = QHBoxLayout(phase_box)

        phase_layout.setContentsMargins(4, 2, 4, 2)
        phase_layout.setSpacing(4)

        phase_label = QLabel("Phase:")
        phase_label.setStyleSheet("font-size: 16px; color: #333; border: none;")

        phase_combo = QComboBox()
        phase_combo.addItems(["Memorization", "Play"])
        phase_combo.setCurrentText("Memorization")
        phase_combo.setFixedWidth(150)
        phase_combo.setStyleSheet("QComboBox { padding: 2px 6px; font-size: 16px; }")

        phase_layout.addWidget(phase_label)
        phase_layout.addWidget(phase_combo)

        header_layout.addStretch(1)
        header_layout.addWidget(phase_box)
        layout.addWidget(header_widget)

        game_indices = []
        mem_cards = []
        mem_grid = []
        mem_labels = []
        mem_other = []

        play_cards = []
        play_grid = []
        play_labels = []
        play_other = []

        for idx, g in enumerate(session_games, start=1):
            gaze_log_path = self._resolve_gaze_log_path(g.get("gaze_log_path"))
            if not gaze_log_path or not os.path.exists(gaze_log_path):
                continue

            stats = self._compute_gaze_statistics(gaze_log_path)

            m_samples = stats.get("memorization_samples", 0)
            p_samples = stats.get("play_samples", 0)
            if m_samples == 0 and p_samples == 0:
                continue

            game_indices.append(str(len(game_indices) + 1))

            mem_cards.append(stats.get("memorization_card_percentage", 0.0))
            mem_grid.append(stats.get("memorization_gridFrame_percentage", 0.0))
            mem_labels.append(stats.get("memorization_labels_percentage", 0.0))
            mem_other.append(stats.get("memorization_other_percentage", 0.0))
            play_cards.append(stats.get("play_card_percentage", 0.0))
            play_grid.append(stats.get("play_gridFrame_percentage", 0.0))
            play_labels.append(stats.get("play_labels_percentage", 0.0))
            play_other.append(stats.get("play_other_percentage", 0.0))

        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas, 1)

        if not game_indices:
            ax.text(
                0.5,
                0.5,
                "No gaze data\navailable for this session.",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw()
            return container

        x = np.arange(len(game_indices))

        def draw_phase(phase_name: str):
            ax.clear()

            if phase_name == "Memorization":
                c = np.array(mem_cards)
                g = np.array(mem_grid)
                l = np.array(mem_labels)
                o = np.array(mem_other)
                phase_title = "Memorization phase"
            else:
                c = np.array(play_cards)
                g = np.array(play_grid)
                l = np.array(play_labels)
                o = np.array(play_other)
                phase_title = "Play phase"

            bottom1 = c
            bottom2 = c + g
            bottom3 = c + g + l

            ax.bar(x, c, label="Cards")
            ax.bar(x, g, bottom=bottom1, label="Grid frame")
            ax.bar(x, l, bottom=bottom2, label="Timer & status labels")
            ax.bar(x, o, bottom=bottom3, label="Other")

            ax.set_ylim(0, 100)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_ylabel("% of gaze samples")
            ax.set_xlabel("Game number")
            ax.set_xticks(x)
            ax.set_xticklabels(game_indices)
            ax.legend(loc="upper right")

            canvas.draw_idle()

        draw_phase("Memorization")

        def on_phase_changed(text):
            draw_phase(text)

        phase_combo.currentTextChanged.connect(on_phase_changed)

        return container

    def _plot_gaze_per_card(self, gaze_log_path: str) -> FigureCanvas:
        rows = self._load_gaze_log_rows(gaze_log_path)
        mem = {}
        play = {}
        mem_total = 0
        play_total = 0
        for r in rows:
            if r.get("event_type") != "gaze_sample":
                continue
            phase = (r.get("phase") or "").strip()
            if r.get("element_type") != "card":
                continue
            card_id = r.get("card_id")
            if card_id in (None, "", "None"):
                continue

            if phase == "memorization":
                mem_total += 1
                mem[card_id] = mem.get(card_id, 0) + 1
            elif phase == "play":
                play_total += 1
                play[card_id] = play.get(card_id, 0) + 1

        card_ids = sorted(set(mem.keys()) | set(play.keys()))
        mem_pct = [(mem.get(cid, 0) / mem_total * 100) if mem_total else 0 for cid in card_ids]
        play_pct = [(play.get(cid, 0) / play_total * 100) if play_total else 0 for cid in card_ids]

        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        if not card_ids or (mem_total == 0 and play_total == 0):
            ax.text(
                0.5, 0.5,
                "No gaze-on-card samples found\n(for memorization/play)",
                ha="center", va="center"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        x = np.arange(len(card_ids))
        w = 0.40
        ax.bar(x - w / 2, mem_pct, width=w, label="Memorization")
        ax.bar(x + w / 2, play_pct, width=w, label="Play")
        ax.set_xlabel("Card ID")
        ax.set_ylabel("% of gaze samples on cards")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in card_ids])
        ax.legend()

        return FigureCanvas(fig)

    def _plot_gaze_before_matched(self, gaze_log_path: str, window_ms: int = 3000) -> FigureCanvas:
        rows = self._load_gaze_log_rows(gaze_log_path)

        gaze_samples = []
        match_clicks = []

        for r in rows:
            et = r.get("event_type")
            if et == "gaze_sample":
                if (r.get("phase") == "play") and (r.get("element_type") == "card") and (
                        r.get("card_id") not in (None, "", "None")):
                    gaze_samples.append(r)
            elif et == "click":
                if r.get("phase") == "play" and r.get("matched") == 1 and r.get("card_id") not in (None, "", "None"):
                    match_clicks.append(r)

        if not match_clicks:
            for r in rows:
                if r.get("matched") == 1 and r.get("card_id") not in (None, "", "None"):
                    if r.get("game_time_ms") is not None:
                        match_clicks.append(r)

        per_card_seconds = {}
        sample_period_s = 0.05
        gaze_samples.sort(key=lambda r: r.get("game_time_ms", 0))

        for c in match_clicks:
            t = c.get("game_time_ms")
            cid = c.get("card_id")
            if t is None or cid is None:
                continue

            start = t - window_ms
            count = 0
            for g in gaze_samples:
                gt = g.get("game_time_ms")
                if gt is None:
                    continue
                if gt < start:
                    continue
                if gt > t:
                    break
                if g.get("card_id") == cid:
                    count += 1

            per_card_seconds[cid] = per_card_seconds.get(cid, 0) + count * sample_period_s

        card_ids = sorted(per_card_seconds.keys())
        values = [per_card_seconds[c] for c in card_ids]

        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        if not card_ids:
            ax.text(0.5, 0.5, "No matched clicks found\n(or missing game_time_ms/card_id)", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        ax.bar([str(c) for c in card_ids], values)
        ax.set_xlabel("Card ID")
        ax.set_ylabel(f"Seconds in last {window_ms / 1000:.0f}s (approx)")

        return FigureCanvas(fig)

    def _plot_correct_vs_incorrect(self, gaze_log_path: str, window_ms: int = 1000) -> FigureCanvas:
        rows = self._load_gaze_log_rows(gaze_log_path)

        gaze_samples = [r for r in rows
                        if r.get("event_type") == "gaze_sample"
                        and r.get("phase") == "play"
                        and r.get("game_time_ms") is not None]

        clicks = [r for r in rows
                  if r.get("phase") == "play"
                  and r.get("matched") in (0, 1)
                  and r.get("game_time_ms") is not None]

        if not clicks:
            clicks = [r for r in rows
                      if r.get("matched") in (0, 1)
                      and r.get("game_time_ms") is not None]

        gaze_samples.sort(key=lambda r: r.get("game_time_ms", 0))
        sample_period_s = 0.05
        correct = []
        incorrect = []

        for c in clicks:
            t = c["game_time_ms"]
            start = t - window_ms

            on_cards = 0
            total = 0

            for g in gaze_samples:
                gt = g.get("game_time_ms")
                if gt < start:
                    continue
                if gt > t:
                    break
                total += 1
                if g.get("element_type") == "card":
                    on_cards += 1

            seconds_on_cards = on_cards * sample_period_s
            if c.get("matched") == 1:
                correct.append(seconds_on_cards)
            else:
                incorrect.append(seconds_on_cards)

        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        data = []
        labels = []
        if correct:
            data.append(correct)
            labels.append("Correct")
        if incorrect:
            data.append(incorrect)
            labels.append("Incorrect")

        if data:
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_ylabel(f"Seconds on cards in last {window_ms / 1000:.0f}s (approx)")
        else:
            ax.text(0.5, 0.5, "No click/match data found", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        return FigureCanvas(fig)

    def _plot_gaze_over_time(self, gaze_log_path: str, bin_ms: int = 1000) -> FigureCanvas:
        rows = self._load_gaze_log_rows(gaze_log_path)

        play_gaze = [r for r in rows
                     if r.get("event_type") == "gaze_sample"
                     and r.get("phase") == "play"
                     and r.get("game_time_ms") is not None]

        if not play_gaze:
            fig = Figure(figsize=(5, 3), tight_layout=True)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No play gaze samples", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        max_t = max(r["game_time_ms"] for r in play_gaze)
        n_bins = max(1, int(max_t // bin_ms) + 1)

        total = np.zeros(n_bins, dtype=int)
        on_cards = np.zeros(n_bins, dtype=int)

        for r in play_gaze:
            b = int(r["game_time_ms"] // bin_ms)
            total[b] += 1
            if r.get("element_type") == "card":
                on_cards[b] += 1

        pct = np.where(total > 0, on_cards / total * 100.0, 0.0)
        x = np.arange(n_bins) * (bin_ms / 1000.0)

        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(x, pct)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("% gaze samples on cards")

        return FigureCanvas(fig)

    def _plot_session_efficiency(self, session_games) -> FigureCanvas:
        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)
        MIN_MOVE_TIME = 1.5
        indices = []
        efficiencies = []
        paces = []

        for idx, g in enumerate(session_games, start=1):
            num_cards = g.get("num_cards", 0) or 0
            moves = g.get("moves", 0) or 0
            time_s = g.get("time_seconds", 0) or 0

            if num_cards <= 0 or moves <= 0 or time_s <= 0:
                continue

            pairs = num_cards / 2.0
            eff = pairs / float(moves)
            pace = moves * MIN_MOVE_TIME / float(time_s)

            indices.append(len(indices) + 1)
            efficiencies.append(eff)
            paces.append(pace)

        if not indices:
            ax.text(
                0.5, 0.5,
                "No sufficient data\nfor efficiency trend.",
                ha="center", va="center"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        ax.plot(indices, efficiencies, marker="o", label="Efficiency (pairs / moves)")
        ax.plot(indices, paces, marker="s", label="Game pace index")

        for x, y in zip(indices, efficiencies):
            ax.text(
                x,
                y + 0.01,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        for x, y in zip(indices, paces):
            ax.text(
                x,
                y + 0.01,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_xlabel("Game number")
        ax.set_ylabel("Index value")

        ax.set_xticks(indices)
        ax.set_xlim(min(indices) - 0.5, max(indices) + 0.5)

        ax.set_ylim(0.0, 1.08)

        ax.legend()

        return FigureCanvas(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory Game with Eye Tracking")
    parser.add_argument("--dev", action="store_true", help="Enable developer mode")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-family: 'Segoe UI'; }")
    win = MemoryGameWindow(dev_mode=args.dev)
    win.showMaximized()
    sys.exit(app.exec_())
