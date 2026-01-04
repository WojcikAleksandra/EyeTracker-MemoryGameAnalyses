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


# --- Import setup for local modules ---
# We add extra paths so the app can import gaze tracking modules that live outside this folder.
sys.path.append("..")
sys.path.append("../GazeLocalization")
sys.path.append("/pages")
try:
    # Calibration screen UI + gaze logger + heatmap viewer + helpers for filesystem paths.
    from MemoryGame_App.pages.calibration_screen import CalibrationScreen
    from gaze_data_logger import GazeDataLogger
    from MemoryGame_App.pages.heatmap_view import HeatmapWindow
    from app_data_paths import (
        get_app_data_dir, get_images_dir, get_haar_cascade_path,
        get_game_history_path, get_gaze_data_dir, get_click_log_path,
        get_latest_archived_gaze_file_path
    )
except ImportError as e:
    # Fail fast: without gaze tracking dependencies the app should not run, because core features rely on those modules.
    print(f"ERROR: Required gaze tracking modules not available: {e}")
    print("\nPlease install required dependencies:")
    print("  pip install opencv-python scikit-learn numpy")
    print("\nApplication cannot start without gaze tracking support.")
    sys.exit(1)


# ========================= #
#       Style Constants     #
# ========================= #
class Styles:
    """UI stylesheet constants used across the application (Qt style strings)."""
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

    # Small text labels (HUD = heads-up display) used for time / moves
    HUD = "font-size: 18px; color: #333;"
    SUBTITLE = "font-size: 26px; font-weight: 600; color: #8549c9;"

    # Styling for the card buttons (front/back images shown via icon)
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

    # Frame around the board grid
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

    # Used for plot / heatmap containers
    PLOT_BOX = """
        QFrame {
            background-color: #E0E0E0;
            border-radius: 12px;
            border: 2px solid #CCCCCC;
        }
    """


# ========================= #
#     Memory Game Board     #
# ========================= #
class MemoryGameBoard(QWidget):
    """
    Main game board widget:
      - Displays a grid of card buttons (QPushButton)
      - Manages game state (flipped cards, matched cards, timer, moves)
      - Optionally (if eye tracking enabled) samples gaze data and logs gaze + clicks for later analysis
    """
    GRID_SPACING = 8
    PREVIEW_MS = 5000
    FLIP_CHECK_DELAY_MS = 800

    def __init__(self, num_cards=8, gaze_engine=None, gaze_logger=None):
        """Initialize the board, create UI, and prepare timers/loggers."""
        super().__init__()

        # --- game configuration ---
        self.num_cards = num_cards

        # Compute board layout (rows x cols) based on number of cards.
        # For divisible-by-3 counts, prefer 3 rows; otherwise use 2 rows.
        self.rows, self.cols = ((3, num_cards // 3) if num_cards % 3 == 0 else (2, num_cards // 2))

        # --- core game state ---
        self.cards = []      # list[QPushButton] all card buttons
        self.flipped = []    # list[QPushButton] currently flipped (max 2)
        self.matched = []    # list[QPushButton] already matched and kept face-up
        self.locked = True   # when True, player input is ignored (preview / check animation)
        self.elapsed = 0     # elapsed gameplay time in seconds
        self.moves = 0       # number of pair attempts
        self.game_finished = False

        # --- geometry for gaze mapping (screen-global coordinates) ---
        # These store the board rectangle and each card rectangle in screen coordinates
        # so that gaze predictions can be mapped reliably.
        self.board_rect_screen = None
        self.card_rects_screen = {}

        # Debug flag
        self.debug_hitboxes = False  #false -> mozna usunac paintEvent, i w update_hitboxes usunac self_update

        # -- gaze tracking --
        self.gaze_engine = gaze_engine
        self.gaze_logger = gaze_logger

        # Timer that periodically samples gaze position (when tracking_active True).
        self.gaze_timer = QTimer(self)
        self.gaze_timer.timeout.connect(self._sample_gaze)

        self.tracking_active = False
        # current game phase is logged with gaze samples, so analysis can separate phases
        self.current_phase = None  # "memorization" or "play"

        # -- game timers --
        self.preview_timer = QTimer(self)  # handles the initial memorization countdown
        self.preview_timer.timeout.connect(self._update_preview)

        self.game_timer = QTimer(self)  # increments elapsed time during play phase
        self.game_timer.timeout.connect(self._update_timer)

        # --- logging click data ---
        self.log_file_path = get_click_log_path()
        self.log_file = None

        # click_counter alternates 1/2 to represent "first card" vs "second card" in a pair
        self.click_counter = 2
        self.game_start_time = None  # QTime when preview ends and gameplay starts

        # -- build UI and create cards --
        self._build_ui()
        self._create_cards()

    # ---------- UI ----------
    def _build_ui(self):
        """Create board widgets (labels + grid container) and load card images."""
        layout = QVBoxLayout(self)
        layout.setSpacing(30)

        # Status label shows instructions and preview countdown.
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
        images_dir = get_images_dir()
        self.front_images = [os.path.join(images_dir, f"{i}.png") for i in range(1, self.num_cards // 2 + 1)] * 2
        random.shuffle(self.front_images)
        self.back_image = os.path.join(images_dir, "backOfCard.png")

    def _create_cards(self):
        """Create card buttons, attach metadata, and add them to the grid."""
        for i, img in enumerate(self.front_images):
            btn = QPushButton()
            btn.setStyleSheet(Styles.CARD_BUTTON)
            btn.image_path = img
            btn.card_index = i
            btn.card_row = i // self.cols
            btn.card_col = i % self.cols

            # Override click handler to capture exact click coordinates
            btn.mousePressEvent = lambda event, b=btn: self._on_card_mouse_press(b, event)

            btn.setIcon(QIcon(img))  # face-up during preview
            self.grid.addWidget(btn, btn.card_row, btn.card_col)
            self.cards.append(btn)

    def update_hitboxes(self):
        """Recompute board and card rectangles in screen-global coordinates (to compare with eye-tracking data)."""
        top_left_board = self.grid_frame.mapToGlobal(QPoint(0, 0))
        self.board_rect_screen = QRect(top_left_board, self.grid_frame.size())

        # rectangles for each card
        self.card_rects_screen.clear()
        for btn in self.cards:
            top_left = btn.mapToGlobal(QPoint(0, 0))
            rect = QRect(top_left, btn.size())
            self.card_rects_screen[btn] = rect

        #self.update()

    @property
    def elapsed_seconds(self):
        """Return elapsed gameplay time in seconds."""
        return self.elapsed

    @property
    def move_count(self):
        """Return the current move count (pair attempts)."""
        return self.moves

    def resizeEvent(self, event):
        """Keep cards square when resizing and refresh gaze hitboxes."""
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
    def start_memorize_phase(self):
        """
        Start the game preview phase:
          - reset moves/time
          - show all cards face-up
          - run a countdown (PREVIEW_MS)
          - start gaze logging in "memorization" phase
        """
        self.moves = 0
        self.elapsed = 0
        self._update_hud()

        QTimer.singleShot(0, self.update_hitboxes)

        self._preview_deadline_ms = self.PREVIEW_MS
        self._preview_start = QTime.currentTime()

        self.locked = True
        self.preview_timer.start(100)

        # Start gaze tracking (if calibrated)
        self.current_phase = "memorization"
        self._start_gaze_tracking()
        
        # Log phase start
        if self.gaze_logger:
            timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
            self.gaze_logger.log_phase_event(timestamp_ms, "memorization", "phase_start", 0)

    def _update_preview(self):
        """
        Update the preview countdown label.
        When countdown ends:
          - flip all cards to back
          - unlock the board
          - start gameplay timer + click logging
          - switch gaze phase to 'play'
        """
        remaining = max(0, self._preview_deadline_ms - self._preview_start.msecsTo(QTime.currentTime()))
        seconds = remaining // 1000 + 1

        self.status_label.setText(f"Memorize the cards! {seconds}")

        if remaining <= 0:
            self.preview_timer.stop()
            self._flip_all_to_back()
            self.status_label.setText("Now find the pairs!")
            self.locked = False
            self._start_game_timer()
            
            # Log phase transition: memorization -> play
            if self.gaze_logger:
                timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
                self.gaze_logger.log_phase_event(timestamp_ms, "memorization", "phase_end", 0)
                self.current_phase = "play"
                self.gaze_logger.log_phase_event(timestamp_ms, "play", "phase_start", 0)

    def _start_game_timer(self):
        """Start gameplay timer and open click log file."""
        self.elapsed = 0
        self._update_hud()
        self.game_timer.start(1000)

        # Start click logging: ms since start, click position, flip number, matched flag, and card id
        self.game_start_time = QTime.currentTime()
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        self.log_file.write("ms,x,y,flip,matched,card_id\n")

    def _update_timer(self):
        """Called once per second: increase elapsed time and refresh HUD."""
        self.elapsed += 1
        self._update_hud()

    def _update_hud(self):
        """Update the on-screen HUD text (time + moves)."""
        self.timer_label.setText(f"Time: {self.elapsed}s | Moves: {self.moves}")

    def _flip_all_to_back(self):
        """Flip all cards to the back image."""
        for c in self.cards:
            c.setIcon(QIcon(self.back_image))

    # ---------- interactions ----------
    def _on_card_mouse_press(self, btn, event):
        """Handle mouse press on card button - captures actual click position."""
        # Ignore clicks while board is locked (preview or checking a pair).
        if self.locked:
            QPushButton.mousePressEvent(btn, event)
            return

        # Ignore clicks on already matched cards or already flipped cards.
        if btn in self.matched or btn in self.flipped:
            QPushButton.mousePressEvent(btn, event)
            return

        # Get actual click position in screen-global coordinates
        click_pos_global = btn.mapToGlobal(event.pos())
        click_x = click_pos_global.x()
        click_y = click_pos_global.y()

        # Call parent's mousePressEvent to ensure normal button behavior
        QPushButton.mousePressEvent(btn, event)

        # Flip chosen card face-up and store it in "flipped" list.
        btn.setIcon(QIcon(btn.image_path))
        self.flipped.append(btn)

        # If this is the first card in the pair, log immediately (flip=1, matched=0).
        if len(self.flipped) == 1:
            self.log_click(btn, matched_flag=0, click_x=click_x, click_y=click_y)

        # If two cards are flipped, lock input and schedule match checking.
        if len(self.flipped) == 2:
            self.locked = True
            QTimer.singleShot(self.FLIP_CHECK_DELAY_MS, lambda: self._check_match(click_x, click_y))

    def _check_match(self, click_x=None, click_y=None):
        """
        Compare the two flipped cards:
          - If equal: mark as matched and keep face-up
          - Else: flip both back down
        Also logs the second card click with matched_flag 0/1.
        """
        a, b = self.flipped
        self.moves += 1
        self._update_hud()

        if a.image_path == b.image_path:
            self.matched += [a, b]
            self.status_label.setText("Nice! You found a pair!")
            # LOG CLICK (flip=2, matched=1)
            # Use provided click position or get current cursor position as fallback
            if click_x is None or click_y is None:
                cursor_pos = QCursor.pos()
                click_x, click_y = cursor_pos.x(), cursor_pos.y()
            self.log_click(b, matched_flag=1, click_x=click_x, click_y=click_y)

        else:
            self.status_label.setText("Try again!")
            for btn in self.flipped:
                btn.setIcon(QIcon(self.back_image))
            # LOG CLICK (flip=2, matched=0)
            # Use provided click position or get current cursor position as fallback
            if click_x is None or click_y is None:
                cursor_pos = QCursor.pos()
                click_x, click_y = cursor_pos.x(), cursor_pos.y()
            self.log_click(b, matched_flag=0, click_x=click_x, click_y=click_y)

        self.flipped.clear()
        self.locked = False

        # End game when all cards are matched
        if len(self.matched) == len(self.cards):
            self._finish_game()

    def _finish_game(self):
        """
        End-of-game cleanup:
          - stop timers
          - stop gaze tracking + close gaze logging
          - save results (via parent window)
          - navigate to win page
        """
        self.game_finished = True
        self.locked = True
        if self.game_timer.isActive():
            self.game_timer.stop()
        self.status_label.setText("You found all pairs! Great job!")
        
        # Stop gaze tracking
        self._stop_gaze_tracking()
        
        # Log phase end and store log path for heatmap
        if self.gaze_logger:
            timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
            game_time_ms = self.game_start_time.msecsTo(QTime.currentTime()) if self.game_start_time else 0
            self.gaze_logger.log_phase_event(timestamp_ms, "play", "phase_end", game_time_ms)
            
            # Store log path in parent window for heatmap access
            main_window = self.window()
            if main_window and hasattr(main_window, 'last_game_info'):
                log_path = self.gaze_logger.get_log_file_path()
                main_window.last_game_info["gaze_log_path"] = log_path
                print(f"Gaze data saved to: {log_path}")
            
            self.gaze_logger.stop_logging()
            print(f"Gaze logging stopped. File should be saved at: {self.gaze_logger.get_log_file_path()}")
        
        # Save game results for leaderboard
        main_window = self.window()
        if main_window and hasattr(main_window, 'save_game_result'):
            main_window.save_game_result(self.elapsed, self.moves, self.num_cards)
        
        # Note: Keep calibration for multiple games - don't close gaze engine here
        # Camera stays ready for next game

        # Navigate to win page after a short delay (lets the user see the last message)
        QTimer.singleShot(1500, lambda: getattr(self.window(), "show_win_page", lambda: None)())

        # Close click log file
        if self.log_file:
            self.log_file.close()
            self.log_file = None


    def stop_all_timers(self):
        """
        Stop everything related to active gameplay (used when leaving the page).
        Safe to call multiple times.
        """
        self.preview_timer.stop()
        self.game_timer.stop()
        self._stop_gaze_tracking()
        self.locked = True

        if self.log_file:
            self.log_file.close()
            self.log_file = None

        if not self.game_finished and self.log_file_path and os.path.exists(self.log_file_path):
            # try:
            #     os.remove(self.log_file_path)
            # except Exception as e:
            #     print("Could not delete click log:", e)
            # stop gaze logger and delete gaze file if aborted
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

        # Note: Don't close gaze engine here - keep calibration for multiple games


    # def paintEvent(self, event):
    #     # najpierw normalne rysowanie
    #     super().paintEvent(event)
    #
    #     if not self.debug_hitboxes:
    #         return
    #     if self.board_rect_screen is None:
    #         return
    #
    #     painter = QPainter(self)
    #     pen = QPen(Qt.red)
    #     pen.setWidth(3)
    #     painter.setPen(pen)
    #
    #     # 1) narysuj ramkę całej planszy (grid_frame)
    #     top_left_local = self.mapFromGlobal(self.board_rect_screen.topLeft())
    #     board_rect_local = QRect(top_left_local, self.board_rect_screen.size())
    #     painter.drawRect(board_rect_local)
    #
    #     # 2) narysuj prostokąty po każdej karcie
    #     for rect_screen in self.card_rects_screen.values():
    #         tl_local = self.mapFromGlobal(rect_screen.topLeft())
    #         card_rect_local = QRect(tl_local, rect_screen.size())
    #         painter.drawRect(card_rect_local)

    # ---------- Gaze tracking ----------
    def _start_gaze_tracking(self):
        """Start gaze tracking timer."""
        if self.gaze_engine and self.gaze_engine.is_calibrated():
            self.tracking_active = True
            self.gaze_timer.start(50)  # ~20 FPS

    def _stop_gaze_tracking(self):
        """Stop gaze tracking timer."""
        self.tracking_active = False
        self.gaze_timer.stop()

    def _sample_gaze(self):
        """Sample gaze position and log it."""
        if not self.tracking_active or not self.gaze_engine or not self.gaze_logger:
            return

        gaze = self.gaze_engine.predict_gaze()
        if gaze is None:
            return

        gaze_x, gaze_y = gaze
        timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
        game_time_ms = self.game_start_time.msecsTo(QTime.currentTime()) if self.game_start_time else 0

        # Ensure hitboxes are updated before detecting elements
        if self.board_rect_screen is None:
            self.update_hitboxes()

        # Convert gaze from window-relative to screen-global coordinates
        # The gaze engine predicts in window coordinates, but hitboxes are in screen coordinates
        window = self.window()
        if window:
            window_pos = window.mapToGlobal(QPoint(0, 0))
            gaze_screen_x = gaze_x + window_pos.x()
            gaze_screen_y = gaze_y + window_pos.y()
        else:
            gaze_screen_x, gaze_screen_y = gaze_x, gaze_y

        # Detect element at gaze point (using screen coordinates)
        element_info = self._get_element_at_point(gaze_screen_x, gaze_screen_y)

        # Log using screen coordinates for consistency with click coordinates
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
        """Determine what UI element is at the given point."""
        point = QPoint(x, y)
        result = {
            "element_type": "other",
            "card_row": None,
            "card_col": None,
            "card_id": None,
            "card_image_name": None,
        }

        # Check if point is within board area
        if self.board_rect_screen and self.board_rect_screen.contains(point):
            # Check each card
            for btn, rect in self.card_rects_screen.items():
                if rect.contains(point):
                    result["element_type"] = "card"
                    result["card_row"] = btn.card_row + 1  # 1-indexed
                    result["card_col"] = btn.card_col + 1  # 1-indexed
                    
                    # Extract card ID and image name
                    image_path = getattr(btn, "image_path", "")
                    if image_path:
                        filename = image_path.rsplit("/", 1)[-1]  # FULL PATH to the card image
                        result["card_image_name"] = filename
                        base = os.path.basename(filename)  # "3.png"
                        name, _ = os.path.splitext(base)  # ("3", ".png")
                        # name = filename.split(".", 1)[0]  # "3"
                        if name.isdigit():
                            result["card_id"] = int(name)
                    return result

            # Point is in board area but not on a card
            result["element_type"] = "grid_frame"
            return result

        # Check other UI elements (status_label, timer_label)
        # Convert global point to local coordinates for label checks
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
        """Extract card information from button."""
        result = {
            "card_row": None,
            "card_col": None,
            "card_id": None,
            "card_image_name": None,
        }

        if hasattr(btn, "card_row") and hasattr(btn, "card_col"):
            result["card_row"] = btn.card_row + 1  # 1-indexed
            result["card_col"] = btn.card_col + 1  # 1-indexed

        image_path = getattr(btn, "image_path", "")
        if image_path:
            filename = image_path.rsplit("/", 1)[-1]  # FULL PATH to the card image
            result["card_image_name"] = filename

            base = os.path.basename(filename) # "3.png"
            name, _ = os.path.splitext(base)  # ("3", ".png")
            #name = filename.split(".", 1)[0]  # "3"
            if name.isdigit():
                result["card_id"] = int(name)

        return result

    def log_click(self, btn, matched_flag=0, click_x=None, click_y=None):
        """Loguje: time from start of game, click coordinates,
        flip 1/2, matched_flag 0/1, card_id (for example: 3 z images/3.png)"""
        now = QTime.currentTime()
        ms = self.game_start_time.msecsTo(now) if self.game_start_time else 0

        # Use actual click position if provided, otherwise fallback to card center
        if click_x is not None and click_y is not None:
            x = click_x
            y = click_y
        else:
            # Fallback
            rect = self.card_rects_screen.get(btn)
            if rect:
                x = rect.center().x()
                y = rect.center().y()
            else:
                x = y = -1

        # Parse card_id from the image filename "3.png" -> 3
        card_id = -1
        image_path = getattr(btn, "image_path", "")
        if image_path:
            filename = image_path.rsplit("/", 1)[-1]  # "3.png" -> NO, FULL PATH to the card image

            base = os.path.basename(filename) # "3.png"
            name, _ = os.path.splitext(base)  # ("3", ".png")
            #name = filename.split(".", 1)[0]  # "3"
            if name.isdigit():
                card_id = int(name)

        # Alternate flip number between 1 and 2
        self.click_counter = 1 if self.click_counter == 2 else 2

        # Legacy CSV logging
        if self.log_file:
            self.log_file.write(
                f"{ms},{x},{y},{self.click_counter},{matched_flag},{card_id}\n"
            )

        # Enhanced gaze logger
        if self.gaze_logger:
            # Get current gaze position and convert to screen coordinates
            gaze_screen_x, gaze_screen_y = None, None
            if self.gaze_engine:
                gaze = self.gaze_engine.predict_gaze()
                if gaze:
                    gaze_x, gaze_y = gaze
                    # Convert to screen coordinates
                    window = self.window()
                    if window:
                        window_pos = window.mapToGlobal(QPoint(0, 0))
                        gaze_screen_x = gaze_x + window_pos.x()
                        gaze_screen_y = gaze_y + window_pos.y()
                    else:
                        gaze_screen_x, gaze_screen_y = gaze_x, gaze_y

            # Get card info
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


# ========================= #
#      Main Game Window     #
# ========================= #
class MemoryGameWindow(QMainWindow):
    """
    Main application window.
    Manages navigation between pages (home / calibration / countdown / board / stats),
    controls window resizing rules during gameplay, and stores game + gaze history.
    """
    # Session ID for this app run
    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Path to the JSON file where game history (leaderboard) is stored
    GAME_HISTORY_FILE = get_game_history_path()

    def __init__(self, dev_mode=False):
        """Initialize the main window, state variables, and build the initial UI."""
        super().__init__()
        self.setWindowTitle("Memory Game")
        self.dev_mode = dev_mode

        # Countdown state (used when transitioning from Home -> Game)
        self._countdown_timer = None
        self._countdown_page = None
        self._countdown_cancelled = False
        self._auto_nav_enabled = True  # prevents auto-navigation after user leaves a page
        self.board_page = None

        # Window resize lock state (prevents resizing during the game)
        self._resize_locked = False
        self._old_min_size = QSize(0, 0)
        self._old_max_size = QSize(16777215, 16777215)

        # Eye-tracking / gaze tracking state
        self.gaze_engine = None
        self.gaze_logger = None
        self.calibration_done = False
        self.pending_num_cards = None

        # Store last game info so the stats page / heatmap can access it
        self.last_game_info = {
            "num_cards": 8,
            "front_images": [],
            "gaze_log_path": None,
            "board_size": None,
        }

        # Heatmap window reference
        self.heatmap_window = None

        # Game history for leaderboard
        self.game_history = self._load_game_history()

        # Archive previous gaze data files on startup (keeps the main folder clean)
        gaze_data_dir = get_gaze_data_dir()
        for f in os.listdir(gaze_data_dir):
            path = os.path.join(gaze_data_dir, f)
            if os.path.isfile(path):
                shutil.move(path, gaze_data_dir + "/archived/" + f)
                #os.remove(path)

        if self.dev_mode:
            self.setWindowTitle("Memory Game [DEV MODE]")

        self._build_ui()

    # ---- small helpers ----
    def _add_menu_action(self, menu, text, slot):
        """Create and add a QAction to a menu."""
        act = QAction(text, self, triggered=slot)
        menu.addAction(act)
        return act

    def _lock_window_resize(self):
        """Lock the window size (used during gameplay / calibration screens)."""
        if self._resize_locked:
            return
        self._resize_locked = True

        # Remember old constraints so we can restore them later
        self._old_min_size = self.minimumSize()
        self._old_max_size = self.maximumSize()

        # Fix the window to its current size
        current_size = self.size()
        self.setMinimumSize(current_size)
        self.setMaximumSize(current_size)

    def _unlock_window_resize(self):
        """Restore window resize limits to what they were before locking."""
        if not self._resize_locked:
            return
        self._resize_locked = False

        self.setMinimumSize(self._old_min_size)
        self.setMaximumSize(self._old_max_size)

    # ---- UI ----
    def _build_ui(self):
        """Build the window layout, menu bar, and the stacked page container."""
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

         # Stacked widget to hold different pages

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.show_home_page()

    def show_home_page(self):
        """Show the home page (instructions + difficulty selector)."""
        self._abort_activity()

        # page = QWidget()
        # layout = QVBoxLayout(page)
        # layout.setContentsMargins(80, 20, 80, 60)
        # layout.setSpacing(30)
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

        scroll.setWidget(content_widget)
        page_layout.addWidget(scroll)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def _on_play_clicked(self, num_cards: int):
        """Handle play button click - check calibration first."""
        self.pending_num_cards = num_cards

        if not self.calibration_done:
            # Show message box asking for calibration
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
                # User declined calibration - proceed without gaze tracking
                self.calibration_done = True
                self.gaze_engine = None
                self.show_countdown(num_cards)
        else:
            # Calibration already done, proceed to game
            self.show_countdown(num_cards)

    def show_calibration_screen(self):
        """Show calibration screen."""
        self._lock_window_resize()
        self._auto_nav_enabled = True

        # Get window size for calibration (use geometry for accurate size)
        window_size = (self.geometry().width(), self.geometry().height())
        if window_size[0] <= 0 or window_size[1] <= 0:
            window_size = (self.width(), self.height())

        calibration_page = CalibrationScreen(window_size, self, dev_mode=self.dev_mode)
        self.stack.addWidget(calibration_page)
        self.stack.setCurrentWidget(calibration_page)

        # Start calibration after a short delay to ensure UI is ready
        QTimer.singleShot(100, calibration_page.start_calibration)

        # Check calibration status periodically
        self._calibration_check_timer = QTimer(self)
        self._calibration_check_timer.timeout.connect(
            lambda: self._check_calibration_status(calibration_page)
        )
        self._calibration_check_timer.start(500)  # Check every 500ms

    def _check_calibration_status(self, calibration_page):
        """Check if calibration is complete and handle result."""
        if not calibration_page.calibration_complete:
            return

        # Stop checking
        if hasattr(self, '_calibration_check_timer'):
            self._calibration_check_timer.stop()

        success = calibration_page.is_successful()
        self.gaze_engine = calibration_page.get_gaze_engine()

        # Show result message
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

        # Remove calibration page
        idx = self.stack.indexOf(calibration_page)
        if idx != -1:
            self.stack.removeWidget(calibration_page)
            calibration_page.deleteLater()

        # Handle user choice
        if success:
            # Proceed to countdown
            if self.pending_num_cards:
                num_cards = self.pending_num_cards
                self.pending_num_cards = None  # Clear pending
                self.show_countdown(num_cards)
        else:
            # User chose retry or skip
            clicked_btn = msg.clickedButton()
            if clicked_btn and clicked_btn.text() == "Retry":
                # Retry calibration
                self.show_calibration_screen()
            else:
                # Skip calibration
                self.calibration_done = True
                self.gaze_engine = None
                if self.pending_num_cards:
                    num_cards = self.pending_num_cards
                    self.pending_num_cards = None  # Clear pending
                    self.show_countdown(num_cards)

    def show_countdown(self, num_cards: int):
        """Show a 3-second countdown screen, then start the game automatically."""
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
        """Create the MemoryGameBoard, start logging (if enabled), and begin the memorize phase."""
        self._auto_nav_enabled = True

        # Initialize gaze logger if gaze engine exists
        gaze_logger = None
        if self.gaze_engine:
            try:
                gaze_data_dir = get_gaze_data_dir()
                gaze_logger = GazeDataLogger(output_dir=gaze_data_dir, app_session_id=self.SESSION_ID)
                gaze_logger.start_logging()
                # Store gaze log path for heatmap
                log_path = gaze_logger.get_log_file_path()
                self.last_game_info["gaze_log_path"] = log_path
                print(f"Gaze data will be saved to: {log_path}")
            except Exception as e:
                print(f"Warning: Could not initialize gaze logger: {e}")
                import traceback
                traceback.print_exc()

        self.board_page = MemoryGameBoard(
            num_cards,
            gaze_engine=self.gaze_engine,
            gaze_logger=gaze_logger
        )

        # Store game info for heatmap
        self.last_game_info["num_cards"] = num_cards
        self.last_game_info["front_images"] = self.board_page.front_images.copy()
        self.last_game_info["board_size"] = (self.width(), self.height())

        self.stack.addWidget(self.board_page)
        self.stack.setCurrentWidget(self.board_page)
        self.board_page.start_memorize_phase()

    def show_win_page(self):
        """Show the win screen after the player matches all pairs."""
        if not self._auto_nav_enabled:
            return

        self._unlock_window_resize()

        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("🎉 You Win! 🎉", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-size: 52px; font-weight: bold; color: #8549c9;")

        subtitle = QLabel("Congratulations, you found all pairs!", alignment=Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 24px; margin-bottom: 30px;")

        # Read final stats from the board page
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

    def _create_leaderboard_widget(self, limit=10, top_n = 5):
        """
        Create a leaderboard widget.
        - limit: number of rows to show
        - top_n: number of results shown in table
        """
        # Outer container (no border)
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

        # --- Header row: [title box] ............. [cards box]
        # --- Header container with fixed height ---
        header_widget = QWidget()
        header_widget.setFixedHeight(60)

        header_row = QHBoxLayout(header_widget)
        header_row.setSpacing(12)
        header_row.setContentsMargins(0, 0, 0, 0)

        # Title box
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

        # Filter box
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

        # Add widgets to header row
        header_row.addWidget(title_box, 1)
        header_row.addWidget(filter_box, 0)

        # Add header widget to main layout
        layout.addWidget(header_widget)


        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Cards", "Time", "Moves", "Date"])
        table.verticalHeader().setVisible(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setDefaultAlignment(Qt.AlignCenter)

        header.setFixedHeight(50)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        #table.setSelectionBehavior(QTableWidget.SelectRows)
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

        # ---- helper to (re)fill table based on selected filter ----
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
                table.setItem(i, 1, QTableWidgetItem(f"{game.get('time_seconds', '?')}s"))
                table.setItem(i, 2, QTableWidgetItem(str(game.get("moves", "?"))))

                timestamp = game.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        date_str = dt.strftime("%m/%d %H:%M")
                    except Exception:
                        date_str = "?"
                else:
                    date_str = "?"
                table.setItem(i, 3, QTableWidgetItem(date_str))

        # initial fill + connect dropdown
        refresh_table("All")
        cards_combo.currentTextChanged.connect(refresh_table)

        return leaderboard_box

    def show_stats_page(self):
        """Show the statistics page with selectable mode (Last Game / All Games)."""
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

        # -------------------------
        # Top bar (title + radio + back)
        # -------------------------
        top = QHBoxLayout()

        title = QLabel("Your Statistics")
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

        # -------------------------
        # Mode container (Last Game vs All Games)
        # -------------------------
        mode_stack = QStackedWidget()
        layout.addWidget(mode_stack, 1)  # stretch

        # ===== No Data container =====
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

        # ===== Last Game container =====
        last_game_container = QWidget()
        last_game_layout = QVBoxLayout(last_game_container)
        last_game_layout.setSpacing(15)

        # -------------------------
        # Top row: key metrics (left) + leaderboard + heatmap (right)
        # -------------------------
        top_row = QHBoxLayout()
        top_row.setSpacing(20)

        # --- LEFT: Key metrics box ---
        metrics_box = QFrame()
        metrics_box.setStyleSheet(Styles.LIGHT_FRAME)
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setSpacing(20)

        if self.game_history:
            last = self.game_history[-1]
            time_taken = last.get("time_seconds", "N/A")
            moves = last.get("moves", "N/A")
        else:
            time_taken = "N/A"
            moves = "N/A"

        # gaze_log_path = self.last_game_info.get("gaze_log_path")
        gaze_log_path = get_latest_archived_gaze_file_path(archived=False)

        if not gaze_log_path:
            gaze_log_path = get_latest_archived_gaze_file_path(archived=True)
        # else:
        #     gaze_log_path = get_latest_archived_gaze_file_path(archived=True)

        gaze_stats = self._compute_gaze_statistics(gaze_log_path)
        print(f"Computing gaze stats from: {gaze_log_path}")

        total_samples = gaze_stats.get("memorization_samples", 0) + gaze_stats.get("play_samples", 0)

        numbers_row = QHBoxLayout()
        numbers_row.setSpacing(18)

        time_lbl = QLabel(f"Time to finish\n{time_taken}s")
        moves_lbl = QLabel(f"Moves\n{moves}")
        samples_lbl = QLabel(f"Gaze samples\n{total_samples}")

        for w in (time_lbl, moves_lbl, samples_lbl):
            w.setAlignment(Qt.AlignCenter)
            w.setStyleSheet("font-size: 20px; color: #333; font-weight: 600;")
            #w.setMinimumWidth(160)
            w.setMinimumWidth(0)  # allow shrinking
            w.setWordWrap(True)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        numbers_wrap = QWidget()
        numbers_wrap.setLayout(numbers_row)
        numbers_row.addWidget(time_lbl)
        numbers_row.addWidget(moves_lbl)
        numbers_row.addWidget(samples_lbl)

        # ------- Gaze distribution text
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

        # --- RIGHT: Leaderboard + Heatmap button under it ---
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        small_leaderboard = self._create_leaderboard_widget(limit=5)
        small_leaderboard.setFixedHeight(360)
        right_col.addWidget(small_leaderboard)

        # HETAMAP ACTIVE ONLY IF THERE WAS A PLAYED GAME
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
            heatmap_btn.setToolTip("Heatmap unavailable: no gaze data found.\nYou must play at least one game for heatmap to be availabla.")
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

        # ========= Plots  ==========
        plots_grid = QGridLayout()
        plots_grid.setSpacing(15)

        def plot_frame(title_text, canvas: FigureCanvas):
            box = QFrame()
            box.setStyleSheet("""
                QFrame {
                    background-color: #FFFFFF;
                    border-radius: 12px;
                    border: 2px solid #B68DDE;
                    padding: 10px;
                }
            """)
            v = QVBoxLayout(box)

            t = QLabel(title_text)
            t.setStyleSheet("font-size: 16px; font-weight: bold; color: #4B2C82;")
            v.addWidget(t)

            # Let plot expand
            canvas.setMinimumHeight(260)
            v.addWidget(canvas, 1)

            return box

        canvas1 = self._plot_gaze_per_card(gaze_log_path)
        canvas2 = self._plot_gaze_before_matched(gaze_log_path, window_ms=3000)
        canvas3 = self._plot_correct_vs_incorrect(gaze_log_path, window_ms=1000)
        canvas4 = self._plot_gaze_over_time(gaze_log_path, bin_ms=1000)

        plots_grid.addWidget(plot_frame("Gaze time per card (Memorization vs Play)", canvas1), 0, 0)
        plots_grid.addWidget(plot_frame("Gaze time on a card before it was matched", canvas2), 0, 1)
        plots_grid.addWidget(plot_frame("Correct vs incorrect gaze comparison", canvas3), 1, 0)
        plots_grid.addWidget(plot_frame("Gaze on cards over time", canvas4), 1, 1)

        last_game_layout.addLayout(plots_grid, 1)

        # ===== All Games container (blank) =====
        all_games_container = QWidget()
        all_games_layout = QVBoxLayout(all_games_container)
        all_games_layout.addStretch(1)  # intentionally blank

        mode_stack.addWidget(no_data_container)  # index 0
        mode_stack.addWidget(last_game_container)  # index 1
        mode_stack.addWidget(all_games_container)  # index 2

        # Radio -> stack switch
        def on_mode_changed():
            if multiple.isChecked():
                mode_stack.setCurrentIndex(2)  # All Games
            else:
                mode_stack.setCurrentIndex(0 if (total_samples == 0) else 1)  # No Data or Last Game

        single.toggled.connect(on_mode_changed)
        multiple.toggled.connect(on_mode_changed)
        on_mode_changed()

        # Show page
        scroll.setWidget(content)
        page_layout.addWidget(scroll)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def _show_heatmap(self):
        """Show heatmap visualization window."""
        # Close existing heatmap window if open
        if self.heatmap_window:
            self.heatmap_window.close()

        # Create game config from last game info
        images_dir = get_images_dir()
        default_images = [os.path.join(images_dir, f"{i}.png") for i in range(1, 5)] * 2
        game_config = {
            "num_cards": self.last_game_info.get("num_cards", 8),
            "front_images": self.last_game_info.get("front_images", default_images),
            "board_size": self.last_game_info.get("board_size", (self.width(), self.height())),
        }

        # Get gaze log path
        gaze_log_path = self.last_game_info.get("gaze_log_path")

        # Create and show heatmap window
        self.heatmap_window = HeatmapWindow(
            gaze_data_path=gaze_log_path,
            game_config=game_config,
            parent=None  # Independent window
        )

        # Size the window appropriately
        board_size = game_config.get("board_size", (800, 600))
        self.heatmap_window.resize(
            max(900, board_size[0]),
            max(700, board_size[1])
        )
        self.heatmap_window.show()

    # ---- control / cleanup ----
    def _abort_activity(self):
        """Stop countdown/game timers and disable any pending auto-navigation."""
        self._auto_nav_enabled = False
        self.cancel_countdown()
        if self.board_page:
            self.board_page.stop_all_timers()
        self._unlock_window_resize()

    def cancel_countdown(self):
        """Cancel an active countdown page/timer and remove it from the stack."""
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

    # ---- Game History / Leaderboard ----
    def _load_game_history(self):
        """Load game history from JSON file."""
        try:
            if os.path.exists(self.GAME_HISTORY_FILE):
                with open(self.GAME_HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading game history: {e}")
        return []

    def _save_game_history(self):
        """Save game history to JSON file."""
        try:
            with open(self.GAME_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.game_history, f, indent=2)
        except Exception as e:
            print(f"Error saving game history: {e}")

    def save_game_result(self, time_seconds, moves, num_cards):
        """Save a game result to history."""
        result = {
            "session_id": self.SESSION_ID,
            "timestamp": datetime.now().isoformat(),
            "time_seconds": time_seconds,
            "moves": moves,
            "num_cards": num_cards,
            "gaze_log_path": self.last_game_info.get("gaze_log_path"),
        }
        self.game_history.append(result)
        self._save_game_history()

    def _clear_game_history(self):
        """Clear all game history."""
        reply = QMessageBox.question(
            self,
            "Clear Game History",
            "Are you sure you want to clear all game history?\n",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:

            gaze_data_dir = get_gaze_data_dir()
            app_data_dir = get_app_data_dir()
            print("============== CLEARING GAME HISTORY ==============")
            print("Clearing gaze data from:", gaze_data_dir)
            for f in os.listdir(gaze_data_dir):
                path = os.path.join(gaze_data_dir, f)
                print("Checking file:", path)
                if os.path.isfile(path):
                    print("Moving file to:", str(Path(app_data_dir) / 'data_cleared' / f))
                    shutil.move(path, str(Path(app_data_dir) / 'data_cleared' / f))
                # path_arch = os.path.join(gaze_data_dir + "/archived/", f)
                # if os.path.isfile(path_arch):
                if os.path.isdir(path):
                    print("Found archived directory:", path)
                    for fa in os.listdir(path):
                        path_a = os.path.join(path, fa)
                        print("Moving archived file to:", str(Path(app_data_dir) / 'data_cleared' / "archived" / fa))
                        shutil.move(path_a, str(Path(app_data_dir) / 'data_cleared' / "archived" / fa))
            history_data_path = get_game_history_path()
            print("Clearing game history from:", history_data_path)
            if os.path.isfile(history_data_path):
                print("Moving history file to:", str(Path(app_data_dir) / 'data_cleared' / 'game_history.json'))
                shutil.move(history_data_path, str(Path(app_data_dir) / 'data_cleared' / 'game_history.json'))

            print("All gaze data and game history cleared.")
            self.game_history = []
            self._save_game_history()
            QMessageBox.information(self, "Cleared", "Game history has been cleared.")

    def _restore_game_history(self):
        """Restore previously cleared game history."""
        print("============== RESTORING GAME HISTORY ==============")
        cleared_data_dir = Path(get_app_data_dir()) / 'data_cleared'
        for f in os.listdir(cleared_data_dir):
            src_path = cleared_data_dir / f
            print("Checking cleared file:", src_path)
            if os.path.isfile(src_path):
                if f == 'game_history.json':
                    dest_path = Path(get_game_history_path())
                else:
                    dest_path = Path(get_gaze_data_dir()) / f
                print("Restoring file to:", dest_path)
                shutil.move(str(src_path), str(dest_path))
            elif os.path.isdir(src_path):
                for fa in os.listdir(src_path):
                    src_path_a = src_path / fa
                    dest_path_a = Path(get_gaze_data_dir()) / "archived" / fa
                    print("Restoring archived file:", src_path_a,  ", to:", dest_path_a)
                    shutil.move(str(src_path_a), str(dest_path_a))
        print("Game history and gaze data restored.")
        self.game_history = self._load_game_history()
        QMessageBox.information(self, "Restored", "Game history has been restored.")

    def _recalibrate(self):
        """Force recalibration of eye tracking."""
        # Close existing gaze engine if any
        if self.gaze_engine:
            self.gaze_engine.close()
            self.gaze_engine = None
        self.calibration_done = False

        QMessageBox.information(
            self,
            "Recalibration",
            "Eye tracking calibration has been reset.\n\n"
            "You will be prompted to recalibrate when you start the next game."
        )

    def _compute_gaze_statistics(self, gaze_log_path):
        """Compute gaze statistics from log file."""
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
                        elif element_type == "gridFrame":
                            stats["memorization_on_gridFrame"] += 1
                        elif element_type == "status label" or element_type == "timer label":
                            stats["memorization_on_labels"] += 1
                        else:
                            stats["memorization_on_other"] += 1
                    elif phase == "play":
                        stats["play_samples"] += 1
                        if element_type == "card":
                            stats["play_on_cards"] += 1
                        elif element_type == "gridFrame":
                            stats["play_on_gridFrame"] += 1
                        elif element_type == "status label" or element_type == "timer label":
                            stats["play_on_labels"] += 1
                        else:
                            stats["play_on_other"] += 1

            # Calculate percentages (memorization)
            m = stats["memorization_samples"]
            if m > 0:
                stats["memorization_card_percentage"] = stats["memorization_on_cards"] / m * 100
                stats["memorization_gridFrame_percentage"] = stats["memorization_on_gridFrame"] / m * 100
                stats["memorization_labels_percentage"] = stats["memorization_on_labels"] / m * 100
                stats["memorization_other_percentage"] = stats["memorization_on_other"] / m * 100

            # Calculate percentages (play)
            p = stats["play_samples"]
            if p > 0:
                stats["play_card_percentage"] = stats["play_on_cards"] / p * 100
                stats["play_gridFrame_percentage"] = stats["play_on_gridFrame"] / p * 100
                stats["play_labels_percentage"] = stats["play_on_labels"] / p * 100
                stats["play_other_percentage"] = stats["play_on_other"] / p * 100

        except Exception as e:
            print(f"Error computing gaze statistics: {e}")

        return stats

    # ========== PLOTS ===========
    def _load_gaze_log_rows(self, gaze_log_path: str):
        """
        Load rows from gaze log CSV and normalize types.
        Expected columns (best-effort):
          - event_type: 'gaze_sample' or 'click' (or similar)
          - phase: 'memorization' / 'play'
          - game_time_ms: int
          - element_type: 'card', 'gridFrame', 'status_label', etc.
          - card_id: int or empty
          - matched: 0/1 (for click rows, if present)
        """
        if not gaze_log_path or not os.path.exists(gaze_log_path):
            return []

        rows = []
        with open(gaze_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # normalize ints (safe)
                for k in ("game_time_ms", "timestamp_ms", "card_id", "matched", "card_row", "card_col"):
                    if k in r and r[k] not in (None, "", "None"):
                        try:
                            r[k] = int(float(r[k]))
                        except Exception:
                            pass
                rows.append(r)
        return rows

    def _plot_gaze_per_card(self, gaze_log_path: str) -> FigureCanvas:
        rows = self._load_gaze_log_rows(gaze_log_path)

        # count gaze samples on each card_id per phase
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
            ax.set_title("Gaze time per card (%)")
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        x = np.arange(len(card_ids))
        w = 0.40
        ax.bar(x - w / 2, mem_pct, width=w, label="Memorization")
        ax.bar(x + w / 2, play_pct, width=w, label="Play")

        ax.set_title("Gaze time per card (%)")
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
                # click rows should have matched=0/1 and card_id
                if r.get("phase") == "play" and r.get("matched") == 1 and r.get("card_id") not in (None, "", "None"):
                    match_clicks.append(r)

        # If your logger uses a different event_type string, try to detect it
        if not match_clicks:
            for r in rows:
                if r.get("matched") == 1 and r.get("card_id") not in (None, "", "None"):
                    # treat as "match click" best-effort
                    if r.get("game_time_ms") is not None:
                        match_clicks.append(r)

        # build "gaze time before match" per card_id
        per_card_seconds = {}
        sample_period_s = 0.05  # your gaze timer is 50ms; used as approximation

        # index gaze samples by time for faster filtering
        gaze_samples.sort(key=lambda r: r.get("game_time_ms", 0))

        for c in match_clicks:
            t = c.get("game_time_ms")
            cid = c.get("card_id")
            if t is None or cid is None:
                continue

            start = t - window_ms
            count = 0
            # simple scan (good enough for typical file sizes)
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
            ax.text(
                0.5, 0.5,
                "No matched clicks found\n(or missing game_time_ms/card_id)",
                ha="center", va="center"
            )
            ax.set_title("Gaze time on card before it was matched")
            ax.set_xticks([])
            ax.set_yticks([])
            return FigureCanvas(fig)

        ax.bar([str(c) for c in card_ids], values)
        ax.set_title("Gaze time on card before it was matched")
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

        # fallback if event_type is not exactly "click"
        if not clicks:
            clicks = [r for r in rows
                      if r.get("matched") in (0, 1)
                      and r.get("game_time_ms") is not None]

        gaze_samples.sort(key=lambda r: r.get("game_time_ms", 0))

        sample_period_s = 0.05  # approx

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

        ax.set_title("Correct vs incorrect (gaze before click)")

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
        ax.set_title("Gaze on cards over time (Play)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("% gaze samples on cards")

        return FigureCanvas(fig)



# ========================= #
#           Run App         #
# ========================= #
if __name__ == "__main__":
    """
    Entry point for running the app directly.
    Use --dev to enable developer mode (extra calibration diagnostics).
    """

    import argparse

    parser = argparse.ArgumentParser(description="Memory Game with Eye Tracking")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable developer mode: shows camera view during calibration and sample counts"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-family: 'Segoe UI'; }")

    win = MemoryGameWindow(dev_mode=args.dev)
    win.showMaximized()

    sys.exit(app.exec_())
