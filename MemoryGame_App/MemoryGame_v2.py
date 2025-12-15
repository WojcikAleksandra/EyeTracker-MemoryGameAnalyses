import sys
import random
import os
import csv
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QPushButton, QLabel, QFrame,
    QComboBox, QAction, QMessageBox, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QSize, QTime, QTimer, QPoint, QRect
from PyQt5.QtGui import QIcon, QPainter, QPen

# Add paths for gaze localization
sys.path.append("..")
sys.path.append("../GazeLocalization")
try:
    from calibration_screen import CalibrationScreen
    from gaze_data_logger import GazeDataLogger
    from heatmap_view import HeatmapWindow
except ImportError as e:
    print(f"ERROR: Required gaze tracking modules not available: {e}")
    print("\nPlease install required dependencies:")
    print("  pip install opencv-python scikit-learn numpy")
    print("\nApplication cannot start without gaze tracking support.")
    sys.exit(1)


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


# ========================= #
#     Memory Game Board     #
# ========================= #
class MemoryGameBoard(QWidget):
    GRID_SPACING = 8
    PREVIEW_MS = 5000
    FLIP_CHECK_DELAY_MS = 800

    def __init__(self, num_cards=8, gaze_engine=None, gaze_logger=None):
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
        self.debug_hitboxes = False  #false -> mozna usunac paintEvent, i w update_hitboxes usunac self_update

        # Gaze tracking
        self.gaze_engine = gaze_engine
        self.gaze_logger = gaze_logger
        self.gaze_timer = QTimer(self)
        self.gaze_timer.timeout.connect(self._sample_gaze)
        self.tracking_active = False
        self.current_phase = None  # "memorization" or "play"

        # timers
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self._update_timer)

        # --- logging click data ---
        self.log_file_path = "click_log.csv"
        self.log_file = None
        self.click_counter = 2  # 1/2/1/2...
        self.game_start_time = None  # QTime after preview

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
            btn.card_index = i
            btn.card_row = i // self.cols
            btn.card_col = i % self.cols
            btn.clicked.connect(self.on_card_click)
            btn.setIcon(QIcon(img))  # show face-up during preview
            self.grid.addWidget(btn, btn.card_row, btn.card_col)
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
    def start_memorize_phase(self):
        self.moves = 0
        self.elapsed = 0
        self._update_hud()

        QTimer.singleShot(0, self.update_hitboxes)

        self._preview_deadline_ms = self.PREVIEW_MS
        self._preview_start = QTime.currentTime()
        self.locked = True
        self.preview_timer.start(100)

        # Start gaze tracking
        self.current_phase = "memorization"
        self._start_gaze_tracking()
        
        # Log phase start
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
            
            # Log memorization phase end and play phase start
            if self.gaze_logger:
                timestamp_ms = int(QTime.currentTime().msecsSinceStartOfDay())
                self.gaze_logger.log_phase_event(timestamp_ms, "memorization", "phase_end", 0)
                self.current_phase = "play"
                self.gaze_logger.log_phase_event(timestamp_ms, "play", "phase_start", 0)

    def _start_game_timer(self):
        self.elapsed = 0
        self._update_hud()
        self.game_timer.start(1000)

        # start logging
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
                main_window.last_game_info["gaze_log_path"] = self.gaze_logger.get_log_file_path()
            
            self.gaze_logger.stop_logging()
        
        # Save game results for leaderboard
        main_window = self.window()
        if main_window and hasattr(main_window, 'save_game_result'):
            main_window.save_game_result(self.elapsed, self.moves, self.num_cards)
        
        # Note: Keep calibration for multiple games - don't close gaze engine here
        # Camera stays ready for next game
        
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
        if self.gaze_logger:
            self.gaze_logger.stop_logging()
        # Note: Don't close gaze engine here - keep calibration for multiple games


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
                        filename = image_path.rsplit("/", 1)[-1]  # "3.png"
                        result["card_image_name"] = filename
                        name = filename.split(".", 1)[0]  # "3"
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
            filename = image_path.rsplit("/", 1)[-1]  # "3.png"
            result["card_image_name"] = filename
            name = filename.split(".", 1)[0]  # "3"
            if name.isdigit():
                result["card_id"] = int(name)

        return result

    def log_click(self, btn, matched_flag=0):
        """Loguje: czas od startu gry, współrzędne klikniętej karty,
        flip 1/2, matched_flag 0/1, card_id (np. 3 z images/3.png)"""
        now = QTime.currentTime()
        ms = self.game_start_time.msecsTo(now) if self.game_start_time else 0

        # globalne współrzędne kliknięcia: bierzemy środek karty
        rect = self.card_rects_screen.get(btn)
        if rect:
            x = rect.center().x()
            y = rect.center().y()
        else:
            x = y = -1

        # wyciągamy numer karty z image_path, np. "images/3.png" -> "3"
        card_id = -1
        image_path = getattr(btn, "image_path", "")
        if image_path:
            filename = image_path.rsplit("/", 1)[-1]  # "3.png"
            name = filename.split(".", 1)[0]  # "3"
            if name.isdigit():
                card_id = int(name)

        # flip 1/2/1/2...
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
    # Session ID for this app run
    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    GAME_HISTORY_FILE = "game_history.json"
    
    def __init__(self, dev_mode=False):
        super().__init__()
        self.setWindowTitle("Memory Game")
        self.dev_mode = dev_mode

        self._countdown_timer = None
        self._countdown_page = None
        self._countdown_cancelled = False
        self._auto_nav_enabled = True
        self.board_page = None

        # flagi do blokady rozmiaru po rozpoczeciu gry
        self._resize_locked = False
        self._old_min_size = QSize(0, 0)
        self._old_max_size = QSize(16777215, 16777215)

        # Gaze tracking
        self.gaze_engine = None
        self.gaze_logger = None
        self.calibration_done = False
        self.pending_num_cards = None
        
        # Store last game info for heatmap
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

        if self.dev_mode:
            self.setWindowTitle("Memory Game [DEV MODE]")

        self._build_ui()

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
        self._add_menu_action(settings, "Recalibrate Eye-Tracking", self._recalibrate)
        self._add_menu_action(settings, "Clear Game History", self._clear_game_history)

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
        
        # Initialize gaze logger if gaze engine exists
        gaze_logger = None
        if self.gaze_engine:
            try:
                gaze_logger = GazeDataLogger(app_session_id=self.SESSION_ID)
                gaze_logger.start_logging()
                # Store gaze log path for heatmap
                self.last_game_info["gaze_log_path"] = gaze_logger.get_log_file_path()
            except Exception as e:
                print(f"Warning: Could not initialize gaze logger: {e}")
        
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

        # Main content in horizontal layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)

        # Left side: Last game stats and gaze statistics
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        # Last game info
        last_game_box = QFrame()
        last_game_box.setStyleSheet(Styles.LIGHT_FRAME)
        last_game_layout = QVBoxLayout(last_game_box)
        
        last_game_title = QLabel("Last Game")
        last_game_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4B2C82;")
        last_game_layout.addWidget(last_game_title)
        
        # Get last game info
        if self.game_history:
            last = self.game_history[-1]
            last_info = QLabel(
                f"Cards: {last.get('num_cards', 'N/A')}\n"
                f"Time: {last.get('time_seconds', 'N/A')}s\n"
                f"Moves: {last.get('moves', 'N/A')}"
            )
        else:
            last_info = QLabel("No games played yet")
        last_info.setStyleSheet("font-size: 16px; color: #333;")
        last_game_layout.addWidget(last_info)
        left_panel.addWidget(last_game_box)

        # Gaze statistics
        gaze_stats_box = QFrame()
        gaze_stats_box.setStyleSheet(Styles.LIGHT_FRAME)
        gaze_stats_layout = QVBoxLayout(gaze_stats_box)
        
        gaze_title = QLabel("Gaze Statistics")
        gaze_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4B2C82;")
        gaze_stats_layout.addWidget(gaze_title)
        
        # Compute gaze stats from last game
        gaze_log_path = self.last_game_info.get("gaze_log_path")
        gaze_stats = self._compute_gaze_statistics(gaze_log_path)
        
        if gaze_stats["memorization_samples"] > 0:
            gaze_info = QLabel(
                f"Memorization Phase:\n"
                f"  Time on cards: {gaze_stats['memorization_card_percentage']:.1f}%\n"
                f"  ({gaze_stats['memorization_on_cards']}/{gaze_stats['memorization_samples']} samples)\n\n"
                f"Play Phase:\n"
                f"  Time on cards: {gaze_stats['play_card_percentage']:.1f}%\n"
                f"  ({gaze_stats['play_on_cards']}/{gaze_stats['play_samples']} samples)"
            )
        else:
            gaze_info = QLabel("No gaze data available\n(Play with eye tracking enabled)")
        gaze_info.setStyleSheet("font-size: 14px; color: #333;")
        gaze_stats_layout.addWidget(gaze_info)
        left_panel.addWidget(gaze_stats_box)

        # Heatmap button
        heatmap_btn = QPushButton("View Gaze Heatmap")
        heatmap_btn.setFixedHeight(50)
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
        """)
        heatmap_btn.clicked.connect(self._show_heatmap)
        left_panel.addWidget(heatmap_btn)
        left_panel.addStretch()

        content_layout.addLayout(left_panel, 1)

        # Right side: Leaderboard
        right_panel = QVBoxLayout()
        
        leaderboard_box = QFrame()
        leaderboard_box.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 2px solid #B68DDE;
                padding: 10px;
            }
        """)
        leaderboard_layout = QVBoxLayout(leaderboard_box)
        
        leaderboard_title = QLabel("Leaderboard")
        leaderboard_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4B2C82;")
        leaderboard_title.setAlignment(Qt.AlignCenter)
        leaderboard_layout.addWidget(leaderboard_title)
        
        # Create leaderboard table
        leaderboard_table = QTableWidget()
        leaderboard_table.setColumnCount(4)
        leaderboard_table.setHorizontalHeaderLabels(["Cards", "Time", "Moves", "Date"])
        leaderboard_table.horizontalHeader().setVisible(True)
        leaderboard_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        leaderboard_table.horizontalHeader().setMinimumHeight(35)
        leaderboard_table.verticalHeader().setVisible(False)
        leaderboard_table.setEditTriggers(QTableWidget.NoEditTriggers)
        leaderboard_table.setSelectionBehavior(QTableWidget.SelectRows)
        leaderboard_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                font-size: 14px;
                gridline-color: #ddd;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #8549c9;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-right: 1px solid #7239b5;
            }
        """)
        
        # Sort by time (best first) and take top 10
        sorted_history = sorted(self.game_history, key=lambda x: x.get("time_seconds", 9999))[:10]
        leaderboard_table.setRowCount(len(sorted_history))
        
        for i, game in enumerate(sorted_history):
            leaderboard_table.setItem(i, 0, QTableWidgetItem(str(game.get("num_cards", "?"))))
            leaderboard_table.setItem(i, 1, QTableWidgetItem(f"{game.get('time_seconds', '?')}s"))
            leaderboard_table.setItem(i, 2, QTableWidgetItem(str(game.get("moves", "?"))))
            
            # Format date
            timestamp = game.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    date_str = dt.strftime("%m/%d %H:%M")
                except:
                    date_str = "?"
            else:
                date_str = "?"
            leaderboard_table.setItem(i, 3, QTableWidgetItem(date_str))
        
        leaderboard_layout.addWidget(leaderboard_table)
        
        # Best records summary
        if self.game_history:
            best_time = min(g.get("time_seconds", 9999) for g in self.game_history)
            best_moves = min(g.get("moves", 9999) for g in self.game_history)
            total_games = len(self.game_history)
            
            records_label = QLabel(
                f"Best Time: {best_time}s | Fewest Moves: {best_moves} | Total Games: {total_games}"
            )
            records_label.setStyleSheet("font-size: 14px; color: #666; margin-top: 10px;")
            records_label.setAlignment(Qt.AlignCenter)
            leaderboard_layout.addWidget(records_label)
        
        right_panel.addWidget(leaderboard_box)
        content_layout.addLayout(right_panel, 2)

        layout.addLayout(content_layout, 1)

        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    def _show_heatmap(self):
        """Show heatmap visualization window."""
        # Close existing heatmap window if open
        if self.heatmap_window:
            self.heatmap_window.close()
        
        # Create game config from last game info
        game_config = {
            "num_cards": self.last_game_info.get("num_cards", 8),
            "front_images": self.last_game_info.get("front_images", 
                [f"images/{i}.png" for i in range(1, 5)] * 2),
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
            "Are you sure you want to clear all game history?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.game_history = []
            self._save_game_history()
            QMessageBox.information(self, "Cleared", "Game history has been cleared.")

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
            "memorization_card_percentage": 0,
            "play_samples": 0,
            "play_on_cards": 0,
            "play_card_percentage": 0,
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
                    elif phase == "play":
                        stats["play_samples"] += 1
                        if element_type == "card":
                            stats["play_on_cards"] += 1
            
            # Calculate percentages
            if stats["memorization_samples"] > 0:
                stats["memorization_card_percentage"] = (
                    stats["memorization_on_cards"] / stats["memorization_samples"] * 100
                )
            if stats["play_samples"] > 0:
                stats["play_card_percentage"] = (
                    stats["play_on_cards"] / stats["play_samples"] * 100
                )
        except Exception as e:
            print(f"Error computing gaze statistics: {e}")
        
        return stats


# ========================= #
#           Run App         #
# ========================= #
if __name__ == "__main__":
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
