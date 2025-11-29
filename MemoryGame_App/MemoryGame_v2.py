import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QPushButton, QLabel, QFrame,
    QComboBox, QAction
)
from PyQt5.QtCore import Qt, QSize, QTime, QTimer, QPoint, QRect
from PyQt5.QtGui import QIcon, QPainter, QPen


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

    def __init__(self, num_cards=8):
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
    def start_memorize_phase(self):
        self.moves = 0
        self.elapsed = 0
        self._update_hud()

        QTimer.singleShot(0, self.update_hitboxes)

        self._preview_deadline_ms = self.PREVIEW_MS
        self._preview_start = QTime.currentTime()
        self.locked = True
        self.preview_timer.start(100)

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
        QTimer.singleShot(1500, lambda: getattr(self.window(), "show_win_page", lambda: None)())

        if self.log_file:
            self.log_file.close()
            self.log_file = None


    def stop_all_timers(self):
        self.preview_timer.stop()
        self.game_timer.stop()
        self.locked = True
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
        flip 1/2, matched_flag 0/1, card_id (np. 3 z images/3.png)"""
        if self.log_file is None:
            return

        now = QTime.currentTime()
        ms = self.game_start_time.msecsTo(now)

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

        # dopisujemy card_id na końcu
        self.log_file.write(
            f"{ms},{x},{y},{self.click_counter},{matched_flag},{card_id}\n"
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
        self._add_menu_action(settings, "Set file directory", lambda: None)
        self._add_menu_action(settings, "Restart data", lambda: None)
        self._add_menu_action(settings, "Recalibrate eye-tracking", lambda: None)

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
        play_btn.clicked.connect(lambda: self.show_countdown(int(card_box.currentText())))

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
        self.board_page = MemoryGameBoard(num_cards)
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
