"""
Heatmap visualization for gaze data analysis.
Shows where the user was looking during memorization and play phases.
"""

import csv
import os
import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QSize, QPoint, QRect, QTimer
from PyQt5.QtGui import QPainter, QColor, QIcon, QLinearGradient, QBrush, QPen


class HeatmapOverlay(QWidget):
    """Transparent overlay that draws the heatmap on top of the game board."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.gaze_points = []  # List of (x, y) tuples
        self.board_rect = None  # QRect of the board area
        self.heatmap_opacity = 0.6
        self.grid_size = 20  # Size of heatmap cells
        self.heatmap_data = {}  # (grid_x, grid_y) -> count

    def set_gaze_data(self, gaze_points, board_rect):
        """Set gaze data points and board rectangle."""
        self.gaze_points = gaze_points
        self.board_rect = board_rect
        self._compute_heatmap()
        self.update()

    def _compute_heatmap(self):
        """Compute heatmap grid from gaze points."""
        self.heatmap_data = {}
        if not self.gaze_points or not self.board_rect:
            return

        for x, y in self.gaze_points:
            # Convert to grid coordinates
            grid_x = int((x - self.board_rect.x()) / self.grid_size)
            grid_y = int((y - self.board_rect.y()) / self.grid_size)

            key = (grid_x, grid_y)
            self.heatmap_data[key] = self.heatmap_data.get(key, 0) + 1

    def paintEvent(self, event):
        """Draw the heatmap overlay."""
        if not self.heatmap_data or not self.board_rect:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Find max count for normalization
        max_count = max(self.heatmap_data.values()) if self.heatmap_data else 1

        # Draw heatmap cells
        for (grid_x, grid_y), count in self.heatmap_data.items():
            # Calculate cell position relative to widget
            cell_x = self.board_rect.x() + grid_x * self.grid_size
            cell_y = self.board_rect.y() + grid_y * self.grid_size

            # Normalize intensity (0 to 1)
            intensity = count / max_count

            # Color gradient: blue (cold) -> green -> yellow -> red (hot)
            color = self._intensity_to_color(intensity)
            color.setAlpha(int(self.heatmap_opacity * 255 * intensity))

            painter.fillRect(
                cell_x, cell_y,
                self.grid_size, self.grid_size,
                color
            )

        # Draw legend
        self._draw_legend(painter)

    def _intensity_to_color(self, intensity):
        """Convert intensity (0-1) to color (blue -> green -> yellow -> red)."""
        if intensity < 0.25:
            # Blue to Cyan
            r = 0
            g = int(255 * (intensity / 0.25))
            b = 255
        elif intensity < 0.5:
            # Cyan to Green
            r = 0
            g = 255
            b = int(255 * (1 - (intensity - 0.25) / 0.25))
        elif intensity < 0.75:
            # Green to Yellow
            r = int(255 * ((intensity - 0.5) / 0.25))
            g = 255
            b = 0
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (1 - (intensity - 0.75) / 0.25))
            b = 0

        return QColor(r, g, b)

    def _draw_legend(self, painter):
        """Draw color legend in top-right corner."""
        legend_width = 120
        legend_height = 25
        margin = 10

        x = self.width() - legend_width - margin
        y = margin

        # Background
        painter.fillRect(x - 5, y - 5, legend_width + 10, legend_height + 30,
                        QColor(255, 255, 255, 200))

        # Gradient bar
        gradient = QLinearGradient(x, y, x + legend_width, y)
        gradient.setColorAt(0.0, QColor(0, 0, 255))
        gradient.setColorAt(0.25, QColor(0, 255, 255))
        gradient.setColorAt(0.5, QColor(0, 255, 0))
        gradient.setColorAt(0.75, QColor(255, 255, 0))
        gradient.setColorAt(1.0, QColor(255, 0, 0))

        painter.fillRect(x, y, legend_width, legend_height, QBrush(gradient))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawRect(x, y, legend_width, legend_height)

        # Labels
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(x, y + legend_height + 15, "Low")
        painter.drawText(x + legend_width - 25, y + legend_height + 15, "High")


class HeatmapGameBoard(QWidget):
    """
    Game board replica for heatmap visualization.
    Shows cards that can be individually flipped, with heatmap overlay.
    """

    GRID_SPACING = 8

    def __init__(self, num_cards, board_size, front_images, parent=None):
        super().__init__(parent)
        self.num_cards = num_cards
        self.board_size = board_size  # (width, height) of original board
        self.front_images = front_images
        self.rows, self.cols = ((3, num_cards // 3) if num_cards % 3 == 0 else (2, num_cards // 2))

        self.cards = []
        self.card_flipped = {}  # card_index -> bool

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Grid frame for cards
        self.grid_frame = QFrame()
        self.grid_frame.setStyleSheet("""
            QFrame {
                border: 4px solid #B68DDE;
                border-radius: 16px;
                background-color: #fdfcff;
                padding: 5px;
            }
        """)

        self.grid = QGridLayout(self.grid_frame)
        self.grid.setSpacing(self.GRID_SPACING)

        for r in range(self.rows):
            self.grid.setRowStretch(r, 1)
        for c in range(self.cols):
            self.grid.setColumnStretch(c, 1)

        # Create cards
        self.back_image = "images/backOfCard.png"
        for i, img in enumerate(self.front_images):
            btn = QPushButton()
            btn.setStyleSheet("""
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
            """)
            btn.image_path = img
            btn.card_index = i
            btn.setIcon(QIcon(self.back_image))  # Start face down
            btn.clicked.connect(lambda checked, idx=i: self._toggle_card(idx))
            self.grid.addWidget(btn, i // self.cols, i % self.cols)
            self.cards.append(btn)
            self.card_flipped[i] = False

        layout.addWidget(self.grid_frame)

        # Heatmap overlay
        self.heatmap_overlay = HeatmapOverlay(self)

    def _toggle_card(self, card_index):
        """Toggle a card between face up and face down."""
        btn = self.cards[card_index]
        self.card_flipped[card_index] = not self.card_flipped[card_index]

        if self.card_flipped[card_index]:
            btn.setIcon(QIcon(btn.image_path))
        else:
            btn.setIcon(QIcon(self.back_image))

    def flip_all_up(self):
        """Flip all cards face up."""
        for i, btn in enumerate(self.cards):
            self.card_flipped[i] = True
            btn.setIcon(QIcon(btn.image_path))

    def flip_all_down(self):
        """Flip all cards face down."""
        for i, btn in enumerate(self.cards):
            self.card_flipped[i] = False
            btn.setIcon(QIcon(self.back_image))

    def set_heatmap_data(self, gaze_points):
        """Set gaze data for heatmap overlay."""
        # Get board rect in widget coordinates
        board_rect = QRect(
            self.grid_frame.x(),
            self.grid_frame.y(),
            self.grid_frame.width(),
            self.grid_frame.height()
        )
        self.heatmap_overlay.set_gaze_data(gaze_points, board_rect)

    def resizeEvent(self, event):
        """Handle resize - update card sizes and heatmap overlay."""
        super().resizeEvent(event)

        # Update card sizes
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

        # Update heatmap overlay size and position
        self.heatmap_overlay.setGeometry(self.rect())


class HeatmapWindow(QWidget):
    """
    Window for viewing gaze heatmap over the game board.
    """

    def __init__(self, gaze_data_path=None, game_config=None, parent=None, on_back=None):
        super().__init__(parent)
        self.setWindowTitle("Gaze Heatmap Visualization")
        self.gaze_data_path = gaze_data_path
        self.game_config = game_config or {}
        self.on_back = on_back

        # Data
        self.gaze_data = {"memorization": [], "play": []}
        self.current_phase = "play"

        # Default game config
        self.num_cards = self.game_config.get("num_cards", 8)
        self.front_images = self.game_config.get("front_images",
            [f"images/{i}.png" for i in range(1, self.num_cards // 2 + 1)] * 2)

        self._build_ui()

        if gaze_data_path:
            self._load_gaze_data(gaze_data_path)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Top controls
        controls = QHBoxLayout()

        # Phase selector
        phase_label = QLabel("Phase:")
        phase_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        controls.addWidget(phase_label)

        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["Play Phase", "Memorization Phase", "Both Phases"])
        self.phase_combo.setFixedWidth(180)
        self.phase_combo.setStyleSheet("font-size: 14px; padding: 5px;")
        self.phase_combo.currentIndexChanged.connect(self._on_phase_changed)
        controls.addWidget(self.phase_combo)

        controls.addSpacing(20)

        # Card controls
        flip_up_btn = QPushButton("Flip All Up")
        flip_up_btn.setStyleSheet("""
            QPushButton {
                background-color: #8549c9;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #7239b5; }
        """)
        flip_up_btn.clicked.connect(self._flip_all_up)
        controls.addWidget(flip_up_btn)

        flip_down_btn = QPushButton("Flip All Down")
        flip_down_btn.setStyleSheet("""
            QPushButton {
                background-color: #8549c9;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #7239b5; }
        """)
        flip_down_btn.clicked.connect(self._flip_all_down)
        controls.addWidget(flip_down_btn)

        controls.addSpacing(20)
        controls.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #c94949;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #b53939; }
        """)
        close_btn.setText("Back")
        close_btn.clicked.connect(lambda: self.on_back() if self.on_back else self.close())

        controls.addWidget(close_btn)

        layout.addLayout(controls)

        # Stats label
        self.stats_label = QLabel("No gaze data loaded")
        self.stats_label.setStyleSheet("font-size: 14px; color: #333;")
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)

        # Game board with heatmap
        board_size = self.game_config.get("board_size", (800, 600))
        self.game_board = HeatmapGameBoard(
            self.num_cards,
            board_size,
            self.front_images
        )
        layout.addWidget(self.game_board, 1)

    def _load_gaze_data(self, filepath):
        """Load gaze data from CSV file."""
        self.gaze_data = {"memorization": [], "play": []}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("event_type") != "gaze_sample":
                        continue

                    gaze_x = row.get("gaze_x", "")
                    gaze_y = row.get("gaze_y", "")
                    phase = row.get("phase", "")

                    if gaze_x and gaze_y and phase:
                        try:
                            x = float(gaze_x)
                            y = float(gaze_y)
                            if phase in self.gaze_data:
                                self.gaze_data[phase].append((x, y))
                        except ValueError:
                            continue

            # Update stats
            mem_count = len(self.gaze_data["memorization"])
            play_count = len(self.gaze_data["play"])
            self.stats_label.setText(
                f"Loaded: {mem_count} memorization samples, {play_count} play samples"
            )

            # Update heatmap
            QTimer.singleShot(0, self._update_heatmap)

        except Exception as e:
            self.stats_label.setText(f"Error loading data: {str(e)}")

    def _load_data_dialog(self):
        """Open file dialog to load gaze data."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Gaze Data File",
            ".",
            "CSV Files (*.csv);;All Files (*)"
        )
        if filepath:
            self._load_gaze_data(filepath)

    def _on_phase_changed(self, index):
        """Handle phase selection change."""
        phases = ["play", "memorization", "both"]
        self.current_phase = phases[index]
        QTimer.singleShot(0, self._update_heatmap)
        #self._update_heatmap()

    def _update_heatmap(self):
        """Update heatmap based on current phase selection."""
        if self.current_phase == "both":
            points = self.gaze_data["memorization"] + self.gaze_data["play"]
        else:
            points = self.gaze_data.get(self.current_phase, [])

        # Convert screen coordinates to widget coordinates
        widget_points = []
        board_rect = self.game_board.grid_frame.geometry()

        for x, y in points:
            board_top_left = self.game_board.grid_frame.mapToGlobal(QPoint(0, 0))
            widget_x = x - board_top_left.x()
            widget_y = y - board_top_left.y()
            widget_points.append((widget_x, widget_y))

        self.game_board.set_heatmap_data(widget_points)

    def _flip_all_up(self):
        """Flip all cards face up."""
        self.game_board.flip_all_up()

    def _flip_all_down(self):
        """Flip all cards face down."""
        self.game_board.flip_all_down()

    def set_game_config(self, num_cards, front_images, board_size=None):
        """Update game configuration."""
        self.num_cards = num_cards
        self.front_images = front_images
        if board_size:
            self.game_config["board_size"] = board_size

        # Rebuild board
        layout = self.layout()
        layout.removeWidget(self.game_board)
        self.game_board.deleteLater()

        self.game_board = HeatmapGameBoard(
            self.num_cards,
            self.game_config.get("board_size", (800, 600)),
            self.front_images
        )
        layout.insertWidget(2, self.game_board, 1)

        self._update_heatmap()


