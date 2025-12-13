import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any


class GazeDataLogger:
    """
    Centralized logging system for gaze tracking data.
    Generates unique session IDs and logs comprehensive data to CSV.
    """

    def __init__(self, output_dir: str = "."):
        """
        Initialize the logger.
        output_dir: Directory where log files will be saved.
        """
        self.output_dir = output_dir
        self.session_id = self._generate_session_id()
        self.log_file_path = os.path.join(
            output_dir, f"gaze_data_{self.session_id}.csv"
        )
        self.log_file = None
        self.fieldnames = [
            "session_id",
            "timestamp_ms",
            "phase",
            "event_type",
            "gaze_x",
            "gaze_y",
            "click_x",
            "click_y",
            "element_type",
            "card_row",
            "card_col",
            "card_id",
            "card_image_name",
            "matched",
            "game_time_ms",
        ]

    def _generate_session_id(self) -> str:
        """Generate unique session ID: YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def start_logging(self):
        """Open log file and write header."""
        self.log_file = open(self.log_file_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(self.log_file, fieldnames=self.fieldnames)
        writer.writeheader()

    def stop_logging(self):
        """Close log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def log_gaze_sample(
        self,
        timestamp_ms: int,
        phase: str,
        gaze_x: Optional[int],
        gaze_y: Optional[int],
        element_type: Optional[str] = None,
        card_row: Optional[int] = None,
        card_col: Optional[int] = None,
        card_id: Optional[int] = None,
        card_image_name: Optional[str] = None,
        game_time_ms: int = 0,
    ):
        """Log a gaze tracking sample."""
        if not self.log_file:
            return

        row = {
            "session_id": self.session_id,
            "timestamp_ms": timestamp_ms,
            "phase": phase,
            "event_type": "gaze_sample",
            "gaze_x": gaze_x if gaze_x is not None else "",
            "gaze_y": gaze_y if gaze_y is not None else "",
            "click_x": "",
            "click_y": "",
            "element_type": element_type if element_type else "",
            "card_row": card_row if card_row is not None else "",
            "card_col": card_col if card_col is not None else "",
            "card_id": card_id if card_id is not None else "",
            "card_image_name": card_image_name if card_image_name else "",
            "matched": "",
            "game_time_ms": game_time_ms,
        }

        writer = csv.DictWriter(self.log_file, fieldnames=self.fieldnames)
        writer.writerow(row)
        self.log_file.flush()

    def log_click(
        self,
        timestamp_ms: int,
        phase: str,
        click_x: int,
        click_y: int,
        gaze_x: Optional[int] = None,
        gaze_y: Optional[int] = None,
        element_type: Optional[str] = None,
        card_row: Optional[int] = None,
        card_col: Optional[int] = None,
        card_id: Optional[int] = None,
        card_image_name: Optional[str] = None,
        matched: Optional[int] = None,
        game_time_ms: int = 0,
    ):
        """Log a click event with full context."""
        if not self.log_file:
            return

        row = {
            "session_id": self.session_id,
            "timestamp_ms": timestamp_ms,
            "phase": phase,
            "event_type": "click",
            "gaze_x": gaze_x if gaze_x is not None else "",
            "gaze_y": gaze_y if gaze_y is not None else "",
            "click_x": click_x,
            "click_y": click_y,
            "element_type": element_type if element_type else "",
            "card_row": card_row if card_row is not None else "",
            "card_col": card_col if card_col is not None else "",
            "card_id": card_id if card_id is not None else "",
            "card_image_name": card_image_name if card_image_name else "",
            "matched": matched if matched is not None else "",
            "game_time_ms": game_time_ms,
        }

        writer = csv.DictWriter(self.log_file, fieldnames=self.fieldnames)
        writer.writerow(row)
        self.log_file.flush()

    def log_phase_event(
        self,
        timestamp_ms: int,
        phase: str,
        event_type: str,
        game_time_ms: int = 0,
    ):
        """Log phase start/end events."""
        if not self.log_file:
            return

        row = {
            "session_id": self.session_id,
            "timestamp_ms": timestamp_ms,
            "phase": phase,
            "event_type": event_type,
            "gaze_x": "",
            "gaze_y": "",
            "click_x": "",
            "click_y": "",
            "element_type": "",
            "card_row": "",
            "card_col": "",
            "card_id": "",
            "card_image_name": "",
            "matched": "",
            "game_time_ms": game_time_ms,
        }

        writer = csv.DictWriter(self.log_file, fieldnames=self.fieldnames)
        writer.writerow(row)
        self.log_file.flush()

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id

    def get_log_file_path(self) -> str:
        """Get the path to the log file."""
        return self.log_file_path

