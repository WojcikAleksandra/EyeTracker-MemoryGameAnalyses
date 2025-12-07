"""
Quick test to verify calibration points are visible.
This creates a minimal calibration screen to test the visual display.
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen

class TestCalibrationScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Point Test")
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")
        
        # Generate test points
        self.points = self._generate_test_points()
        self.current_idx = 0
        
        layout = QVBoxLayout(self)
        self.label = QLabel(f"Point {self.current_idx + 1}/{len(self.points)}\nClick on the red point", 
                           alignment=Qt.AlignTop | Qt.AlignHCenter)
        self.label.setStyleSheet("font-size: 20px; padding: 20px;")
        layout.addWidget(self.label)
        layout.addStretch()
    
    def _generate_test_points(self):
        """Generate 20 calibration points in a 5x4 grid."""
        points = []
        width = self.width()
        height = self.height()
        
        # Increased margins to keep points away from edges
        margin_x = 0.08 * width  # 8% margin on sides
        margin_y = 0.10 * height  # 10% margin on top/bottom
        usable_w = width - 2 * margin_x
        usable_h = height - 2 * margin_y
        
        cols = 5
        rows = 4
        
        for r in range(rows):
            for c in range(cols):
                x = int(margin_x + c * usable_w / (cols - 1))
                y = int(margin_y + r * usable_h / (rows - 1))
                points.append((x, y))
        return points
    
    def mousePressEvent(self, event):
        click_x = event.pos().x()
        click_y = event.pos().y()
        target_x, target_y = self.points[self.current_idx]
        
        # Calculate distance
        import math
        dist = math.sqrt((click_x - target_x)**2 + (click_y - target_y)**2)
        
        print(f"Clicked at ({click_x}, {click_y}), target at ({target_x}, {target_y}), distance: {dist:.1f}px")
        
        if dist <= 50:
            self.current_idx += 1
            if self.current_idx >= len(self.points):
                print("\nAll points clicked! Calibration complete.")
                self.label.setText("All points clicked! ✓")
            else:
                self.label.setText(f"Point {self.current_idx + 1}/{len(self.points)}\nClick on the red point")
            self.update()
        else:
            print("  -> Too far from target!")
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.current_idx >= len(self.points):
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw current calibration point - simple red dot
        target_x, target_y = self.points[self.current_idx]
        center = QPoint(target_x, target_y)
        
        # Simple solid red circle
        painter.setBrush(Qt.red)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, 10, 10)
        
        # Draw all remaining points as small gray circles (for reference)
        painter.setBrush(Qt.transparent)
        painter.setPen(QPen(Qt.lightGray, 1))
        for i, (px, py) in enumerate(self.points):
            if i != self.current_idx:
                painter.drawEllipse(QPoint(px, py), 5, 5)

def main():
    app = QApplication(sys.argv)
    window = TestCalibrationScreen()
    window.show()
    
    print("="*60)
    print("Calibration Point Visual Test")
    print("="*60)
    print("You should see a red point with white center.")
    print("Click on each point to advance to the next one.")
    print("Gray circles show remaining points.")
    print("="*60)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

