"""
Simple integration test for the Memory Game with Eye Tracking.
This verifies that all components can be imported and initialized.
"""

import sys
import os

# Add eye tracking module to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eye-detection-final"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("  ✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ NumPy imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import NumPy: {e}")
        return False
    
    try:
        from sklearn.linear_model import Ridge
        print("  ✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import Scikit-learn: {e}")
        return False
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("  ✓ PyQt5 imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import PyQt5: {e}")
        return False
    
    try:
        from eye_detector import EyeDetector
        print("  ✓ EyeDetector imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import EyeDetector: {e}")
        return False
    
    return True


def test_eye_detector_init():
    """Test that EyeDetector can be initialized."""
    print("\nTesting EyeDetector initialization...")
    
    try:
        from eye_detector import EyeDetector
        detector = EyeDetector()
        print("  ✓ EyeDetector initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize EyeDetector: {e}")
        return False


def test_camera_availability():
    """Test if camera is available."""
    print("\nTesting camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"  ✓ Camera is available (frame size: {frame.shape})")
                return True
            else:
                print("  ✗ Camera opened but cannot read frames")
                return False
        else:
            print("  ✗ Cannot open camera")
            return False
    except Exception as e:
        print(f"  ✗ Error testing camera: {e}")
        return False


def test_game_components():
    """Test that game components can be imported."""
    print("\nTesting game components...")
    
    try:
        from MemoryGame_v2 import (
            GazeFeatureExtractor,
            EyeFrameValidator,
            CameraThread,
            CalibrationScreen,
            MemoryGameBoard,
            MemoryGameWindow
        )
        print("  ✓ All game components imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import game components: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Memory Game Eye Tracking Integration Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("EyeDetector", test_eye_detector_init()))
    results.append(("Camera", test_camera_availability()))
    results.append(("Game Components", test_game_components()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! The game is ready to run.")
        print("\nTo start the game, run: python MemoryGame_v2.py")
    else:
        print("\n✗ Some tests failed. Please install missing dependencies:")
        print("\n  pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

