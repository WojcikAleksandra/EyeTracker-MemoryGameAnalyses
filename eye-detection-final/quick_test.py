"""
Quick test script - verify the detector works.
Run this first to check everything is set up correctly.
"""

import cv2
import json
from eye_detector import EyeDetector


def quick_test():
    """Quick test of the eye detector."""
    print("=" * 60)
    print("EYE DETECTOR - QUICK TEST")
    print("=" * 60)
    print("\nTesting detector initialization...")
    
    try:
        detector = EyeDetector()
        print("[OK] Detector initialized successfully")
    except Exception as e:
        print(f"[FAIL] Failed to initialize detector: {e}")
        return
    
    print("\nTesting camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[FAIL] Failed to open camera")
        print("  - Check if camera is connected")
        print("  - Close other apps using camera")
        return
    
    print("[OK] Camera opened successfully")
    
    print("\nCapturing and processing 10 frames...")
    print("(Position yourself 60-70 cm from camera)\n")
    
    success_count = 0
    
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"[X] Failed to read frame {i}")
            continue
        
        result = detector.detect(frame, frame_id=i)
        
        status_symbol = {
            'ok': '[OK]',
            'no_face': '[  ]',
            'no_eyes': '[ -]',
            'partial': '[+-]'
        }
        
        print(f"Frame {i}: {status_symbol.get(result['status'], '[?]')} {result['status']}")
        
        if result['status'] == 'ok':
            success_count += 1
            print(f"  Left iris: {result['left_eye']['iris_center']}")
            print(f"  Right iris: {result['right_eye']['iris_center']}")
        
        cv2.imshow('Quick Test', frame)
        cv2.waitKey(100)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Success rate: {success_count}/10 ({success_count * 10}%)")
    
    if success_count >= 7:
        print("\n[OK] PASS - Detector is working well!")
        print("\nNext steps:")
        print("  1. Read README_FOR_NEXT_PROGRAMMER.md")
        print("  2. Run example_usage.py")
        print("  3. Integrate into your gaze estimation pipeline")
    elif success_count >= 4:
        print("\n[+-] PARTIAL - Detector works but not consistently")
        print("\nTips to improve:")
        print("  - Better lighting (face well-lit)")
        print("  - Position at 60-70 cm from camera")
        print("  - Face camera directly")
        print("  - Remove glasses if causing issues")
    else:
        print("\n[FAIL] - Detector not working reliably")
        print("\nTroubleshooting:")
        print("  1. Check lighting - face should be well-lit")
        print("  2. Distance - should be 60-70 cm")
        print("  3. Camera angle - look directly at camera")
        print("  4. Check if face is detected (status='no_face')")
        print("  5. See README_FOR_NEXT_PROGRAMMER.md troubleshooting section")
    
    # Save one sample result
    print("\nSaving sample result to 'sample_output.json'...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        result = detector.detect(frame)
        with open('sample_output.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("✓ Sample saved")
    cap.release()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    quick_test()

