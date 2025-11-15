"""
Example usage of the Eye Detection Module.
Demonstrates how to use the detector and work with the output data.
"""

import cv2
import json
import numpy as np
from eye_detector import EyeDetector


def example_1_basic_usage():
    """Basic usage: detect eyes in video stream."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    detector = EyeDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("\nProcessing frames... Press 'q' to quit\n")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect eyes and iris
        result = detector.detect(frame, frame_id=frame_count)
        
        # Print status
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Status = {result['status']}")
            if result['status'] == 'ok':
                print(f"  Left eye: {result['left_eye']['bbox']}")
                print(f"  Right eye: {result['right_eye']['bbox']}")
                print(f"  Left iris center: {result['left_eye']['iris_center']}")
                print(f"  Right iris center: {result['right_eye']['iris_center']}")
        
        # Visualize
        display = visualize_detection(frame, result)
        cv2.imshow('Eye Detection', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Example 1 complete")


def example_2_save_to_json():
    """Save detection results to JSON file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Save Results to JSON")
    print("=" * 60)
    
    detector = EyeDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    results = []
    
    print("\nCollecting 100 frames... Press 'q' to stop early\n")
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame, frame_id=i)
        results.append(result)
        
        if i % 10 == 0:
            print(f"Collected {i} frames...")
        
        # Show progress
        cv2.imshow('Collecting...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save to JSON
    with open('eye_detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved {len(results)} frames to eye_detection_results.json")


def example_3_extract_iris_positions():
    """Extract only iris positions for gaze estimation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Extract Iris Positions")
    print("=" * 60)
    
    detector = EyeDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("\nExtracting iris positions... Press 'q' to quit\n")
    
    iris_data = []
    
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        
        if result['status'] == 'ok':
            # Extract just what you need for gaze estimation
            iris_info = {
                'frame_id': result['frame_id'],
                'timestamp': result['timestamp_ms'],
                'left_iris_rel': result['left_eye']['iris_center_rel'],
                'right_iris_rel': result['right_eye']['iris_center_rel'],
                'left_iris_abs': result['left_eye']['iris_center'],
                'right_iris_abs': result['right_eye']['iris_center']
            }
            iris_data.append(iris_info)
            
            print(f"Frame {i}: L={iris_info['left_iris_rel']}, R={iris_info['right_iris_rel']}")
        
        cv2.imshow('Processing...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save compact format
    with open('iris_positions.json', 'w') as f:
        json.dump(iris_data, f, indent=2)
    
    print(f"\n✓ Saved iris positions for {len(iris_data)} frames")


def example_4_get_eye_patches():
    """Get 30x30 eye patches for feature extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Get Eye Patches")
    print("=" * 60)
    
    detector = EyeDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("\nCapturing eye patches... Press 's' to save, 'q' to quit\n")
    
    patch_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        
        if result['status'] == 'ok':
            # Get 30x30 patches
            left_patch = np.array(result['left_eye']['eye_patch_30x30'], dtype=np.uint8)
            right_patch = np.array(result['right_eye']['eye_patch_30x30'], dtype=np.uint8)
            
            # Display patches (enlarged)
            left_large = cv2.resize(left_patch, (150, 150), interpolation=cv2.INTER_NEAREST)
            right_large = cv2.resize(right_patch, (150, 150), interpolation=cv2.INTER_NEAREST)
            patches = np.hstack([left_large, right_large])
            
            cv2.putText(patches, "Left Eye", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.putText(patches, "Right Eye", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.putText(patches, "Press 's' to save", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            
            cv2.imshow('Eye Patches (30x30)', patches)
        
        cv2.imshow('Original', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and result['status'] == 'ok':
            # Save patches
            cv2.imwrite(f'left_eye_patch_{patch_count}.png', left_patch)
            cv2.imwrite(f'right_eye_patch_{patch_count}.png', right_patch)
            print(f"Saved patch pair {patch_count}")
            patch_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Saved {patch_count} patch pairs")


def visualize_detection(frame: np.ndarray, result: dict) -> np.ndarray:
    """Visualize detection results on frame."""
    display = frame.copy()
    
    # Draw face
    if result['face_bbox'] is not None:
        x, y, w, h = result['face_bbox']
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw eyes and iris
    if result['status'] in ['ok', 'partial']:
        for eye_name, color in [('left_eye', (255, 0, 0)), ('right_eye', (0, 255, 255))]:
            if result[eye_name] is not None:
                eye = result[eye_name]
                
                # Eye bbox
                ex, ey, ew, eh = eye['bbox']
                cv2.rectangle(display, (ex, ey), (ex + ew, ey + eh), color, 2)
                
                # Iris center
                ix, iy = eye['iris_center']
                cv2.circle(display, (int(ix), int(iy)), 3, color, -1)
                
                # Iris bbox
                ibx, iby, ibw, ibh = eye['iris_bbox']
                cv2.rectangle(display, (ibx, iby), (ibx + ibw, iby + ibh), color, 1)
    
    # Status text
    status_color = {
        'ok': (0, 255, 0),
        'no_face': (0, 0, 255),
        'no_eyes': (0, 165, 255),
        'partial': (0, 255, 255)
    }
    cv2.putText(display, f"Status: {result['status']}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color[result['status']], 2)
    
    return display


if __name__ == "__main__":
    print("\nEye Detection Module - Examples\n")
    print("Choose an example:")
    print("1. Basic usage (real-time display)")
    print("2. Save results to JSON")
    print("3. Extract iris positions only")
    print("4. Get 30x30 eye patches")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        example_1_basic_usage()
    elif choice == '2':
        example_2_save_to_json()
    elif choice == '3':
        example_3_extract_iris_positions()
    elif choice == '4':
        example_4_get_eye_patches()
    else:
        print("Invalid choice")



