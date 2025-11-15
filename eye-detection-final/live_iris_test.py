"""
Live Iris Detection Test
Shows real-time camera feed with iris centers marked as red dots.
Press 'q' to quit, 's' to save a screenshot.
"""

import cv2
import numpy as np
from eye_detector import EyeDetector
import time


def draw_iris_visualization(frame, result):
    """
    Draw visualization on frame showing detected face, eyes, and iris centers.
    
    Args:
        frame: Original camera frame
        result: Detection result from EyeDetector
    
    Returns:
        Annotated frame
    """
    vis_frame = frame.copy()
    
    # Draw face bounding box
    if result['face_bbox'] is not None:
        fx, fy, fw, fh = result['face_bbox']
        cv2.rectangle(vis_frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
        cv2.putText(vis_frame, "Face", (fx, fy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw left eye
    if result['left_eye'] is not None:
        left_eye = result['left_eye']
        ex, ey, ew, eh = left_eye['bbox']
        
        # Eye bounding box (green)
        cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(vis_frame, "L Eye", (ex, ey - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Iris center (red dot)
        iris_cx, iris_cy = left_eye['iris_center']
        cv2.circle(vis_frame, (iris_cx, iris_cy), 5, (0, 0, 255), -1)
        cv2.circle(vis_frame, (iris_cx, iris_cy), 8, (0, 0, 255), 2)
        
        # Iris bounding box (blue)
        iris_x, iris_y, iris_w, iris_h = left_eye['iris_bbox']
        cv2.rectangle(vis_frame, (iris_x, iris_y), 
                     (iris_x + iris_w, iris_y + iris_h), (255, 0, 0), 1)
        
        # Show relative position
        rel_x, rel_y = left_eye['iris_center_rel']
        cv2.putText(vis_frame, f"({rel_x:.2f}, {rel_y:.2f})", 
                   (ex, ey + eh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Draw right eye
    if result['right_eye'] is not None:
        right_eye = result['right_eye']
        ex, ey, ew, eh = right_eye['bbox']
        
        # Eye bounding box (green)
        cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(vis_frame, "R Eye", (ex, ey - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Iris center (red dot)
        iris_cx, iris_cy = right_eye['iris_center']
        cv2.circle(vis_frame, (iris_cx, iris_cy), 5, (0, 0, 255), -1)
        cv2.circle(vis_frame, (iris_cx, iris_cy), 8, (0, 0, 255), 2)
        
        # Iris bounding box (blue)
        iris_x, iris_y, iris_w, iris_h = right_eye['iris_bbox']
        cv2.rectangle(vis_frame, (iris_x, iris_y), 
                     (iris_x + iris_w, iris_y + iris_h), (255, 0, 0), 1)
        
        # Show relative position
        rel_x, rel_y = right_eye['iris_center_rel']
        cv2.putText(vis_frame, f"({rel_x:.2f}, {rel_y:.2f})", 
                   (ex, ey + eh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    return vis_frame


def draw_status_info(frame, result, fps, frame_count):
    """Draw status information overlay."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Status text
    status_color = {
        'ok': (0, 255, 0),
        'no_face': (0, 0, 255),
        'no_eyes': (0, 165, 255),
        'partial': (0, 255, 255)
    }
    
    status = result['status']
    color = status_color.get(status, (255, 255, 255))
    
    cv2.putText(frame, f"Status: {status.upper()}", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "RED DOT = Iris Center", (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return frame


def create_eye_patches_view(result):
    """Create a side panel showing 30x30 eye patches."""
    if result['left_eye'] is None or result['right_eye'] is None:
        return None
    
    left_patch = np.array(result['left_eye']['eye_patch_30x30'], dtype=np.uint8)
    right_patch = np.array(result['right_eye']['eye_patch_30x30'], dtype=np.uint8)
    
    # Scale up for better visibility (30x30 -> 150x150)
    left_scaled = cv2.resize(left_patch, (150, 150), interpolation=cv2.INTER_NEAREST)
    right_scaled = cv2.resize(right_patch, (150, 150), interpolation=cv2.INTER_NEAREST)
    
    # Convert to BGR for color overlay
    left_bgr = cv2.cvtColor(left_scaled, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_scaled, cv2.COLOR_GRAY2BGR)
    
    # Draw iris center on scaled patches
    left_rel = result['left_eye']['iris_center_rel']
    right_rel = result['right_eye']['iris_center_rel']
    
    left_cx = int(left_rel[0] * 150)
    left_cy = int(left_rel[1] * 150)
    right_cx = int(right_rel[0] * 150)
    right_cy = int(right_rel[1] * 150)
    
    cv2.circle(left_bgr, (left_cx, left_cy), 8, (0, 0, 255), -1)
    cv2.circle(right_bgr, (right_cx, right_cy), 8, (0, 0, 255), -1)
    
    # Stack vertically with labels
    panel = np.zeros((370, 150, 3), dtype=np.uint8)
    
    # Left eye
    cv2.putText(panel, "Left Eye 30x30", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    panel[30:180, :] = left_bgr
    
    # Right eye
    cv2.putText(panel, "Right Eye 30x30", (10, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    panel[220:370, :] = right_bgr
    
    return panel


def live_iris_test():
    """Run live iris detection test with visualization."""
    print("=" * 60)
    print("LIVE IRIS DETECTION TEST")
    print("=" * 60)
    print("\nInitializing detector...")
    
    try:
        detector = EyeDetector()
        print("[OK] Detector initialized")
    except Exception as e:
        print(f"[FAIL] Failed to initialize detector: {e}")
        return
    
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[FAIL] Could not open camera")
        return
    
    print("[OK] Camera opened")
    print("\n" + "=" * 60)
    print("LIVE VIEW - Look at the camera!")
    print("=" * 60)
    print("\nVisualization Legend:")
    print("  - YELLOW box: Detected face")
    print("  - GREEN boxes: Detected eyes")
    print("  - BLUE boxes: Iris regions")
    print("  - RED DOTS: Iris centers (pupil position)")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Position yourself 60-70 cm from camera")
    print("\n" + "=" * 60)
    
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[FAIL] Failed to read frame")
                break
            
            # Detect
            result = detector.detect(frame, frame_id=frame_count)
            
            # Draw visualization
            vis_frame = draw_iris_visualization(frame, result)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Draw status info
            vis_frame = draw_status_info(vis_frame, result, fps, frame_count)
            
            # Create eye patches panel if both eyes detected
            if result['status'] == 'ok':
                try:
                    patches_panel = create_eye_patches_view(result)
                    if patches_panel is not None:
                        # Combine main view and patches side by side
                        h, w = vis_frame.shape[:2]
                        panel_h = patches_panel.shape[0]
                        
                        # Resize panel to match frame height
                        panel_resized = cv2.resize(patches_panel, 
                                                  (int(150 * h / panel_h), h))
                        
                        # Concatenate horizontally
                        vis_frame = np.hstack([vis_frame, panel_resized])
                except Exception as e:
                    # If panel creation fails, just show main view
                    pass
            
            # Display
            try:
                cv2.imshow('Live Iris Detection Test', vis_frame)
            except:
                # Window was closed
                print("\n[OK] Window closed by user")
                break
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[OK] User quit")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"iris_screenshot_{screenshot_count}.png"
                cv2.imwrite(filename, vis_frame)
                print(f"\n[OK] Screenshot saved: {filename}")
            
            # Check if window was closed
            if cv2.getWindowProperty('Live Iris Detection Test', cv2.WND_PROP_VISIBLE) < 1:
                print("\n[OK] Window closed by user")
                break
    
    except KeyboardInterrupt:
        print("\n[OK] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Screenshots saved: {screenshot_count}")
        print("\nThank you for testing!")


if __name__ == "__main__":
    live_iris_test()

