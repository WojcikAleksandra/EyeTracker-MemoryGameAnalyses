"""
Eye Detection Module - Production Ready
Detects eyes and iris, outputs structured data for gaze estimation systems.

This module provides eye detection up to iris localization (Steps 1-7 of the algorithm).
Output format is designed for easy integration with gaze estimation pipelines.
"""

import os
import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple


class EyeDetector:
    """
    Complete eye detection pipeline: face → eyes → iris.
    
    Output includes:
    - Eye bounding boxes and 30x30 patches
    - Iris location (absolute and relative coordinates)
    - Quality metrics
    - Timestamps and frame info
    """
    
    def __init__(self, 
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_face_size: Tuple[int, int] = (120, 120)):
        """
        Initialize eye detector.
        
        Args:
            scale_factor: Haar cascade scale factor
            min_neighbors: Haar cascade min neighbors
            min_face_size: Minimum face size in pixels
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load Haar Cascade
        # cascade_paths = [
        #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else '',
        #     'haarcascade_frontalface_default.xml',
        #     '../haarcascade_frontalface_default.xml',
        # ]

        cascade_paths = [
            os.path.join(base_dir, 'haarcascade_frontalface_default.xml'),
            os.path.join(base_dir, '..', 'haarcascade_frontalface_default.xml'),
        ]
        
        self.face_cascade = None
        for path in cascade_paths:
            if not path:
                continue
            try:
                temp = cv2.CascadeClassifier(path)
                if not temp.empty():
                    self.face_cascade = temp
                    break
            except:
                continue
        
        if self.face_cascade is None or self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade. Ensure OpenCV is installed correctly.")
        
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        
        self.frame_count = 0
        self.start_time = time.time()
    
    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect eyes and iris in frame.
        
        Args:
            frame: Input BGR image (e.g., 640x480)
            frame_id: Optional frame identifier
        
        Returns:
            Dictionary with detection results in standard format
        """
        self.frame_count += 1
        if frame_id is None:
            frame_id = self.frame_count
        
        timestamp_ms = int(time.time() * 1000)
        
        # Initialize result structure
        result = {
            "frame_id": int(frame_id),
            "timestamp_ms": int(timestamp_ms),
            "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "status": "no_face",
            "face_bbox": None,
            "left_eye": None,
            "right_eye": None
        }
        
        # Step 1: Detect face
        face_bbox = self._detect_face(frame)
        if face_bbox is None:
            return result
        
        result["face_bbox"] = [int(x) for x in face_bbox]
        result["status"] = "no_eyes"
        
        # Step 2-6: Detect eyes
        eye_pair = self._detect_eyes(frame, face_bbox)
        if eye_pair is None:
            return result
        
        left_eye_data, right_eye_data = eye_pair
        
        # Step 7: Detect iris for each eye
        left_eye_full = self._process_eye(frame, face_bbox, left_eye_data)
        right_eye_full = self._process_eye(frame, face_bbox, right_eye_data)
        
        if left_eye_full is None or right_eye_full is None:
            result["status"] = "partial"
        else:
            result["status"] = "ok"
        
        result["left_eye"] = left_eye_full
        result["right_eye"] = right_eye_full
        
        return result
    
    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        return tuple(map(int, largest))
    
    def _detect_eyes(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]):
        """Detect eye pair (Steps 2-6 of algorithm)."""
        face_x, face_y, face_w, face_h = face_bbox
        face_roi = frame[face_y:face_y+face_h, face_x:face_x+face_w]
        
        # Illuminance filtering (BT.709)
        gray = self._bt709_grayscale(face_roi)
        
        # Use adaptive median clipping for better contrast
        # This helps when eyes have less contrast (looking straight)
        upper_half = gray[:face_h // 2, :]
        median_val = np.median(upper_half)
        
        # Apply histogram equalization before filtering for better contrast
        gray_eq = cv2.equalizeHist(gray)
        
        # Blend original and equalized for better detection
        gray_blended = cv2.addWeighted(gray, 0.6, gray_eq, 0.4, 0)
        
        # Apply median clipping
        filtered = np.minimum(gray_blended, median_val).astype(np.uint8)
        
        # Binarization
        binary = self._binarize(filtered)
        
        # Connected components
        components = self._find_components(binary, face_w, face_h)
        
        # Select eye pair
        eye_pair = self._select_eye_pair(components, face_w, face_h)
        
        return eye_pair
    
    def _process_eye(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                     eye_data: Dict) -> Optional[Dict[str, Any]]:
        """Process single eye: extract patch and detect iris."""
        face_x, face_y, face_w, face_h = face_bbox
        eye_x, eye_y, eye_w, eye_h = eye_data['bbox']
        
        # Global coordinates
        x_global = face_x + eye_x
        y_global = face_y + eye_y
        
        # Extract eye region
        eye_region = frame[y_global:y_global+eye_h, x_global:x_global+eye_w]
        if eye_region.size == 0:
            return None
        
        # Convert to grayscale
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        
        # Resize to 30x30 (as per paper)
        eye_patch_30x30 = cv2.resize(eye_gray, (30, 30))
        
        # Detect iris
        iris_info = self._detect_iris(eye_patch_30x30)
        
        # Scale iris coordinates back to eye region size
        if iris_info is not None:
            scale_x = eye_w / 30.0
            scale_y = eye_h / 30.0
            
            iris_x_in_eye = int(iris_info['x'] * scale_x)
            iris_y_in_eye = int(iris_info['y'] * scale_y)
            iris_w_scaled = int(iris_info['w'] * scale_x)
            iris_h_scaled = int(iris_info['h'] * scale_y)
            
            iris_cx_in_eye = iris_x_in_eye + iris_w_scaled // 2
            iris_cy_in_eye = iris_y_in_eye + iris_h_scaled // 2
            
            # Global iris coordinates
            iris_x_global = x_global + iris_x_in_eye
            iris_y_global = y_global + iris_y_in_eye
            iris_cx_global = x_global + iris_cx_in_eye
            iris_cy_global = y_global + iris_cy_in_eye
            
            # Relative coordinates (0-1 range)
            iris_cx_rel = iris_cx_in_eye / eye_w if eye_w > 0 else 0.5
            iris_cy_rel = iris_cy_in_eye / eye_h if eye_h > 0 else 0.5
        else:
            # No iris detected - use eye center as fallback
            iris_x_global = x_global
            iris_y_global = y_global
            iris_w_scaled = eye_w
            iris_h_scaled = eye_h
            iris_cx_global = x_global + eye_w // 2
            iris_cy_global = y_global + eye_h // 2
            iris_cx_rel = 0.5
            iris_cy_rel = 0.5
        
        # Quality assessment
        quality = self._assess_quality(eye_patch_30x30)
        
        return {
            "bbox": [int(x_global), int(y_global), int(eye_w), int(eye_h)],
            "eye_patch_30x30": eye_patch_30x30.tolist(),  # Can be serialized to JSON
            "iris_bbox": [int(iris_x_global), int(iris_y_global), int(iris_w_scaled), int(iris_h_scaled)],
            "iris_center": [int(iris_cx_global), int(iris_cy_global)],
            "iris_center_rel": [float(iris_cx_rel), float(iris_cy_rel)],
            "quality": quality
        }
    
    def _detect_iris(self, eye_patch: np.ndarray) -> Optional[Dict[str, int]]:
        """Detect iris in 30x30 eye patch."""
        # Histogram equalization
        equalized = cv2.equalizeHist(eye_patch)
        
        # Convert to YCbCr
        eye_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        ycbcr = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel = ycbcr[:, :, 0]
        
        # Bottom 10% (darkest pixels)
        threshold = np.percentile(y_channel, 10)
        iris_mask = (y_channel <= threshold).astype(np.uint8) * 255
        
        # Find largest component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(iris_mask, connectivity=4)
        
        if num_labels <= 1:
            return None
        
        # Largest component (skip background)
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        
        return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
    
    def _assess_quality(self, eye_patch: np.ndarray) -> Dict[str, Any]:
        """Assess eye patch quality."""
        mean_intensity = np.mean(eye_patch)
        std_intensity = np.std(eye_patch)
        
        # Simple heuristics
        illumination_ok = 30 < mean_intensity < 200
        occlusion = 0.0  # Placeholder - could check for large uniform regions
        
        return {
            "illumination_ok": bool(illumination_ok),
            "occlusion": float(occlusion),
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity)
        }
    
    # Helper methods from original implementation
    
    def _bt709_grayscale(self, bgr_image: np.ndarray) -> np.ndarray:
        """BT.709 grayscale conversion."""
        b, g, r = cv2.split(bgr_image)
        return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Edge-based binarization with adaptive threshold."""
        img_float = image.astype(np.float32)
        padded = np.pad(img_float, pad_width=1, mode='edge')
        
        h_grad = np.abs(padded[1:-1, :-2] - padded[1:-1, 2:])
        v_grad = np.abs(padded[:-2, 1:-1] - padded[2:, 1:-1])
        edge_strength = np.maximum(h_grad, v_grad)
        
        numerator = np.sum(img_float * edge_strength)
        denominator = np.sum(edge_strength)
        threshold = numerator / denominator if denominator > 1e-6 else np.mean(img_float)
        
        # Make threshold slightly more permissive for low-contrast cases
        # This helps detect eyes when looking straight (less contrast)
        threshold_adjusted = threshold * 1.05  # 5% more permissive
        
        return (image <= threshold_adjusted).astype(np.uint8) * 255
    
    def _find_components(self, binary: np.ndarray, face_w: int, face_h: int):
        """Find and filter eye candidates."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
        
        face_area = face_w * face_h
        # Slightly more permissive area constraints for low-contrast eyes
        min_area = face_area * 0.003  # Reduced from 0.005
        max_area = face_area * 0.18   # Increased from 0.15
        
        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            if h == 0:
                continue
            
            aspect = w / h
            if aspect < 0.5 or aspect > 2.5:
                continue
            
            cx, cy = centroids[i]
            
            # Reject upper 35% (eyebrows)
            if cy < face_h * 0.35:
                continue
            
            components.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'area': area
            })
        
        return components
    
    def _select_eye_pair(self, components, face_w: int, face_h: int):
        """Select best eye pair."""
        if len(components) < 2:
            return None
        
        # Filter by grid zones (middle row, center columns)
        filtered = [c for c in components if 0.35 <= c['centroid'][1] / face_h <= 0.65]
        
        if len(filtered) < 2:
            filtered = components
        
        if len(filtered) < 2:
            return None
        
        # Find best symmetric pair
        best_pair = None
        best_score = -1
        
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                c1, c2 = filtered[i], filtered[j]
                
                cx1, cy1 = c1['centroid']
                cx2, cy2 = c2['centroid']
                
                # Must be horizontally separated
                if abs(cx1 - cx2) < face_w * 0.15:
                    continue
                
                # Symmetry score
                height_diff = abs(cy1 - cy2)
                height_score = max(0, 1.0 - height_diff / 16.0)
                
                area_ratio = min(c1['area'], c2['area']) / max(c1['area'], c2['area'])
                
                score = 0.7 * (height_score + area_ratio) / 2.0 + 0.3 * 1.0
                
                if score > best_score:
                    best_score = score
                    if cx1 < cx2:
                        best_pair = (c1, c2)
                    else:
                        best_pair = (c2, c1)
        
        # Relaxed threshold for low-contrast detection (was 0.3)
        # This helps when eyes are less prominent (looking straight)
        if best_score < 0.25:
            return None
        
        return best_pair

