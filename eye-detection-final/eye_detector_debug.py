"""
Eye Detector Debug Version
Takes a single image and saves visualization of each processing step.

Usage:
    python eye_detector_debug.py <image_path> [output_folder]
"""

import os
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
import sys


class EyeDetectorDebug:
    """Eye detection with step-by-step visualization output."""

    def __init__(self, output_folder: str = "debug_output"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Load Haar Cascade
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_paths = [
            os.path.join(base_dir, 'haarcascade_frontalface_default.xml'),
            os.path.join(base_dir, '..', 'haarcascade_frontalface_default.xml'),
        ]
        try:
            if hasattr(cv2, 'data') and cv2.data.haarcascades:
                cascade_paths.append(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        except:
            pass

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
            raise RuntimeError("Failed to load Haar Cascade.")

        self.step_counter = 0

    def _save_step(self, name: str, image: np.ndarray, is_color: bool = False):
        """Save an image with step number prefix."""
        self.step_counter += 1
        filename = f"{self.step_counter:02d}_{name}.png"
        filepath = os.path.join(self.output_folder, filename)
        
        if not is_color and len(image.shape) == 2:
            # Convert grayscale to BGR for consistent saving
            save_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            save_img = image.copy()
        
        cv2.imwrite(filepath, save_img)
        print(f"  Saved: {filename}")

    def detect(self, image_path: str) -> Dict[str, Any]:
        """Main detection with step-by-step visualization."""
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"Output folder: {self.output_folder}")
        print(f"{'='*60}\n")
        
        # Step 1: Load original image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"ERROR: Cannot load image: {image_path}")
            return {"status": "error", "message": "Cannot load image"}
        
        self._save_step("01_original", frame, is_color=True)
        
        result = {
            "status": "no_face",
            "face_bbox": None,
            "left_eye": None,
            "right_eye": None
        }

        # Step 2: Detect face
        face_bbox = self._detect_face(frame)
        if face_bbox is None:
            print("\nResult: NO FACE DETECTED")
            return result

        result["face_bbox"] = [int(x) for x in face_bbox]
        result["status"] = "no_eyes"

        # Step 3: Detect eyes
        eye_pair = self._detect_eyes(frame, face_bbox)
        if eye_pair is None:
            # Save final result with face bbox even if no eyes
            self._save_final_result(frame, result)
            print("\nResult: FACE FOUND, NO EYES DETECTED")
            return result

        # Step 4: Process each eye
        left_eye_data, right_eye_data = eye_pair
        left_eye_full = self._process_eye(frame, face_bbox, left_eye_data, "left")
        right_eye_full = self._process_eye(frame, face_bbox, right_eye_data, "right")

        if left_eye_full is None or right_eye_full is None:
            result["status"] = "partial"
        else:
            result["status"] = "ok"

        result["left_eye"] = left_eye_full
        result["right_eye"] = right_eye_full

        # Save final result with all boxes
        self._save_final_result(frame, result)
        
        print(f"\nResult: {result['status'].upper()}")
        return result

    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face with visualization."""
        print("\n--- FACE DETECTION ---")
        
        # Grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._save_step("02_grayscale_for_face", gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )
        
        if len(faces) == 0:
            print("  No faces detected")
            return None
        
        # Show all detected faces
        all_faces_img = frame.copy()
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(all_faces_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(all_faces_img, f"Face {i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self._save_step("03_all_detected_faces", all_faces_img, is_color=True)
        
        # Select largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest
        
        selected_face_img = frame.copy()
        cv2.rectangle(selected_face_img, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 3)
        cv2.putText(selected_face_img, "SELECTED FACE", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        self._save_step("04_selected_face", selected_face_img, is_color=True)
        
        print(f"  Found {len(faces)} face(s), selected largest: ({fx}, {fy}, {fw}, {fh})")
        return tuple(map(int, largest))

    def _detect_eyes(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]):
        """Detect eyes with step-by-step visualization."""
        print("\n--- EYE DETECTION ---")
        
        face_x, face_y, face_w, face_h = face_bbox
        face_roi = frame[face_y:face_y+face_h, face_x:face_x+face_w]
        self._save_step("05_face_roi", face_roi, is_color=True)

        # BT.709 Grayscale
        gray = self._bt709_grayscale(face_roi)
        self._save_step("06_bt709_grayscale", gray)
        
        # Upper half for median calculation
        upper_half = gray[:face_h // 2, :]
        median_val = np.median(upper_half)
        print(f"  Median intensity (upper half): {median_val:.1f}")
        
        # Visualize upper half
        upper_vis = np.zeros_like(gray)
        upper_vis[:face_h // 2, :] = upper_half
        self._save_step("07_upper_half_for_median", upper_vis)

        # Histogram equalization
        gray_eq = cv2.equalizeHist(gray)
        self._save_step("08_histogram_equalized", gray_eq)
        
        # Blended
        gray_blended = cv2.addWeighted(gray, 0.6, gray_eq, 0.4, 0)
        self._save_step("09_blended_0.6_orig_0.4_eq", gray_blended)
        
        # Filtered with median
        filtered = np.minimum(gray_blended, median_val).astype(np.uint8)
        self._save_step("10_filtered_min_with_median", filtered)

        # Binarization
        binary = self._binarize(filtered)
        self._save_step("11_binarized", binary)
        
        # Connected components
        components = self._find_components(binary, face_w, face_h)
        
        # Visualize components
        components_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, comp in enumerate(components):
            x, y, w, h = comp['bbox']
            color = colors[i % len(colors)]
            cv2.rectangle(components_vis, (x, y), (x+w, y+h), color, 2)
            cv2.circle(components_vis, (int(comp['centroid'][0]), int(comp['centroid'][1])), 3, color, -1)
            cv2.putText(components_vis, f"C{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        self._save_step("12_candidate_components", components_vis, is_color=True)
        
        print(f"  Found {len(components)} candidate eye regions")
        
        # Select eye pair
        eye_pair = self._select_eye_pair(components, face_w, face_h)
        
        if eye_pair is None:
            print("  No valid eye pair found")
            return None
        
        # Visualize selected pair
        left_eye, right_eye = eye_pair
        pair_vis = face_roi.copy()
        
        lx, ly, lw, lh = left_eye['bbox']
        rx, ry, rw, rh = right_eye['bbox']
        
        cv2.rectangle(pair_vis, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)
        cv2.putText(pair_vis, "LEFT", (lx, ly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.rectangle(pair_vis, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2)
        cv2.putText(pair_vis, "RIGHT", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        self._save_step("13_selected_eye_pair", pair_vis, is_color=True)
        
        print(f"  Selected eye pair - Left: {left_eye['bbox']}, Right: {right_eye['bbox']}")
        return eye_pair

    def _process_eye(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                     eye_data: Dict, eye_name: str) -> Optional[Dict[str, Any]]:
        """Process single eye with iris detection visualization."""
        print(f"\n--- PROCESSING {eye_name.upper()} EYE ---")
        
        face_x, face_y, face_w, face_h = face_bbox
        eye_x, eye_y, eye_w, eye_h = eye_data['bbox']

        x_global = face_x + eye_x
        y_global = face_y + eye_y

        eye_region = frame[y_global:y_global+eye_h, x_global:x_global+eye_w]
        if eye_region.size == 0:
            return None

        self._save_step(f"14_{eye_name}_eye_region", eye_region, is_color=True)
        
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        self._save_step(f"15_{eye_name}_eye_grayscale", eye_gray)
        
        eye_patch_30x30 = cv2.resize(eye_gray, (30, 30))
        # Save enlarged version for visibility
        eye_patch_enlarged = cv2.resize(eye_patch_30x30, (150, 150), interpolation=cv2.INTER_NEAREST)
        self._save_step(f"16_{eye_name}_eye_patch_30x30_enlarged", eye_patch_enlarged)
        
        iris_info = self._detect_iris(eye_patch_30x30, eye_name)

        if iris_info is not None:
            scale_x = eye_w / 30.0
            scale_y = eye_h / 30.0
            iris_x_in_eye = int(iris_info['x'] * scale_x)
            iris_y_in_eye = int(iris_info['y'] * scale_y)
            iris_w_scaled = int(iris_info['w'] * scale_x)
            iris_h_scaled = int(iris_info['h'] * scale_y)
            iris_cx_in_eye = iris_x_in_eye + iris_w_scaled // 2
            iris_cy_in_eye = iris_y_in_eye + iris_h_scaled // 2

            iris_x_global = x_global + iris_x_in_eye
            iris_y_global = y_global + iris_y_in_eye
            iris_cx_global = x_global + iris_cx_in_eye
            iris_cy_global = y_global + iris_cy_in_eye
            iris_cx_rel = iris_cx_in_eye / eye_w if eye_w > 0 else 0.5
            iris_cy_rel = iris_cy_in_eye / eye_h if eye_h > 0 else 0.5
            
            print(f"  Iris found at ({iris_cx_global}, {iris_cy_global})")
        else:
            iris_x_global = x_global
            iris_y_global = y_global
            iris_w_scaled = eye_w
            iris_h_scaled = eye_h
            iris_cx_global = x_global + eye_w // 2
            iris_cy_global = y_global + eye_h // 2
            iris_cx_rel = 0.5
            iris_cy_rel = 0.5
            print(f"  Iris not detected, using eye center")

        quality = self._assess_quality(eye_patch_30x30)
        return {
            "bbox": [int(x_global), int(y_global), int(eye_w), int(eye_h)],
            "eye_patch_30x30": eye_patch_30x30.tolist(),
            "iris_bbox": [int(iris_x_global), int(iris_y_global), int(iris_w_scaled), int(iris_h_scaled)],
            "iris_center": [int(iris_cx_global), int(iris_cy_global)],
            "iris_center_rel": [float(iris_cx_rel), float(iris_cy_rel)],
            "quality": quality
        }

    def _detect_iris(self, eye_patch: np.ndarray, eye_name: str) -> Optional[Dict[str, int]]:
        """Detect iris with visualization of each step."""
        print(f"  --- Iris detection for {eye_name} ---")
        
        # Histogram equalization
        equalized = cv2.equalizeHist(eye_patch)
        eq_enlarged = cv2.resize(equalized, (150, 150), interpolation=cv2.INTER_NEAREST)
        self._save_step(f"17_{eye_name}_iris_equalized", eq_enlarged)
        
        # Convert to YCbCr
        eye_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        ycbcr = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel = ycbcr[:, :, 0]
        
        y_enlarged = cv2.resize(y_channel, (150, 150), interpolation=cv2.INTER_NEAREST)
        self._save_step(f"18_{eye_name}_ycbcr_y_channel", y_enlarged)

        # Threshold at 10th percentile (darkest 10%)
        threshold = np.percentile(y_channel, 10)
        print(f"    Y-channel 10th percentile threshold: {threshold:.1f}")
        
        iris_mask = (y_channel <= threshold).astype(np.uint8) * 255
        mask_enlarged = cv2.resize(iris_mask, (150, 150), interpolation=cv2.INTER_NEAREST)
        self._save_step(f"19_{eye_name}_iris_mask_darkest_10pct", mask_enlarged)
        
        # Connected components on iris mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(iris_mask, connectivity=4)
        
        # Visualize components
        comp_vis = np.zeros((30, 30, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i in range(1, num_labels):
            color = colors[(i-1) % len(colors)]
            comp_vis[labels == i] = color
        comp_vis_enlarged = cv2.resize(comp_vis, (150, 150), interpolation=cv2.INTER_NEAREST)
        self._save_step(f"20_{eye_name}_iris_components", comp_vis_enlarged, is_color=True)

        if num_labels <= 1:
            print(f"    No iris components found")
            return None

        # Select largest component
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        
        # Visualize selected iris
        iris_vis = cv2.cvtColor(eye_patch, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(iris_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cx, cy = int(centroids[largest_idx][0]), int(centroids[largest_idx][1])
        cv2.circle(iris_vis, (cx, cy), 2, (0, 0, 255), -1)
        iris_vis_enlarged = cv2.resize(iris_vis, (150, 150), interpolation=cv2.INTER_NEAREST)
        self._save_step(f"21_{eye_name}_iris_detected", iris_vis_enlarged, is_color=True)
        
        print(f"    Selected iris bbox: ({x}, {y}, {w}, {h})")
        return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}

    def _save_final_result(self, frame: np.ndarray, result: Dict):
        """Save final image with all bounding boxes."""
        print("\n--- FINAL RESULT ---")
        final = frame.copy()
        
        # Face (yellow)
        if result['face_bbox']:
            fx, fy, fw, fh = result['face_bbox']
            cv2.rectangle(final, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 3)
            cv2.putText(final, "FACE", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Left eye (green)
        if result['left_eye']:
            ex, ey, ew, eh = result['left_eye']['bbox']
            cv2.rectangle(final, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(final, "L_EYE", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Iris (red)
            ix, iy, iw, ih = result['left_eye']['iris_bbox']
            cv2.rectangle(final, (ix, iy), (ix+iw, iy+ih), (0, 0, 255), 1)
            icx, icy = result['left_eye']['iris_center']
            cv2.circle(final, (icx, icy), 4, (0, 0, 255), -1)
        
        # Right eye (blue)
        if result['right_eye']:
            ex, ey, ew, eh = result['right_eye']['bbox']
            cv2.rectangle(final, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            cv2.putText(final, "R_EYE", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Iris (magenta)
            ix, iy, iw, ih = result['right_eye']['iris_bbox']
            cv2.rectangle(final, (ix, iy), (ix+iw, iy+ih), (255, 0, 255), 1)
            icx, icy = result['right_eye']['iris_center']
            cv2.circle(final, (icx, icy), 4, (255, 0, 255), -1)
        
        self._save_step("99_final_result", final, is_color=True)

    def _assess_quality(self, eye_patch: np.ndarray) -> Dict[str, Any]:
        mean_intensity = np.mean(eye_patch)
        std_intensity = np.std(eye_patch)
        illumination_ok = 30 < mean_intensity < 200
        return {
            "illumination_ok": bool(illumination_ok),
            "occlusion": 0.0,
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity)
        }

    def _bt709_grayscale(self, bgr_image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(bgr_image)
        return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Edge-based adaptive binarization with visualization."""
        img_float = image.astype(np.float32)
        padded = np.pad(img_float, pad_width=1, mode='edge')
        
        h_grad = np.abs(padded[1:-1, :-2] - padded[1:-1, 2:])
        v_grad = np.abs(padded[:-2, 1:-1] - padded[2:, 1:-1])
        edge_strength = np.maximum(h_grad, v_grad)
        
        # Save edge strength visualization
        edge_vis = (edge_strength / edge_strength.max() * 255).astype(np.uint8) if edge_strength.max() > 0 else edge_strength.astype(np.uint8)
        self._save_step("10a_edge_strength", edge_vis)

        numerator = np.sum(img_float * edge_strength)
        denominator = np.sum(edge_strength)
        threshold = numerator / denominator if denominator > 1e-6 else np.mean(img_float)
        threshold_adjusted = threshold * 1.05
        
        print(f"  Binarization threshold: {threshold_adjusted:.1f}")
        return (image <= threshold_adjusted).astype(np.uint8) * 255

    def _find_components(self, binary: np.ndarray, face_w: int, face_h: int):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)

        # Save all components visualization with labels
        all_comp = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        np.random.seed(42)  # Consistent colors
        for i in range(1, num_labels):
            color = np.random.randint(50, 255, 3).tolist()
            all_comp[labels == i] = color
        
        # Add component numbers
        all_comp_labeled = all_comp.copy()
        for i in range(1, num_labels):
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cv2.putText(all_comp_labeled, str(i), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        self._save_step("11a_all_connected_components", all_comp_labeled, is_color=True)

        face_area = face_w * face_h
        min_area = face_area * 0.003
        max_area = face_area * 0.18
        
        print(f"  Component filtering criteria:")
        print(f"    - Area: {min_area:.0f} - {max_area:.0f} pixels")
        print(f"    - Aspect ratio: 0.5 - 2.5")
        print(f"    - Position: below 35% of face height (cy > {face_h * 0.35:.0f})")
        print(f"  Total components found: {num_labels - 1}")

        # Visualize ALL components with rejection reasons
        rejected_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            rejection_reason = None
            
            if area < min_area:
                rejection_reason = f"area too small ({area:.0f}<{min_area:.0f})"
            elif area > max_area:
                rejection_reason = f"area too large ({area:.0f}>{max_area:.0f})"
            elif h == 0:
                rejection_reason = "zero height"
            elif w / h < 0.5:
                rejection_reason = f"aspect too narrow ({w/h:.2f}<0.5)"
            elif w / h > 2.5:
                rejection_reason = f"aspect too wide ({w/h:.2f}>2.5)"
            elif cy < face_h * 0.35:
                rejection_reason = f"too high (cy={cy:.0f}<{face_h*0.35:.0f})"
            
            if rejection_reason:
                # Draw rejected in red
                cv2.rectangle(rejected_vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
                print(f"    Component {i}: REJECTED - {rejection_reason}")
            else:
                # Draw accepted in green
                cv2.rectangle(rejected_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(rejected_vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                components.append({'bbox': (x, y, w, h), 'centroid': (cx, cy), 'area': area})
                print(f"    Component {i}: ACCEPTED - area={area:.0f}, aspect={w/h:.2f}, cy={cy:.0f}")
        
        self._save_step("11b_component_filtering", rejected_vis, is_color=True)
        print(f"  Accepted components: {len(components)}")
        
        return components

    def _select_eye_pair(self, components, face_w: int, face_h: int):
        if len(components) < 2:
            return None

        filtered = [c for c in components if 0.35 <= c['centroid'][1] / face_h <= 0.65]
        if len(filtered) < 2:
            filtered = components
        if len(filtered) < 2:
            return None

        best_pair = None
        best_score = -1

        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                c1, c2 = filtered[i], filtered[j]
                cx1, cy1 = c1['centroid']
                cx2, cy2 = c2['centroid']

                if abs(cx1 - cx2) < face_w * 0.15:
                    continue

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

        if best_score < 0.25:
            return None
        return best_pair


def main():
    if len(sys.argv) < 2:
        print("Usage: python eye_detector_debug.py <image_path> [output_folder]")
        print("\nExample:")
        print("  python eye_detector_debug.py photo.jpg")
        print("  python eye_detector_debug.py photo.jpg my_debug_output")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "debug_output"
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    detector = EyeDetectorDebug(output_folder=output_folder)
    result = detector.detect(image_path)
    
    print(f"\n{'='*60}")
    print(f"DETECTION COMPLETE")
    print(f"Status: {result['status']}")
    print(f"Output saved to: {output_folder}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

