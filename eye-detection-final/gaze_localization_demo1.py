import cv2
import numpy as np
from collections import deque
from sklearn.linear_model import Ridge
import time
import ctypes
import math

from eye_detector import EyeDetector


# ---------- Auxiliary visualization of eye detection in camera ----------

def visualize_detection(frame: np.ndarray, result: dict) -> np.ndarray:
    display = frame.copy()

    # Face
    if result.get("face_bbox") is not None:
        x, y, w, h = result["face_bbox"]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Eyes + iris
    if result.get("status") in ["ok", "partial"]:
        for eye_name, color in [("left_eye", (255, 0, 0)), ("right_eye", (0, 255, 255))]:
            eye = result.get(eye_name)
            if eye is None:
                continue

            ex, ey, ew, eh = eye["bbox"]
            cv2.rectangle(display, (ex, ey), (ex + ew, ey + eh), color, 2)

            ix, iy = eye["iris_center"]
            cv2.circle(display, (int(ix), int(iy)), 3, color, -1)

            ibx, iby, ibw, ibh = eye["iris_bbox"]
            cv2.rectangle(display, (ibx, iby), (ibx + ibw, iby + ibh), color, 1)

    return display


# ---------- Features extraction ----------

class GazeFeatureExtractor:
    """
    Creates feature vector based on EyeDetector result.
    Features:
    [xL, yL, xR, yR, Δd, x̄, ȳ, θ, x_face, y_face]
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def __call__(self, result: dict) -> np.ndarray:
        left = result["left_eye"]
        right = result["right_eye"]
        face_x, face_y, face_w, face_h = result["face_bbox"]

        # Normalized pupil coordinates within the eye rectangle
        xL, yL = left["iris_center_rel"]
        xR, yR = right["iris_center_rel"]

        # Pupillary distance (in normalized space of eyes)
        dx = xR - xL
        dy = yR - yL
        dist = float(np.sqrt(dx * dx + dy * dy))

        # Average pupil position
        x_mean = (xL + xR) / 2.0
        y_mean = (yL + yR) / 2.0

        # Pupil line angle
        ixL, iyL = left["iris_center"]
        ixR, iyR = right["iris_center"]
        theta = float(np.arctan2(iyR - iyL, ixR - ixL))

        # Face position in image – center of face rectangle, normalized to [0,1]
        face_cx = face_x + face_w / 2.0
        face_cy = face_y + face_h / 2.0
        x_face = face_cx / self.frame_width
        y_face = face_cy / self.frame_height

        features = np.array(
            [xL, yL, xR, yR, dist, x_mean, y_mean, theta, x_face, y_face],
            dtype=np.float32,
        )
        return features


class EyeFrameValidator:
    """
    Sprawdza, czy pojedyncza klatka z detekcją nadaje się do użycia
    (zarówno w kalibracji, jak i w śledzeniu).

    Kryteria:
    - obie źrenice wykryte,
    - jakość oczu (illumination_ok + std_intensity >= min_std),
    - symetria źrenic względem osi pionowej twarzy,
    - prawie poziomy odcinek łączący źrenice.
    """

    def __init__(
        self,
        symmetry_tolerance: float = 0.2,       # ułamek szerokości twarzy
        angle_tolerance_deg: float = 15.0,     # maks. odchylenie linii źrenic od poziomu
        min_std_intensity: float = 10.0,       # minimum dla std_intensity
    ):
        self.symmetry_tolerance = symmetry_tolerance
        self.angle_tolerance_rad = math.radians(angle_tolerance_deg)
        self.min_std_intensity = min_std_intensity

    def eye_quality_ok(self, q: dict) -> bool:
        """Sprawdza jakość pojedynczego oka."""
        if not q.get("illumination_ok", False):
            return False
        std_intensity = q.get("std_intensity", 0.0)
        if std_intensity < self.min_std_intensity:
            return False
        return True

    def _pupils_detected(self, result: dict) -> bool:
        """
        Sprawdza, czy obie źrenice zostały wykryte.
        Jeśli w słowniku oka jest flaga 'iris_detected', używa jej.
        W przeciwnym razie zakłada, że jeśli status == 'ok' i oko nie jest None,
        to źrenica została wykryta.
        """
        if result.get("status") != "ok":
            return False

        left = result.get("left_eye")
        right = result.get("right_eye")
        if left is None or right is None:
            return False

        left_flag = left.get("iris_detected")
        right_flag = right.get("iris_detected")
        if left_flag is not None and right_flag is not None:
            return bool(left_flag) and bool(right_flag)

        # Tryb kompatybilny wstecz – zakładamy wykrycie
        return True

    def _check_symmetry(self, result: dict) -> bool:
        """Sprawdza, czy środek obu źrenic leży „mniej więcej” na środku twarzy."""
        face_x, face_y, face_w, face_h = result["face_bbox"]
        face_cx = face_x + face_w / 2.0

        left = result["left_eye"]
        right = result["right_eye"]
        ixL, iyL = left["iris_center"]
        ixR, iyR = right["iris_center"]

        midpoint_x = (ixL + ixR) / 2.0
        diff = abs(midpoint_x - face_cx)

        return diff <= face_w * self.symmetry_tolerance

    def _check_horizontal_line(self, result: dict) -> bool:
        """Sprawdza, czy odcinek łączący GLOBALNE położenia źrenic jest prawie poziomy."""
        left = result["left_eye"]
        right = result["right_eye"]
        ixL, iyL = left["iris_center"]
        ixR, iyR = right["iris_center"]

        dx = ixR - ixL
        dy = iyR - iyL
        if dx == 0:
            return False

        angle = math.atan2(dy, dx)
        return abs(angle) <= self.angle_tolerance_rad

    def is_valid_frame(self, result: dict) -> bool:
        """Główna funkcja walidacji pojedynczej ramki."""
        if result is None:
            return False

        if result.get("status") != "ok":
            return False

        if result.get("face_bbox") is None:
            return False

        left = result.get("left_eye")
        right = result.get("right_eye")
        if left is None or right is None:
            return False

        # 1) czy obie źrenice wykryte
        if not self._pupils_detected(result):
            return False

        # 2) jakość oczu
        if not (self.eye_quality_ok(left["quality"]) and
                self.eye_quality_ok(right["quality"])):
            return False

        # 3) symetria względem osi pionowej twarzy
        if not self._check_symmetry(result):
            return False

        # 4) prawie poziomy odcinek łączący źrenice (globalne współrzędne)
        if not self._check_horizontal_line(result):
            return False

        return True


# ---------- Calibration ----------

class GazeCalibrator:
    def __init__(
        self,
        detector: EyeDetector,
        cap: cv2.VideoCapture,
        feature_extractor: GazeFeatureExtractor,
        validator: EyeFrameValidator,
        screen_size=(1280, 720),
        samples_per_point: int = 30,
        alpha = 1.0
    ):
        self.detector = detector
        self.cap = cap
        self.feature_extractor = feature_extractor
        self.validator = validator
        self.screen_w, self.screen_h = screen_size
        self.samples_per_point = samples_per_point
        self.alpha = alpha

        self.aborted = False

        # 9 calibration points (normalized coordinates 0–1)
        self.calib_points_norm = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
        ]

    def _draw_point_window(self, point_px):
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 255
        cv2.circle(img, point_px, 10, (0, 0, 255), -1)
        cv2.imshow("Gaze Screen", img)

    def _collect_features_for_point(self, target_px):
        """
        Collects feature samples for a single point on the screen
        and returns their average (feature vector).
        """
        collected = []
        print(f"  Collecting samples for point {target_px}...")

        while len(collected) < self.samples_per_point:
            ret, frame = self.cap.read()
            if not ret:
                print("  No frames from camera – aborting calibration.")
                self.aborted = True
                return None

            result = self.detector.detect(frame)

            # Camera display
            cam_disp = visualize_detection(frame, result)
            cv2.imshow("Camera", cam_disp)
            self._draw_point_window(target_px)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.aborted = True
                return None

            if not self.validator.is_valid_frame(result):
                continue

            features = self.feature_extractor(result)
            collected.append(features)

            if len(collected) % 5 == 0:
                print(f"  Collected {len(collected)}/{self.samples_per_point} good samples.")

        if len(collected) == 0:
            return None

        print(f"  Accepted frames for point {target_px}: {len(collected)}")
        return np.mean(np.stack(collected, axis=0), axis=0)


    def run(self):
        """
        Performs full calibration:
        - displays points one by one,
        - collects features,
        - trains two Ridge models (X and Y).
        Returns: (model_x, model_y)
        """
        time.sleep(10)

        X = []
        y_x = []
        y_y = []

        for idx, (nx, ny) in enumerate(self.calib_points_norm, start=1):
            if self.aborted:
                break

            px = int(nx * self.screen_w)
            py = int(ny * self.screen_h)
            target_px = (px, py)

            print(f"\nCalibration point {idx}/{len(self.calib_points_norm)}: {target_px}")

            feat_mean = self._collect_features_for_point(target_px)
            if self.aborted:
                break

            if feat_mean is None:
                print("  Note: no data for this point (no frame passed validation).")
                continue

            X.append(feat_mean)
            y_x.append(px)
            y_y.append(py)

        if self.aborted or len(X) == 0:
            print("\nCalibration failed - insufficient data.")
            return None, None

        X = np.asarray(X, dtype=np.float32)
        y_x = np.asarray(y_x, dtype=np.float32)
        y_y = np.asarray(y_y, dtype=np.float32)

        model_x = Ridge(alpha=self.alpha)
        model_y = Ridge(alpha=self.alpha)
        model_x.fit(X, y_x)
        model_y.fit(X, y_y)

        print("\nCalibration ended.")
        print(f"Number of used calibration points: {len(X)}")
        return model_x, model_y

        # for (nx, ny) in self.calib_points_norm:
        #     px = int(nx * self.screen_w)
        #     py = int(ny * self.screen_h)
        #     target_px = (px, py)
        #
        #     print(f"Look at the calibration point: {target_px}")
        #
        #     feat_mean = self._collect_features_for_point(target_px)
        #     if feat_mean is None:
        #         continue
        #
        #     X.append(feat_mean)
        #     y_x.append(px)
        #     y_y.append(py)
        #
        # X = np.asarray(X, dtype=np.float32)
        # y_x = np.asarray(y_x, dtype=np.float32)
        # y_y = np.asarray(y_y, dtype=np.float32)
        #
        # model_x = Ridge(alpha=1.0)
        # model_y = Ridge(alpha=1.0)
        # model_x.fit(X, y_x)
        # model_y.fit(X, y_y)
        #
        # print("Calibration completed.")
        # return model_x, model_y


# ---------- Real-time gaze tracking ----------

class RealTimeGazeTrackerWithValidation:
    """
    - uses trained Ridge models,
    - uses only frames that pass validation,
    - applies smoothing (running average over the last N predictions),
    - displays control points,
    - after clicking on a control point, calculates the error between its position

    and the current predicted gaze position,
    - after finishing (press 'q' key), prints the average error to the console.
    """

    def __init__(
        self,
        detector: EyeDetector,
        cap: cv2.VideoCapture,
        feature_extractor: GazeFeatureExtractor,
        validator: EyeFrameValidator,
        model_x: Ridge,
        model_y: Ridge,
        screen_size=(1280, 720),
        smoothing_window: int = 5,
        num_control_points: int = 8,
    ):
        self.detector = detector
        self.cap = cap
        self.feature_extractor = feature_extractor
        self.validator = validator
        self.model_x = model_x
        self.model_y = model_y
        self.screen_w, self.screen_h = screen_size
        self.history = deque(maxlen=smoothing_window)

        self.control_points = self._generate_control_points(num_control_points)
        self.current_gaze = (self.screen_w // 2, self.screen_h // 2)
        self.control_errors = []

    def _generate_control_points(self, num_points: int):
        points = []
        for _ in range(num_points):
            px = np.random.randint(0, self.screen_w)
            py = np.random.randint(0, self.screen_h)
            points.append((px, py))
        return points

    def _draw_screen(self):
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 255

        # Punkty kontrolne – niebieskie
        for (cx, cy) in self.control_points:
            cv2.circle(img, (int(cx), int(cy)), 8, (255, 0, 0), -1)

        # Aktualny punkt spojrzenia – czerwony
        gx, gy = self.current_gaze
        cv2.circle(img, (int(gx), int(gy)), 10, (0, 0, 255), -1)

        cv2.imshow("Gaze Screen", img)

    def _update_gaze(self, result: dict):
        if self.validator.is_valid_frame(result):
            features = self.feature_extractor(result).reshape(1, -1)
            gx = float(self.model_x.predict(features)[0])
            gy = float(self.model_y.predict(features)[0])
            self.history.append((gx, gy))

        if len(self.history) > 0:
            hx, hy = np.mean(np.array(self.history), axis=0)
            self.current_gaze = (int(hx), int(hy))
        else:
            self.current_gaze = (self.screen_w // 2, self.screen_h // 2)

    def on_mouse(self, event, x, y, flags, param):
        """Callback myszki dla fazy punktów kontrolnych."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # znajdź najbliższy punkt kontrolny
        click_pos = np.array([x, y], dtype=np.float32)
        cp_array = np.array(self.control_points, dtype=np.float32)
        dists = np.linalg.norm(cp_array - click_pos, axis=1)
        idx = int(np.argmin(dists))
        min_dist = float(dists[idx])

        # próg – klik musi być "w pobliżu" punktu kontrolnego
        if min_dist > 20.0:
            print("Click outside the control point.")
            return

        cp = self.control_points[idx]
        gx, gy = self.current_gaze
        err = math.sqrt((gx - cp[0]) ** 2 + (gy - cp[1]) ** 2)
        self.control_errors.append(err)
        print(f"Control point clicked {idx}: {cp}, "
              f"gaze={self.current_gaze}, error={err:.1f} px")

    def run(self):
        cv2.setMouseCallback("Gaze Screen", self.on_mouse)

        print("Click on the blue control points to measure the error.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No camera frames - tracking ended.")
                break

            result = self.detector.detect(frame)

            cam_display = visualize_detection(frame, result)
            cv2.imshow("Camera", cam_display)

            # aktualizacja punktu spojrzenia (tylko na podstawie dobrych klatek)
            self._update_gaze(result)

            # rysowanie punktów kontrolnych + śledzonego punktu
            self._draw_screen()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Po zakończeniu – podsumowanie dokładności
        if len(self.control_errors) > 0:
            errors = np.array(self.control_errors, dtype=np.float32)
            mean_err = float(np.mean(errors))
            std_err = float(np.std(errors))
            print("\n=== Model precision summary ===")
            print(f"Mean error (Euclidean) : {mean_err:.2f} px")
            print(f"Standard deviation : {std_err:.2f} px\n")
        else:
            print("\nNo registered control points.\n")

    # def _draw_gaze_point(self, point_px):
    #     img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 255
    #     cv2.circle(img, point_px, 10, (0, 0, 255), -1)
    #     cv2.imshow("Gaze Screen", img)
    #
    # def run(self):
    #     """
    #     Main loop:
    #     - eye and pupil detection,
    #     - feature calculation,
    #     - gaze point prediction,
    #     - smoothing and dot drawing.
    #     """
    #     while True:
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             break
    #
    #         result = self.detector.detect(frame)
    #
    #         cam_display = visualize_detection(frame, result)
    #         cv2.imshow("Camera", cam_display)
    #
    #         if result["status"] == "ok":
    #             left_q = result["left_eye"]["quality"]
    #             right_q = result["right_eye"]["quality"]
    #             if left_q["illumination_ok"] and right_q["illumination_ok"]:
    #                 features = self.feature_extractor(result).reshape(1, -1)
    #                 gx = float(self.model_x.predict(features)[0])
    #                 gy = float(self.model_y.predict(features)[0])
    #                 self.history.append((gx, gy))
    #
    #         if len(self.history) > 0:
    #             hx, hy = np.mean(np.array(self.history), axis=0)
    #             gaze_point = (int(hx), int(hy))
    #             self._draw_gaze_point(gaze_point)
    #         else:
    #             self._draw_gaze_point((self.screen_w // 2, self.screen_h // 2))
    #
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break


# ---------- Launching entire demo ----------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Reading the first frame to know the resolution
    ret, frame = cap.read()
    if not ret:
        print("No camera frames.")
        cap.release()
        return

    frame_h, frame_w = frame.shape[:2]
    detector = EyeDetector()
    feature_extractor = GazeFeatureExtractor(frame_w, frame_h)

    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    window_w = int(screen_w * 0.9)
    window_h = int(screen_h * 0.9)
    screen_size = (window_w, window_h)

    cv2.namedWindow("Gaze Screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Screen", window_w, window_h)
    cv2.moveWindow("Gaze Screen", 50, 0)

    blank = np.ones((screen_size[1], screen_size[0], 3), dtype=np.uint8) * 255
    cv2.imshow("Gaze Screen", blank)
    cv2.waitKey(1)

    ################################################################################
    # ----------------- Parametry do modyfikacji podczas testowania ----------------
    validator = EyeFrameValidator(
        symmetry_tolerance=0.3,   # z zakresu [0.15 - 0.35] - im większy tym
                                   # łagodniejszy próg walidacji - więcej ramek dla modelu
        angle_tolerance_deg=25.0,   # z zakresu [15.0 - 30.0] - im więcej tym
                                    # łagodniejszy próg
        min_std_intensity=8.0,    # z zakresu [5 - 15] - im większy tym bardziej
                                  # restrykcyjny próg - dobry przy dobrym oświetleniu
                                  # i kontraście
    )
    ################################################################################

    # ----------------- Parametr alpha modelu Ridge też można zmieniać -------------
    alpha = 0.5  # z zakresu [0.1, 10]
    ################################################################################

    # 1) Calibration
    calibrator = GazeCalibrator(detector, cap, feature_extractor, validator, screen_size, alpha=alpha)
    model_x, model_y = calibrator.run()
    if model_x is None or model_y is None:
        print("Calibration failed – program ended.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # 2) Gaze tracking + control points
    tracker = RealTimeGazeTrackerWithValidation(
        detector, cap, feature_extractor, validator, model_x, model_y, screen_size
    )
    tracker.run()

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.\n")


if __name__ == "__main__":
    main()



# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Could not open camera.")
#         return
#
#     # Reading the first frame to know the resolution
#     ret, frame = cap.read()
#     if not ret:
#         print("No camera frames.")
#         return
#
#     frame_h, frame_w = frame.shape[:2]
#     detector = EyeDetector()
#     feature_extractor = GazeFeatureExtractor(frame_w, frame_h)
#
#     # screen_size = (1280, 720)
#
#     user32 = ctypes.windll.user32
#     screen_w = user32.GetSystemMetrics(0)
#     screen_h = user32.GetSystemMetrics(1)
#
#     window_w = int(screen_w * 0.9)
#     window_h = int(screen_h * 0.9)
#     screen_size = (window_w, window_h)
#
#     cv2.namedWindow("Gaze Screen", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Gaze Screen", window_w, window_h)
#     cv2.moveWindow("Gaze Screen", 50, 0)
#
#     blank = np.ones((screen_size[1], screen_size[0], 3), dtype=np.uint8) * 255
#     cv2.imshow("Gaze Screen", blank)
#     cv2.waitKey(1)
#
#     # 1) Calibration
#     calibrator = GazeCalibrator(detector, cap, feature_extractor, screen_size)
#     model_x, model_y = calibrator.run()
#
#     # 2) Gaze tracking
#     tracker = RealTimeGazeTracker(
#         detector, cap, feature_extractor, model_x, model_y, screen_size
#     )
#     tracker.run()
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
