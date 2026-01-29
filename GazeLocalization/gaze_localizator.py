import cv2
import numpy as np
from collections import deque
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import sys

sys.path.append("..")
sys.path.append("../eye-detection-final")

from eye_detector import EyeDetector


# ---------- Feature extraction - pixel values from eye patches ----------

class GazeFeatureExtractor:
    """
    Constructs a feature vector based on face and eye detection results from EyeDetector.
    Pipeline:
    - extract eye patches from the left and right eye regions,
    - convert patches to grayscale,
    - resize each patch to patch_height x patch_width (default: 6x10),
    - normalize pixel values to [0,1],
    - flatten and concatenate: [left_patch, right_patch] -> feature vector.
    """

    def __init__(
        self,
        patch_height: int = 6,
        patch_width: int = 10,
    ):
        self.patch_height = patch_height
        self.patch_width = patch_width

    def _extract_eye_features(self, frame: np.ndarray, eye_bbox) -> np.ndarray:
        x_global, y_global, eye_w, eye_h = eye_bbox

        eye_region = frame[y_global:y_global + eye_h, x_global:x_global + eye_w]
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        patch_resized = cv2.resize(eye_gray, (self.patch_width, self.patch_height), interpolation=cv2.INTER_AREA)
        patch_norm = patch_resized.astype(np.float32) / 255.0

        return patch_norm.flatten()

    def __call__(self, frame: np.ndarray, result: dict) -> np.ndarray:
        left = result["left_eye"]
        right = result["right_eye"]

        left_feats = self._extract_eye_features(frame, left["bbox"])
        right_feats = self._extract_eye_features(frame, right["bbox"])

        features = np.concatenate([left_feats, right_feats], axis=0)
        return features.astype(np.float32)


# ---------- Single camera frame quality validation - for eye patches ----------

class EyeFrameValidator:
    """
    Validates a single frame for eye patches detection.

    A frame is considered valid if:
    - a face has been detected (face_bbox is not None),
    - both the left and right eye have been detected (left_eye and right_eye are not None).
    """

    def __init__(self):
        self.max_center_y_diff_rel = 0.25
        self.aspect_tolerance = 0.2
        self.min_area_ratio = 0.4
        self.min_size_ratio = 0.4
        self.min_eye_offset_x_rel = 0.05

    def is_valid_frame(self, result: dict) -> bool:
        if result is None:
            return False

        face_bbox = result.get("face_bbox")
        if face_bbox is None:
            return False

        left = result.get("left_eye")
        right = result.get("right_eye")
        if left is None or right is None:
            return False

        return True


def create_regressor(model_name: str, alpha: float = 1.0, c: int = 10,
                     epsilon: float = 0.1, gamma: str = "scale",
                     n_estimators: int = 300, learning_rate: float = 0.05,
                     max_depth: int = 2):
    """
    Returns a configured regression model based on the provided model name.

    model_name:
        - "ridge"
        - "random_forest"
        - "gbrt"
        - "svr"
        - "mlp"
    """

    model_name = model_name.lower()
    if model_name == "ridge":
        return Ridge(alpha=alpha)
    elif model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "gbrt":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    elif model_name == "svr":
        return SVR(
            kernel="rbf",
            C=c,
            epsilon=epsilon,
            gamma=gamma
        )
    elif model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(32, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
        )
    else:
        print(f"Unknown model type '{model_name}', using Ridge.")
        return Ridge(alpha=alpha)


class GazeEngine:
    """
    Eye-tracking engine for integration with external GUI.

    Assumptions:
    - does not create any OpenCV windows,
    - manages camera handling, eye detection, frame validation, and feature extraction,
    - enables:
        * collection of calibration samples (target_x, target_y),
        * regression model training,
        * real-time prediction of the gaze point (gx, gy)
        in the screen_size coordinate system (e.g. a game window).
    """

    def __init__(
        self,
        screen_size,
        model_type: str = "ridge",
        patch_height: int = 8,
        patch_width: int = 9,
        min_samples: int = 60,
        smoothing_window: int = 5,
        alpha: float = 1.0,
        c: int = 10,
        epsilon: float = 0.1,
        gamma: str = "scale",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 2,
    ):
        """
        screen_size: (width, height) – size of the area in which the game/GUI operates.
        model_type: "ridge", "gbrt", "svr", "mlp", "random_forest"
        patch_height, patch_width: size of the eye patches.
        min_samples: minimum number of samples required to train the model.
        smoothing_window: number of last predictions to average.
        remaining: model hyperparameters.
        """

        self.screen_w, self.screen_h = screen_size
        self.min_samples = min_samples

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open the camera.")

        self.detector = EyeDetector()
        self.validator = EyeFrameValidator()
        self.feature_extractor = GazeFeatureExtractor(
            patch_height=patch_height,
            patch_width=patch_width,
        )

        self.model_type = model_type
        self.model_x = create_regressor(
            model_type,
            alpha=alpha,
            c=c,
            epsilon=epsilon,
            gamma=gamma,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        self.model_y = create_regressor(
            model_type,
            alpha=alpha,
            c=c,
            epsilon=epsilon,
            gamma=gamma,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )

        # --- Calibration buffers ---
        self.calib_X = []   # feature vector list
        self.calib_yx = []  # target X list (pixels in screen_size)
        self.calib_yy = []  # target Y list

        # --- Calibration / tracking state ---
        self._is_calibrated = False
        self.history = deque(maxlen=smoothing_window)
        self.last_gaze = (self.screen_w // 2, self.screen_h // 2)


    # ======================================================================
    # --- LOW-LEVEL: camera frame + detection ----------------------------------
    # ======================================================================

    def capture_and_detect(self):
        """
        Captures a single frame from the camera and performs eye detection.

        Returns:
            (frame, result) or (None, None) in case of an error.
        """
        if self.cap is None:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        result = self.detector.detect(frame)
        return frame, result


    # ======================================================================
    # --- CAMERA / FEATURES ---------------------------------------------------
    # ======================================================================

    def grab_features_if_valid(self, frame=None, result=None):
        """
        Uses the provided frame and detection result to perform validation
        and feature extraction. Returns:
            - a 1D np.ndarray (feature vector) if the frame is valid,
            - None if valid features could not be extracted.
        """
        if frame is None or result is None:
            frame, result = self.capture_and_detect()
            if frame is None:
                return None

        if not self.validator.is_valid_frame(result):
            return None

        feats = self.feature_extractor(frame, result)
        return feats


    # ======================================================================
    # --- CALIBRATION -------------------------------------------------------
    # ======================================================================

    def start_calibration(self):
        """
        Clears old calibration samples and resets the calibration state.
        Intended to be called before displaying calibration points in the GUI.
        """
        self.calib_X = []
        self.calib_yx = []
        self.calib_yy = []
        self._is_calibrated = False
        self.history.clear()

    def add_calibration_sample(self, target_x, target_y, frame=None, result=None):
        """
        Adds a single calibration sample for the given target point
        (target_x, target_y) in the screen_size coordinate system.

        Operation:
        - attempts to extract features from a valid (validated) frame,
        - if successful, appends (features, target_x, target_y) to the buffers.

        Returns:
            True  – if the sample was successfully added,
            False – if valid features could not be extracted.
        """
        feats = self.grab_features_if_valid(frame, result)
        if feats is None:
            return False

        self.calib_X.append(feats)
        self.calib_yx.append(float(target_x))
        self.calib_yy.append(float(target_y))
        return True

    def fit_models(self):
        """
        Trains model_x and model_y using the collected calibration samples.

        Training condition:
        - number of samples >= self.min_samples

        After successful training, sets _is_calibrated to True.

        Returns:
            True  – if the models were successfully trained,
            False – if there is not enough data or no samples.
        """
        n_samples = len(self.calib_X)
        if n_samples == 0:
            print("No calibration data – cannot train the model.")
            self._is_calibrated = False
            return False

        if n_samples < self.min_samples:
            print(
                f"Not enough calibration data ({n_samples}) – "
                f"min_samples={self.min_samples} required."
            )
            self._is_calibrated = False
            return False

        X = np.asarray(self.calib_X, dtype=np.float32)
        y_x = np.asarray(self.calib_yx, dtype=np.float32)
        y_y = np.asarray(self.calib_yy, dtype=np.float32)

        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        self._is_calibrated = True
        print("\nCalibration completed.")
        print(f"Number of training samples: {n_samples}")
        return True

    def is_calibrated(self) -> bool:
        """
        Returns True if the models have been trained and gaze tracking
        can be performed.
        """
        return self._is_calibrated


    # ======================================================================
    # --- GAZE TRACKING ---------------------------------------------
    # ======================================================================

    def predict_gaze(self, frame=None, result=None):
        """
        Extracts features from the current frame, feeds them into model_x / model_y,
        and returns the gaze point (gx, gy) in the screen_size coordinate system.

        Applies smoothing over the last N predictions (N = smoothing_window).

        Returns:
            (gx, gy) – tuple[int, int] if the frame is valid and the models are trained,
            None     – if the system is not calibrated or no valid frame is available.
        """
        if not self._is_calibrated:
            return None

        feats = self.grab_features_if_valid(frame, result)
        if feats is None:
            # no new valid frame - return last known gaze
            if len(self.history) > 0:
                hx, hy = np.mean(np.array(self.history, dtype=np.float32), axis=0)
                self.last_gaze = (int(hx), int(hy))
                return self.last_gaze
            return None

        feats_2d = feats.reshape(1, -1)
        gx = float(self.model_x.predict(feats_2d)[0])
        gy = float(self.model_y.predict(feats_2d)[0])

        self.history.append((gx, gy))

        # averaging coordinates from history
        hx, hy = np.mean(np.array(self.history, dtype=np.float32), axis=0)
        self.last_gaze = (int(hx), int(hy))
        return self.last_gaze

    # ======================================================================
    # --- CLEANING ------------------------------------------------------
    # ======================================================================

    def close(self):
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None