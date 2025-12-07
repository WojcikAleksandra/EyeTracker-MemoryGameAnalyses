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


# ---------- Ekstrakcja cech - wartości pikseli z prostokątów oczu ----------

class GazeFeatureExtractor:
    """
    Tworzy wektor cech na podstawie wyniku detekcji twarzy i oczu z EyeDetectora.
    Pipeline:
    - pobierz eye_patch z lewej i prawej strony,
    - przekonwertuj do skali szarości,
    - przeskaluj każdy patch do patch_height x patch_width (domyślnie 6x10),
    - znormalizuj do [0,1],
    - spłaszcz i sklej: [left_patch, right_patch] -> wektor cech.
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

        # Ekstrakcja regionu oka
        eye_region = frame[y_global:y_global + eye_h, x_global:x_global + eye_w]

        # Konwersja do skali szarości
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region

        # Skalowanie do patch_height x patch_width
        patch_resized = cv2.resize(eye_gray, (self.patch_width, self.patch_height), interpolation=cv2.INTER_AREA)

        # Normalizacja do [0, 1]
        patch_norm = patch_resized.astype(np.float32) / 255.0

        return patch_norm.flatten()

    def __call__(self, frame: np.ndarray, result: dict) -> np.ndarray:
        left = result["left_eye"]
        right = result["right_eye"]

        left_feats = self._extract_eye_features(frame, left["bbox"])
        right_feats = self._extract_eye_features(frame, right["bbox"])

        features = np.concatenate([left_feats, right_feats], axis=0)
        return features.astype(np.float32)


# # ---------- Walidacja jakości pojedynczej klatki - dla prostokątów oczu ----------

class EyeFrameValidator:
    """
    Walidacja pojedynczej klatki pod kątem geometrii prostokątów oczu.

    Klatka jest uznana za poprawną, jeśli:
    - wykryto twarz (face_bbox != None),
    - wykryto jednocześnie lewe i prawe oko (left_eye i right_eye != None).

    Warunki do dodania później:
    - środki prostokątów oczu leżą po dwóch różnych stronach osi pionowej
      przechodzącej przez środek prostokąta twarzy,
    - różnica współrzędnych Y środków oczu nie jest zbyt duża,
    - dla każdego oka szerokość jest (w przybliżeniu) większa niż wysokość,
    - oba prostokąty oczu mają zbliżony rozmiar.
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

        # fx, fy, fw, fh = face_bbox
        # face_cx = fx + fw / 2.0
        #
        # lx, ly, lw, lh = left["bbox"]
        # rx, ry, rw, rh = right["bbox"]
        #
        # # Środki prostokątów oczu
        # lcx = lx + lw / 2.0
        # lcy = ly + lh / 2.0
        # rcx = rx + rw / 2.0
        # rcy = ry + rh / 2.0
        #
        # ldx = lcx - face_cx
        # rdx = rcx - face_cx
        #
        # # Środki muszą mieć różne znaki (po różnych stronach środkowej osi pionowej twarzy)
        # if ldx == 0 or rdx == 0:
        #     return False
        # if ldx * rdx > 0:
        #     # Ten sam znak => po tej samej stronie
        #     return False
        #
        # # Nieduża różnica w Y środków prostokątów oczu
        # center_y_diff = abs(lcy - rcy)
        # max_center_y_diff = self.max_center_y_diff_rel * fh
        # if center_y_diff > max_center_y_diff:
        #     return False
        #
        # # Szerokość większa niż wysokość eye patcha
        # # w >= (1 - aspect_tolerance) * h
        # def _aspect_ok(w, h) -> bool:
        #     if h <= 0:
        #         return False
        #     return w >= (1.0 - self.aspect_tolerance) * h
        #
        # if not _aspect_ok(lw, lh):
        #     return False
        # if not _aspect_ok(rw, rh):
        #     return False
        #
        # area_l = lw * lh
        # area_r = rw * rh
        # if area_l <= 0 or area_r <= 0:
        #     return False
        #
        # area_ratio = min(area_l, area_r) / max(area_l, area_r)
        # if area_ratio < self.min_area_ratio:
        #     return False
        #
        # # Podobieństwo szerokości i wysokości eye patchy
        # width_ratio = min(lw, rw) / max(lw, rw)
        # height_ratio = min(lh, rh) / max(lh, rh)
        # if width_ratio < self.min_size_ratio or height_ratio < self.min_size_ratio:
        #     return False

        return True


def create_regressor(model_name: str, alpha: float = 1.0, c: int = 10,
                     epsilon: float = 0.1, gamma: str = "scale",
                     n_estimators: int = 300, learning_rate: float = 0.05,
                     max_depth: int = 2):
    """
    Zwraca skonfigurowany model regresyjny na podstawie nazwy.
    model_name:
        - "ridge"
        - "random_forest"
        - "gbrt"
        - "svr"
        - "mlp"
    """
    model_name = model_name.lower()
    if model_name == "ridge":
        # klasyczna regresja z regularyzacją L2
        return Ridge(alpha=alpha)
    elif model_name == "random_forest":
        # prosty las losowy
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "gbrt":
        # gradient boosting
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    elif model_name == "svr":
        # SVR z jądrem RBF
        return SVR(
            kernel="rbf",
            C=c,
            epsilon=epsilon,
            gamma=gamma
        )
    elif model_name == "mlp":
        # mała sieć MLP
        return MLPRegressor(
            hidden_layer_sizes=(32, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
        )
    else:
        print(f"Nieznany typ modelu '{model_name}', używam Ridge.")
        return Ridge(alpha=alpha)


class GazeEngine:
    """
    Silnik eye-trackingu do integracji z zewnętrznym GUI.

    Założenia:
    - nie tworzy okien OpenCV,
    - zarządza kamerą, detekcją oczu, walidacją i ekstrakcją cech,
    - pozwala z zewnątrz:
        * zbierać próbki kalibracyjne (target_x, target_y),
        * trenować modele regresji,
        * w czasie rzeczywistym przewidywać punkt spojrzenia (gx, gy)
          w układzie współrzędnych screen_size (np. okna gry).
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
        screen_size: (width, height) – rozmiar obszaru, w którym pracuje gra/GUI.
        model_type: "ridge", "gbrt", "svr", "mlp", "random_forest"
        patch_height, patch_width: rozmiar patchy oczu
        min_samples: minimalna liczba próbek do wytrenowania modelu
        smoothing_window: liczba ostatnich predykcji do uśredniania
        pozostałe - parametry modeli.
        """

        self.screen_w, self.screen_h = screen_size
        self.min_samples = min_samples

        # --- Kamera ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Nie można otworzyć kamery.")

        # --- Detekcja, walidacja, cechy ---
        self.detector = EyeDetector()
        self.validator = EyeFrameValidator()
        self.feature_extractor = GazeFeatureExtractor(
            patch_height=patch_height,
            patch_width=patch_width,
        )

        # --- Modele regresji (puste, będą trenowane w fit_models) ---
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

        # --- Bufory kalibracyjne ---
        self.calib_X = []   # lista wektorów cech
        self.calib_yx = []  # lista docelowych X (piksele w screen_size)
        self.calib_yy = []  # lista docelowych Y

        # --- Stan kalibracji / śledzenia ---
        self._is_calibrated = False
        self.history = deque(maxlen=smoothing_window)
        self.last_gaze = (self.screen_w // 2, self.screen_h // 2)

    # ======================================================================
    # --- NISKI POZIOM: klatka z kamery + detekcja ----------------------------------
    # ======================================================================

    def capture_and_detect(self):
        """
        Pobiera jedną klatkę z kamery i robi detekcję oczu.

        Zwraca:
            (frame, result) albo (None, None) przy błędzie.
        """
        if self.cap is None:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        result = self.detector.detect(frame)
        return frame, result

    # ======================================================================
    # --- KAMERA / CECHY ---------------------------------------------------
    # ======================================================================

    def grab_features_if_valid(self, frame=None, result=None):
        """
        Używa podanej klatki i wyniku detekcji, robi walidację
        i ekstrakcję cech. Zwraca:
            - np.ndarray 1D (wektor cech) jeśli klatka jest poprawna,
            - None, jeśli nie udało się pobrać poprawnych cech.
        """
        if frame is None or result is None:
            frame, result = self.capture_and_detect()
            if frame is None:
                return None

        if not self.validator.is_valid_frame(result):
            return None

        feats = self.feature_extractor(frame, result)
        return feats  # 1D np.ndarray (float32)


    # ======================================================================
    # --- KALIBRACJA -------------------------------------------------------
    # ======================================================================

    def start_calibration(self):
        """
        Czyści stare próbki kalibracyjne i resetuje stan kalibracji.
        Wywoływane przed rozpoczęciem wyświetlania punktów kalibracyjnych w GUI.
        """
        self.calib_X = []
        self.calib_yx = []
        self.calib_yy = []
        self._is_calibrated = False
        self.history.clear()

    def add_calibration_sample(self, target_x, target_y, frame=None, result=None):
        """
        Dodaje jedną próbkę kalibracyjną dla zadanego punktu docelowego
        (target_x, target_y) w układzie współrzędnych screen_size.

        Działanie:
        - próbuje wyciągnąć cechy z poprawnej (zwalidowanej) klatki,
        - jeśli się uda, dodaje (features, target_x, target_y) do buforów.

        Zwraca:
            True  – jeśli próbka została dodana,
            False – jeśli nie udało się pobrać poprawnych cech.
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
        Uczy model_x i model_y na zebranych próbkach kalibracyjnych.

        Warunek trenowania:
        - liczba próbek >= self.min_samples

        Po udanym trenowaniu ustawia _is_calibrated = True.

        Zwraca:
            True  – jeśli modele zostały wytrenowane,
            False – jeśli zbyt mało danych lub brak próbek.
        """
        n_samples = len(self.calib_X)
        if n_samples == 0:
            print("Brak danych kalibracyjnych – nie można wytrenować modelu.")
            self._is_calibrated = False
            return False

        if n_samples < self.min_samples:
            print(
                f"Za mało danych kalibracyjnych ({n_samples}) – "
                f"wymagane min_samples={self.min_samples}."
            )
            self._is_calibrated = False
            return False

        X = np.asarray(self.calib_X, dtype=np.float32)
        y_x = np.asarray(self.calib_yx, dtype=np.float32)
        y_y = np.asarray(self.calib_yy, dtype=np.float32)

        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        self._is_calibrated = True
        print("\nKalibracja zakończona.")
        print(f"Liczba próbek treningowych: {n_samples}")
        return True

    def is_calibrated(self) -> bool:
        """Zwraca True, jeśli modele zostały wytrenowane i można śledzić spojrzenie."""
        return self._is_calibrated

    # ======================================================================
    # --- ŚLEDZENIE SPOJRZENIA ---------------------------------------------
    # ======================================================================

    def predict_gaze(self, frame=None, result=None):
        """
        Wyciąga cechy z klatki, przepuszcza przez model_x / model_y i zwraca
        (gx, gy) w układzie screen_size.

        Stosuje wygładzanie po ostatnich N predykcjach (N = smoothing_window).

        Zwraca:
            (gx, gy) – tuple[int, int] jeśli klatka poprawna i modele wytrenowane,
            None     – jeśli brak kalibracji lub brak poprawnej klatki.
        """
        if not self._is_calibrated:
            return None

        feats = self.grab_features_if_valid(frame, result)
        if feats is None:
            # brak nowej poprawnej klatki – zwracamy ostatnie znane spojrzenie
            if len(self.history) > 0:
                hx, hy = np.mean(np.array(self.history, dtype=np.float32), axis=0)
                self.last_gaze = (int(hx), int(hy))
                return self.last_gaze
            return None

        feats_2d = feats.reshape(1, -1)
        gx = float(self.model_x.predict(feats_2d)[0])
        gy = float(self.model_y.predict(feats_2d)[0])

        self.history.append((gx, gy))

        # uśrednianie współrzędnych z historii
        hx, hy = np.mean(np.array(self.history, dtype=np.float32), axis=0)
        self.last_gaze = (int(hx), int(hy))
        return self.last_gaze

    # ======================================================================
    # --- CZYSZCZENIE ------------------------------------------------------
    # ======================================================================

    def close(self):
        """
        Zwalnia kamerę.
        """
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None