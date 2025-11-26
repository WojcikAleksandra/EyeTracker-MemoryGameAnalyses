import cv2
import numpy as np
from collections import deque
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import time
import ctypes
import math

from eye_detector import EyeDetector


# ---------- Wizualizacja detekcji w oknie kamery ----------

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


# ---------- Ekstrakcja cech ----------

class GazeFeatureExtractor:
    """
    Tworzy wektor cech na podstawie wyniku EyeDetectora.
    Cechy:
    [xL, yL, xR, yR, Δd, x̄, ȳ, θ, x_face, y_face]
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def __call__(self, result: dict) -> np.ndarray:
        left = result["left_eye"]
        right = result["right_eye"]
        face_x, face_y, face_w, face_h = result["face_bbox"]

        # Znormalizowane współrzędne źrenic w prostokątach oczu
        xL, yL = left["iris_center_rel"]
        xR, yR = right["iris_center_rel"]

        # # Odległość między źrenicami (w znormalizowanej przestrzeni oczu)
        # dx = xR - xL
        # dy = yR - yL
        # dist = float(np.sqrt(dx * dx + dy * dy))
        #
        # # Średnia pozycja źrenic (w znormalizowanej przestrzeni oczu)
        # x_mean = (xL + xR) / 2.0
        # y_mean = (yL + yR) / 2.0

        # Kąt linii łączącej źrenice
        ixL, iyL = left["iris_center"]
        ixR, iyR = right["iris_center"]
        theta = float(np.arctan2(iyR - iyL, ixR - ixL))

        # Pozycja środka twarzy w obrazie kamery – znormalizowana do [0,1]
        face_cx = face_x + face_w / 2.0
        face_cy = face_y + face_h / 2.0
        x_face = face_cx / self.frame_width
        y_face = face_cy / self.frame_height

        features = np.array(
            [xL, yL, xR, yR, theta, x_face, y_face],
            dtype=np.float32,
        )
        return features


# ---------- Walidacja jakości pojedynczej klatki ----------

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
        min_std_intensity: float = 10.0,       # minimalny próg dla std_intensity
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

        # Obsługa opcjonalnej flagi iris_detected
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
        diff = abs(midpoint_x - face_cx) / face_w

        return diff <= self.symmetry_tolerance

    def _check_horizontal_line(self, result: dict) -> bool:
        """Sprawdza, czy odcinek łączący globalne położenia źrenic jest prawie poziomy."""
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

        # 1) czy obie źrenice wykryte
        if not self._pupils_detected(result):
            return False

        # # 2) jakość oczu
        # if not (self.eye_quality_ok(left["quality"]) and
        #         self.eye_quality_ok(right["quality"])):
        #     return False

        # 3) symetria względem osi pionowej twarzy
        if not self._check_symmetry(result):
            return False

        # 4) prawie poziomy odcinek łączący źrenice (globalne współrzędne)
        if not self._check_horizontal_line(result):
            return False

        return True


# ---------- Kalibracja z klikalnymi, losowymi punktami ----------

def _mouse_callback_calibration(event, x, y, flags, param):
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state["clicked"] = True
        state["click_time_ms"] = int(time.time() * 1000)
        state["click_pos"] = (x, y)   # zapisz pozycję kliknięcia


class RandomClickCalibrator:
    """
    Kalibracja:
    - 12 punktów w losowych miejscach (rozklad jednostajny po oknie),
    - użytkownik klika w każdy punkt,
    - zbierane są klatki z 500 ms przed kliknięciem (łącznie z momentem kliknięcia),
    - z nich wybierane są tylko te, które przejdą walidację,
    - dla zaakceptowanych klatek liczone są cechy i każda próbka trafia do modelu,
    - na końcu trenowane są dwa modele regresyjne (dla X i Y),
      przekazane z zewnątrz (wybór modelu w main()).
    """

    def __init__(
        self,
        detector: EyeDetector,
        cap: cv2.VideoCapture,
        feature_extractor: GazeFeatureExtractor,
        validator: EyeFrameValidator,
        screen_size=(1280, 720),
        window_ms: int = 500,
        num_points: int = 12,
        min_samples: int=60,
        model_x=None,
        model_y=None,
    ):
        self.detector = detector
        self.cap = cap
        self.feature_extractor = feature_extractor
        self.validator = validator
        self.screen_w, self.screen_h = screen_size
        self.window_ms = window_ms
        self.num_points = num_points
        self.min_samples = min_samples
        self.model_x = model_x
        self.model_y = model_y

    def _generate_random_points(self):
        points = []
        for _ in range(self.num_points):
            px = np.random.randint(0, self.screen_w)
            py = np.random.randint(0, self.screen_h)
            points.append((px, py))
        return points

    def _draw_calibration_point(self, point_px):
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 255
        cv2.circle(img, point_px, 10, (0, 0, 255), -1)  # czerwony punkt kalibracyjny
        cv2.imshow("Gaze Screen", img)

    def run(self):
        """
        Zwraca: (model_x, model_y)
        """
        points = self._generate_random_points()
        print(f"Rozpoczynam kalibrację z {len(points)} punktami.")

        X = []
        y_x = []
        y_y = []

        for idx, target_px in enumerate(points, start=1):
            print(f"\nPunkt kalibracyjny {idx}/{len(points)}: {target_px}")
            print("Patrz na punkt i kliknij, gdy jesteś gotowy.")

            # Stan dla callbacku
            click_state = {"clicked": False, "click_time_ms": 0}
            cv2.setMouseCallback("Gaze Screen", _mouse_callback_calibration, click_state)

            # Bufor ramek (czas, wynik detekcji)
            frame_buffer = []

            # Pętla do momentu kliknięcia
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                timestamp_ms = int(time.time() * 1000)
                result = self.detector.detect(frame)

                frame_buffer.append((timestamp_ms, result))

                # Wyświetlanie kamery i punktu
                cam_disp = visualize_detection(frame, result)
                cv2.imshow("Camera", cam_disp)
                self._draw_calibration_point(target_px)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Przerwano kalibrację klawiszem 'q'.")
                    return None, None

                if click_state["clicked"]:
                    cx, cy = click_state["click_pos"]
                    tx, ty = target_px
                    dist = np.hypot(cx - tx, cy - ty)

                    if dist <= 20:  # próg odległości od punktu kalibracyjnego
                        # kliknięcie uznajemy za poprawne – wychodzimy z pętli
                        click_time_ms = click_state["click_time_ms"]
                        break
                    else:
                        # kliknięcie w złe miejsce – ignorujemy i czekamy dalej
                        print("Kliknięto poza punktem kalibracyjnym – spróbuj jeszcze raz.")
                        click_state["clicked"] = False

            window_start = click_time_ms - self.window_ms

            # Wybór ramek z okna czasowego [click-500 ms, click]
            window_results = [
                r for (ts, r) in frame_buffer
                if window_start <= ts <= click_time_ms
            ]

            print(f"  Zebrane klatki w oknie {self.window_ms} ms: {len(window_results)}")

            # Walidacja + cechy
            valid_features = []
            for r in window_results:
                if self.validator.is_valid_frame(r):
                    feats = self.feature_extractor(r)
                    valid_features.append(feats)

            if len(valid_features) == 0:
                print("  Uwaga: żadna klatka nie przeszła walidacji dla tego punktu.")
                continue

            # feats_mean = np.mean(np.stack(valid_features, axis=0), axis=0)
            # X.append(feats_mean)
            # y_x.append(float(target_px[0]))
            # y_y.append(float(target_px[1]))

            for feats in valid_features:
                X.append(feats)
                y_x.append(float(target_px[0]))
                y_y.append(float(target_px[1]))

            print(f"  Akceptowanych klatek: {len(valid_features)}")

        if len(X) < self.min_samples:
            print("Za mało danych kalibracyjnych – nie można wytrenować modelu.")
            return None, None

        if len(X) == 0:
            print("Brak danych kalibracyjnych – nie można wytrenować modelu.")
            return None, None

        X = np.asarray(X, dtype=np.float32)
        y_x = np.asarray(y_x, dtype=np.float32)
        y_y = np.asarray(y_y, dtype=np.float32)

        # Trenowanie przekazanych modeli regresyjnych
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        print("\nKalibracja zakończona.")
        print(f"\nŁączna liczba próbek treningowych: {len(X)}")
        return self.model_x, self.model_y


# ---------- Śledzenie spojrzenia + punkty kontrolne ----------

class RealTimeGazeTrackerWithValidation:
    """
    - używa wytrenowanych modeli regresyjnych,
    - wykorzystuje tylko klatki, które przejdą walidację,
    - stosuje wygładzanie (średnia krocząca po ostatnich N predykcjach),
    - wyświetla punkty kontrolne,
    - po kliknięciu w punkt kontrolny liczy błąd między jego pozycją a aktualną
      przewidywaną pozycją spojrzenia,
    - po zakończeniu (klawisz 'q') wypisuje średni błąd.
    """

    def __init__(
        self,
        detector: EyeDetector,
        cap: cv2.VideoCapture,
        feature_extractor: GazeFeatureExtractor,
        validator: EyeFrameValidator,
        model_x,
        model_y,
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
        self.control_errors = []  # lista odległości (piksele)

    def _generate_control_points(self, num_points: int):
        points = []
        for _ in range(num_points):
            px = np.random.randint(0, self.screen_w)
            py = np.random.randint(0, self.screen_h)
            points.append((px, py))
        return points

    def _draw_screen(self):
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 255

        # Punkty kontrolne – niech będą niebieskie
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
            print("Kliknięcie poza punktem kontrolnym.")
            return

        cp = self.control_points[idx]
        gx, gy = self.current_gaze
        err = math.sqrt((gx - cp[0]) ** 2 + (gy - cp[1]) ** 2)
        self.control_errors.append(err)
        print(f"Kliknięty punkt kontrolny {idx}: {cp}, "
              f"gaze={self.current_gaze}, błąd={err:.1f} px")

    def run(self):
        cv2.setMouseCallback("Gaze Screen", self.on_mouse)

        print("\nRozpoczynam śledzenie spojrzenia.")
        print("Klikaj na niebieskie punkty kontrolne, aby zmierzyć błąd.")
        print("Naciśnij 'q', aby zakończyć.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
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
            print("\n=== Podsumowanie dokładności modelu ===")
            print(f"Liczba klikniętych punktów kontrolnych: {len(errors)}")
            print(f"Średni błąd (Euclidean) : {mean_err:.2f} px")
            print(f"Odchylenie standardowe : {std_err:.2f} px")
        else:
            print("\nBrak zarejestrowanych punktów kontrolnych – brak metryki błędu.")


def create_regressor(model_name: str, alpha: float = 1.0):
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
            random_state=42,
        )
    elif model_name == "svr":
        # SVR z jądrem RBF
        return SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
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


# ---------- Funkcja main ----------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie można otworzyć kamery.")
        return

    # Pierwsza klatka do poznania rozdzielczości
    ret, frame = cap.read()
    if not ret:
        print("Brak klatek z kamery.")
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

    ################################################################################
    # ----------------- Parametry do modyfikacji podczas testowania ----------------
    validator = EyeFrameValidator(
        symmetry_tolerance=0.4,    # z zakresu [0.15 - 0.35] - im większy tym
                                   # łagodniejszy próg walidacji - więcej ramek dla modelu
        angle_tolerance_deg=40.0,   # z zakresu [15.0 - 30.0] - im więcej tym
                                    # łagodniejszy próg
        min_std_intensity=5.0,   # z zakresu [5 - 15] - im większy tym bardziej
                                  # restrykcyjny próg - dobry przy dobrym oświetleniu
                                  # i kontraście
    )
    ################################################################################

    # ------------------ Wybór modelu ----------------------------------
    # "ridge", "random_forest", "gbrt", "svr", "mlp"
    model_type = "ridge"
    alpha = 1.0   # z zakresu [0.1, 10] (tylko dla modelu Ridge)

    model_x = create_regressor(model_type, alpha=alpha)
    model_y = create_regressor(model_type, alpha=alpha)
    ################################################################################

    # 1) Kalibracja
    calibrator = RandomClickCalibrator(
        detector, cap, feature_extractor, validator,
        screen_size=screen_size,
        num_points=12,
        min_samples=40,
        model_x=model_x,
        model_y=model_y
    )
    model_x_trained, model_y_trained = calibrator.run()
    if model_x_trained is None or model_y_trained is None:
        print("Kalibracja nieudana – koniec programu.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # 2) Śledzenie spojrzenia + punkty kontrolne
    tracker = RealTimeGazeTrackerWithValidation(
        detector, cap, feature_extractor, validator,
        model_x_trained, model_y_trained, screen_size=screen_size
    )
    tracker.run()

    cap.release()
    cv2.destroyAllWindows()
    print("Program zakończony.\n")

if __name__ == "__main__":
    main()
