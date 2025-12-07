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

    # Eyes + iris + eye patches
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


# ---------- Kalibracja z klikalnymi, losowymi punktami ----------

def _mouse_callback_calibration(event, x, y, flags, param):
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state["clicked"] = True
        state["click_time_ms"] = int(time.time() * 1000)
        state["click_pos"] = (x, y)   # zapisz pozycję kliknięcia


class ClickCalibrator:
    """
    Kalibracja:
    - 12 punktów w losowych miejscach (rozklad jednostajny po oknie),
    - użytkownik klika w każdy punkt,
    - zbierane są klatki z 500-1000 ms przed kliknięciem (łącznie z momentem kliknięcia),
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
        cols: int = 4,
        rows: int = 3,
        min_samples: int = 60,
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
        self.cols = cols
        self.rows = rows
        self.min_samples = min_samples
        self.model_x = model_x
        self.model_y = model_y

    def _generate_calibration_points(self):
        cols = self.cols
        rows = self.rows
        points = []

        margin_x = 0.02 * self.screen_w
        margin_y = 0.035 * self.screen_h

        usable_w = self.screen_w - 2 * margin_x
        usable_h = self.screen_h - 2 * margin_y

        for r in range(rows):
            for c in range(cols):
                x = int(margin_x + c * usable_w / (cols - 1))
                y = int(margin_y + r * usable_h / (rows - 1))
                points.append((x, y))

        return points

    def _draw_calibration_point(self, point_px):
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8) * 255
        cv2.circle(img, point_px, 10, (0, 0, 255), -1)  # czerwony punkt kalibracyjny
        cv2.imshow("Gaze Screen", img)

    def run(self):
        """
        Zwraca: (model_x, model_y)
        """
        points = self._generate_calibration_points()
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

                frame_buffer.append((timestamp_ms, frame.copy(), result))

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
                (frm, r) for (ts, frm, r) in frame_buffer
                if window_start <= ts <= click_time_ms
            ]

            print(f"  Zebrane klatki w oknie {self.window_ms} ms: {len(window_results)}")

            # Walidacja + cechy
            valid_features = []
            for frm, r in window_results:
                if self.validator.is_valid_frame(r):
                    feats = self.feature_extractor(frm, r)
                    valid_features.append(feats)

            if len(valid_features) == 0:
                print("  Uwaga: żadna klatka nie przeszła walidacji dla tego punktu.")
                continue

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

class RealTimeGazeTracker:
    """
    - używa wytrenowanych modeli regresyjnych,
    - wykorzystuje tylko klatki, które przejdą walidację,
    - stosuje wygładzanie (średnia krocząca po ostatnich N predykcjach),
    - wyświetla punkty kontrolne,
    - po kliknięciu w punkt kontrolny liczy błąd między jego pozycją a aktualną
      przewidywaną pozycją spojrzenia,
    - po zakończeniu (klawisz 'q') wypisuje średni błąd i odchylenie standardowe błędu.
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

    def _update_gaze(self, frame: np.ndarray, result: dict):
        if self.validator.is_valid_frame(result):
            features = self.feature_extractor(frame, result).reshape(1, -1)
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
            self._update_gaze(frame, result)

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


# ---------- Funkcja main ----------

def main():
    print("=" * 60)
    print("GAZE LOCALIZATION DEMO")
    print("=" * 60)
    print("\nOtwieranie kamery...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[FAIL] Nie można otworzyć kamery.")
        return

    print("[OK] Kamera otwarta")
    print("\n" + "=" * 60)

    # Pierwsza klatka do poznania rozdzielczości
    ret, frame = cap.read()
    if not ret:
        print("Brak klatek z kamery.")
        return

    detector = EyeDetector()
    validator = EyeFrameValidator()

    ################################################################################
    # Ustawienie rozmiaru eye patchy - liczby pikseli/cech wejściowych modelu
    feature_extractor = GazeFeatureExtractor(
        patch_height=8,  # 2 x (10x10) = 200 cech/pikseli
        patch_width=9,
    )
    ################################################################################

    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    window_w = screen_w
    window_h = int(screen_h * 0.9)
    screen_size = (window_w, window_h)

    cv2.namedWindow("Gaze Screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Screen", window_w, window_h)
    cv2.moveWindow("Gaze Screen", 0, 0)

    ################################################################################
    # ---------------------- Wybór modelu i wartości parametrów --------------------
    # "ridge", "gbrt", "svr" (ew. "mlp", "random_forest")
    model_type = "ridge"

    # poniżej parametry dla modelu Ridge
    alpha = 1.0   # z zakresu [0.1, 10]

    # poniżej parametry dla modelu SVR:
    c = 10  # 1, 5, 10, 20
    epsilon = 0.1  # 0.05, 0.1, 0.2
    gamma = "scale"  # "scale", "auto"

    # poniżej parametry dla modelu GradientBoostingRegressor:
    n_estimators = 300  # 200, 300
    learning_rate = 0.05  # 0.05, 0.1
    max_depth = 2  # 2, 3

    model_x = create_regressor(model_type, alpha=alpha, c=c, epsilon=epsilon, gamma=gamma,
                               n_estimators=n_estimators, learning_rate=learning_rate,
                               max_depth=max_depth)
    model_y = create_regressor(model_type, alpha=alpha, c=c, epsilon=epsilon, gamma=gamma,
                               n_estimators=n_estimators, learning_rate=learning_rate,
                               max_depth=max_depth)
    ################################################################################

    # 1) Kalibracja
    # Ustawienia parametrów (dot. liczby próbek): czas próbkowania na jeden punkt
    # kalibracyjny, liczba punktów kalibracyjnych
    calibrator = ClickCalibrator(
        detector, cap, feature_extractor, validator,
        screen_size=screen_size,
        window_ms = 1000,  # 500, 750, 1000 ms
        # 20/16/15/12/9 punktów kalibracyjnych:
        cols=5,
        rows=4,
        min_samples=60,
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
    tracker = RealTimeGazeTracker(
        detector, cap, feature_extractor, validator,
        model_x_trained, model_y_trained, screen_size=screen_size
    )
    tracker.run()

    cap.release()
    cv2.destroyAllWindows()
    print("Program zakończony.\n")

if __name__ == "__main__":
    main()
