import cv2
import numpy as np
import ctypes
import math
import time

from gaze_localizator import GazeEngine

WINDOW_NAME = "Gaze Screen"


# ======================================================================
#  Detection visualization in the camera window
# ======================================================================

def visualize_detection(frame: np.ndarray, result: dict) -> np.ndarray:
    display = frame.copy()

    if result.get("face_bbox") is not None:
        x, y, w, h = result["face_bbox"]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

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


# ======================================================================
#  Generating calibration points
# ======================================================================

def generate_calibration_points(screen_w, screen_h, cols=5, rows=4):
    points = []

    margin_x = 0.02 * screen_w
    margin_y = 0.035 * screen_h

    usable_w = screen_w - 2 * margin_x
    usable_h = screen_h - 2 * margin_y

    for r in range(rows):
        for c in range(cols):
            x = int(margin_x + c * usable_w / (cols - 1))
            y = int(margin_y + r * usable_h / (rows - 1))
            points.append((x, y))

    return points


# ======================================================================
#  Phase 1: Calibration using GazeEngine
# ======================================================================

def calibration_mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["clicked"] = True
        state["click_pos"] = (x, y)


def run_calibration_with_engine(
    engine: GazeEngine,
    screen_size,
    cols=5,
    rows=4,
    window_ms: int = 1000,
):
    screen_w, screen_h = screen_size

    points = generate_calibration_points(screen_w, screen_h, cols=cols, rows=rows)
    print(f"Starting calibration with {len(points)} points.")

    engine.start_calibration()

    for idx, target_px in enumerate(points, start=1):
        print(f"\nCalibration point {idx}/{len(points)}: {target_px}")
        print("Look at the point and click when you are ready.")

        click_state = {
            "clicked": False,
            "click_pos": (0, 0),
            "click_time_ms": 0,
        }
        cv2.setMouseCallback(WINDOW_NAME, calibration_mouse_callback, click_state)

        frame_buffer = []

        while True:
            # 1) Get the camera frame and detection result from GazeEngine
            frame, result = engine.capture_and_detect()
            if frame is None:
                continue

            timestamp_ms = int(time.time() * 1000)
            frame_buffer.append((timestamp_ms, frame.copy(), result))

            # 2) Camera preview with drawn detection result
            cam_display = visualize_detection(frame, result)
            cv2.imshow("Camera", cam_display)

            # 3) Drawing the red calibration point
            img = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
            cv2.circle(img, target_px, 10, (0, 0, 255), -1)
            cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Calibration interrupted by the 'q' key.")
                return False

            if click_state["clicked"]:
                cx, cy = click_state["click_pos"]
                tx, ty = target_px
                dist = math.hypot(cx - tx, cy - ty)

                if dist <= 20:
                    click_time_ms = int(time.time() * 1000)
                    click_state["click_time_ms"] = click_time_ms
                    break
                else:
                    print("Click missed the calibration point – try again.")
                    click_state["clicked"] = False

        click_time_ms = click_state["click_time_ms"]
        window_start = click_time_ms - window_ms

        window_results = [
            (frm, r)
            for (ts, frm, r) in frame_buffer
            if window_start <= ts <= click_time_ms
        ]

        print(f"  Collected frames in {window_ms} ms window: {len(window_results)}")

        accepted = 0
        for frm, r in window_results:
            ok = engine.add_calibration_sample(
                target_px[0], target_px[1],
                frame=frm,
                result=r,
            )
            if ok:
                accepted += 1

        print(f"  Accepted frames (after validation): {accepted}")

    ok = engine.fit_models()
    if not ok:
        print("Calibration failed – models could not be trained.")
        return False

    return True


# ======================================================================
#  Phase 2: gaze tracking + control points
# ======================================================================

def generate_control_points(screen_w, screen_h, num_points=8):
    points = []
    for _ in range(num_points):
        px = np.random.randint(0, screen_w)
        py = np.random.randint(0, screen_h)
        points.append((px, py))
    return points


def tracking_mouse_callback(event, x, y, flags, state):
    """
    Mouse callback for the control points phase.

    The state maintains:
      - "control_points": list of control points,
      - "current_gaze": (gx, gy),
      - "errors": list of error values.
    """
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    control_points = state["control_points"]
    current_gaze = state["current_gaze"]
    errors = state["errors"]

    click_pos = np.array([x, y], dtype=np.float32)
    cp_array = np.array(control_points, dtype=np.float32)

    dists = np.linalg.norm(cp_array - click_pos, axis=1)
    idx = int(np.argmin(dists))
    min_dist = float(dists[idx])

    if min_dist > 20.0:
        print("Click missed the calibration point.")
        return

    cp = control_points[idx]
    gx, gy = current_gaze
    err = math.sqrt((gx - cp[0]) ** 2 + (gy - cp[1]) ** 2)
    errors.append(err)

    print(
        f"Clicked control point {idx}: {cp}, "
        f"gaze={current_gaze}, error={err:.1f} px"
    )


def run_tracking_with_engine(
    engine: GazeEngine,
    screen_size,
    smoothing_window: int = 5,
    num_control_points: int = 8,
):
    screen_w, screen_h = screen_size

    control_points = generate_control_points(screen_w, screen_h, num_control_points)
    history = []

    state = {
        "control_points": control_points,
        "current_gaze": (screen_w // 2, screen_h // 2),
        "errors": [],
    }

    cv2.setMouseCallback(WINDOW_NAME, tracking_mouse_callback, state)

    print("\nGaze tracking started.")
    print("Click the blue control points to measure the error.")
    print("Press 'q' to exit.")

    while True:
        # 1) Get the frame and detection result
        frame, result = engine.capture_and_detect()

        if frame is not None and result is not None:
            # 2) Camera preview with drawn detection result
            cam_display = visualize_detection(frame, result)
            cv2.imshow("Camera", cam_display)

            # 3) Gaze prediction based on the current frame
            gaze = engine.predict_gaze(frame, result)
        else:
            gaze = None

        if gaze is not None:
            gx, gy = gaze
            state["current_gaze"] = (gx, gy)
            history.append((gx, gy))
            if len(history) > smoothing_window:
                history.pop(0)

        # 4) Drawing the screen with blue control points
        img = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255

        for (cx, cy) in control_points:
            cv2.circle(img, (int(cx), int(cy)), 8, (255, 0, 0), -1)

        # current gaze point – red
        gx, gy = state["current_gaze"]
        cv2.circle(img, (int(gx), int(gy)), 10, (0, 0, 255), -1)

        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    errors = state["errors"]
    if len(errors) > 0:
        errors_arr = np.array(errors, dtype=np.float32)
        mean_err = float(np.mean(errors_arr))
        std_err = float(np.std(errors_arr))
        print("\n=== Summary of model accuracy ===")
        print(f"Number of control points clicked: {len(errors_arr)}")
        print(f"Mean error : {mean_err:.2f} px")
        print(f"Standard deviation : {std_err:.2f} px")
    else:
        print("\nNo control points recorded - no error metrics.")


def main():
    print("=" * 60)
    print("GAZE LOCALIZATION DEMO (GazeEngine)")
    print("=" * 60)

    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    window_w = screen_w
    window_h = int(screen_h * 0.9)
    screen_size = (window_w, window_h)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, window_w, window_h)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    engine = GazeEngine(
        screen_size=screen_size,
        model_type="ridge",
        patch_height=8,
        patch_width=9,
        min_samples=60,
        smoothing_window=5,
        alpha=1.0,
        c=10,
        epsilon=0.1,
        gamma="scale",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
    )

    try:
        ok = run_calibration_with_engine(
            engine,
            screen_size,
            cols=5,
            rows=4,
            window_ms=1000,
        )

        if not ok or not engine.is_calibrated():
            print("Calibration failed – end of program.")
            return

        run_tracking_with_engine(
            engine,
            screen_size,
            smoothing_window=5,
            num_control_points=8,
        )

    finally:
        engine.close()
        cv2.destroyAllWindows()
        print("Program finished.\n")


if __name__ == "__main__":
    main()
