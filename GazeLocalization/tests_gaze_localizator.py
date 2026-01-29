import time
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

import gaze_localizator as gl


# ======================================================================
# Helpers: fake camera to avoid real hardware dependency
# ======================================================================

class FakeVideoCapture:
    def __init__(self, index=0, frame=None):
        self._opened = True
        self._frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return self._opened

    def read(self):
        return True, self._frame.copy()

    def release(self):
        self._opened = False


def make_valid_result(left_bbox, right_bbox, face_bbox=(50, 50, 200, 200)):
    return {
        "face_bbox": face_bbox,
        "left_eye": {"bbox": left_bbox},
        "right_eye": {"bbox": right_bbox},
        "status": "ok",
    }


# ======================================================================
# 1) GazeEngine init/config tests
# ======================================================================

class TestGazeEngineInit(unittest.TestCase):

    @patch.object(gl, "EyeDetector", autospec=True)
    @patch.object(gl.cv2, "VideoCapture", autospec=True)
    def test_default_params_and_initial_state(self, mock_vc, mock_detector):
        """Test default parameters and initial state."""
        mock_vc.return_value = FakeVideoCapture()

        engine = gl.GazeEngine(screen_size=(1920, 1080))

        self.assertEqual(engine.screen_w, 1920)
        self.assertEqual(engine.screen_h, 1080)
        self.assertFalse(engine.is_calibrated())
        self.assertEqual(len(engine.history), 0)
        self.assertEqual(engine.last_gaze, (1920 // 2, 1080 // 2))

        engine.close()

    @patch.object(gl, "EyeDetector", autospec=True)
    @patch.object(gl.cv2, "VideoCapture", autospec=True)
    def test_regressor_models_created(self, mock_vc, mock_detector):
        """Test that regression models are created."""
        mock_vc.return_value = FakeVideoCapture()

        engine = gl.GazeEngine(screen_size=(800, 600), model_type="ridge")

        self.assertIsNotNone(engine.model_x)
        self.assertIsNotNone(engine.model_y)
        self.assertEqual(engine.model_type, "ridge")

        engine.close()


# ======================================================================
# 2) GazeFeatureExtractor tests
# ======================================================================

class TestGazeFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.patch_h = 6
        self.patch_w = 10
        self.extractor = gl.GazeFeatureExtractor(patch_height=self.patch_h, patch_width=self.patch_w)

        self.frame_bgr = np.zeros((200, 300, 3), dtype=np.uint8)
        self.frame_bgr[50:90, 60:120] = 128
        self.frame_bgr[50:90, 160:220] = 200

        self.left_bbox = (60, 50, 60, 40)
        self.right_bbox = (160, 50, 60, 40)
        self.result = make_valid_result(self.left_bbox, self.right_bbox)

    def test_feature_vector_size(self):
        """Test feature vector size."""
        feats = self.extractor(self.frame_bgr, self.result)
        expected = 2 * self.patch_h * self.patch_w
        self.assertEqual(feats.shape, (expected,))

    def test_normalization_range_0_1(self):
        """Test normalization to [0, 1]."""
        feats = self.extractor(self.frame_bgr, self.result)
        self.assertTrue(np.all(feats >= 0.0))
        self.assertTrue(np.all(feats <= 1.0))

    def test_grayscale_and_bgr_input(self):
        """Test handling grayscale and BGR inputs."""
        feats_bgr = self.extractor(self.frame_bgr, self.result)

        frame_gray = np.mean(self.frame_bgr, axis=2).astype(np.uint8)
        feats_gray = self.extractor(frame_gray, self.result)

        self.assertEqual(feats_bgr.shape, feats_gray.shape)
        self.assertTrue(np.all(feats_gray >= 0.0))
        self.assertTrue(np.all(feats_gray <= 1.0))

    def test_determinism(self):
        """Test determinism (same input -> same output)."""
        feats1 = self.extractor(self.frame_bgr, self.result)
        feats2 = self.extractor(self.frame_bgr, self.result)
        self.assertTrue(np.array_equal(feats1, feats2))


# ======================================================================
# 3) EyeFrameValidator tests
# ======================================================================

class TestEyeFrameValidator(unittest.TestCase):

    def setUp(self):
        self.validator = gl.EyeFrameValidator()
        self.left_bbox = (60, 50, 60, 40)
        self.right_bbox = (160, 50, 60, 40)

    def test_reject_none_input(self):
        """Test rejection of missing input."""
        self.assertFalse(self.validator.is_valid_frame(None))

    def test_require_face_bbox(self):
        """Test requirement of face bbox."""
        result = {
            "face_bbox": None,
            "left_eye": {"bbox": self.left_bbox},
            "right_eye": {"bbox": self.right_bbox},
        }
        self.assertFalse(self.validator.is_valid_frame(result))

    def test_require_both_eyes(self):
        """Test requirement of both eyes."""
        result_left_missing = {"face_bbox": (0, 0, 100, 100), "left_eye": None, "right_eye": {"bbox": self.right_bbox}}
        result_right_missing = {"face_bbox": (0, 0, 100, 100), "left_eye": {"bbox": self.left_bbox}, "right_eye": None}
        self.assertFalse(self.validator.is_valid_frame(result_left_missing))
        self.assertFalse(self.validator.is_valid_frame(result_right_missing))

    def test_accept_minimal_valid_record(self):
        """Test acceptance of a minimal valid record."""
        result = make_valid_result(self.left_bbox, self.right_bbox)
        self.assertTrue(self.validator.is_valid_frame(result))


# ======================================================================
# 4) Calibration tests: reset and adding samples
# ======================================================================

class TestCalibration(unittest.TestCase):

    @patch.object(gl, "EyeDetector", autospec=True)
    @patch.object(gl.cv2, "VideoCapture", autospec=True)
    def setUp(self, mock_vc, mock_detector):
        mock_vc.return_value = FakeVideoCapture()
        self.engine = gl.GazeEngine(screen_size=(800, 600), min_samples=3, smoothing_window=5)

        self.frame = np.zeros((200, 300, 3), dtype=np.uint8)
        self.result = make_valid_result((60, 50, 60, 40), (160, 50, 60, 40))

    def tearDown(self):
        self.engine.close()

    def test_start_calibration_resets_state(self):
        """Test calibration reset (start_calibration)."""
        self.engine.calib_X = [np.zeros((10,), dtype=np.float32)]
        self.engine.calib_yx = [10.0]
        self.engine.calib_yy = [20.0]
        self.engine._is_calibrated = True
        self.engine.history.append((100.0, 100.0))

        self.engine.start_calibration()

        self.assertEqual(len(self.engine.calib_X), 0)
        self.assertEqual(len(self.engine.calib_yx), 0)
        self.assertEqual(len(self.engine.calib_yy), 0)
        self.assertFalse(self.engine.is_calibrated())
        self.assertEqual(len(self.engine.history), 0)

    def test_add_calibration_sample_only_when_features_available(self):
        """Test add_calibration_sample adds data only when features exist."""
        with patch.object(self.engine, "grab_features_if_valid", return_value=None):
            before = len(self.engine.calib_X)
            ok = self.engine.add_calibration_sample(100, 200, self.frame, self.result)
            self.assertFalse(ok)
            self.assertEqual(len(self.engine.calib_X), before)

        fake_feats = np.ones((120,), dtype=np.float32)
        with patch.object(self.engine, "grab_features_if_valid", return_value=fake_feats):
            ok = self.engine.add_calibration_sample(100, 200, self.frame, self.result)
            self.assertTrue(ok)
            self.assertEqual(len(self.engine.calib_X), 1)
            self.assertEqual(self.engine.calib_yx[-1], 100.0)
            self.assertEqual(self.engine.calib_yy[-1], 200.0)


# ======================================================================
# 5) fit_models tests
# ======================================================================

class TestFitModels(unittest.TestCase):

    @patch.object(gl, "EyeDetector", autospec=True)
    @patch.object(gl.cv2, "VideoCapture", autospec=True)
    def setUp(self, mock_vc, mock_detector):
        mock_vc.return_value = FakeVideoCapture()
        self.engine = gl.GazeEngine(screen_size=(800, 600), min_samples=3, smoothing_window=5)

    def tearDown(self):
        self.engine.close()

    def test_fit_models_no_samples(self):
        """Test fit_models with no samples."""
        self.engine.start_calibration()
        ok = self.engine.fit_models()
        self.assertFalse(ok)
        self.assertFalse(self.engine.is_calibrated())

    def test_fit_models_too_few_samples(self):
        """Test fit_models with too few samples."""
        self.engine.start_calibration()
        self.engine.calib_X = [np.zeros((10,), dtype=np.float32)] * 2
        self.engine.calib_yx = [10.0, 20.0]
        self.engine.calib_yy = [30.0, 40.0]

        ok = self.engine.fit_models()
        self.assertFalse(ok)
        self.assertFalse(self.engine.is_calibrated())

    def test_fit_models_success(self):
        """Test successful model fitting."""
        self.engine.start_calibration()

        self.engine.calib_X = [
            np.zeros((10,), dtype=np.float32),
            np.ones((10,), dtype=np.float32),
            np.full((10,), 2.0, dtype=np.float32),
        ]
        self.engine.calib_yx = [10.0, 20.0, 30.0]
        self.engine.calib_yy = [40.0, 50.0, 60.0]

        ok = self.engine.fit_models()
        self.assertTrue(ok)
        self.assertTrue(self.engine.is_calibrated())


# ======================================================================
# 6) predict_gaze tests + smoothing
# ======================================================================

class TestPredictGaze(unittest.TestCase):

    @patch.object(gl, "EyeDetector", autospec=True)
    @patch.object(gl.cv2, "VideoCapture", autospec=True)
    def setUp(self, mock_vc, mock_detector):
        mock_vc.return_value = FakeVideoCapture()
        self.engine = gl.GazeEngine(screen_size=(800, 600), min_samples=3, smoothing_window=3)

        self.engine.model_x = MagicMock()
        self.engine.model_y = MagicMock()

    def tearDown(self):
        self.engine.close()

    def test_predict_without_calibration(self):
        """Test prediction without calibration."""
        self.engine._is_calibrated = False
        out = self.engine.predict_gaze(frame=np.zeros((10, 10, 3), dtype=np.uint8), result={})
        self.assertIsNone(out)

    def test_behavior_when_no_valid_frame(self):
        """Test behavior when frame is invalid (no features)."""
        self.engine._is_calibrated = True

        with patch.object(self.engine, "grab_features_if_valid", return_value=None):
            out = self.engine.predict_gaze(frame=np.zeros((10, 10, 3), dtype=np.uint8), result={})
            self.assertIsNone(out)

        self.engine.history.append((100.0, 200.0))
        self.engine.history.append((110.0, 210.0))

        with patch.object(self.engine, "grab_features_if_valid", return_value=None):
            out = self.engine.predict_gaze(frame=np.zeros((10, 10, 3), dtype=np.uint8), result={})
            self.assertEqual(out, (105, 205))

    def test_history_update_and_smoothing(self):
        """Test history update and smoothing average."""
        self.engine._is_calibrated = True
        feats = np.ones((10,), dtype=np.float32)

        self.engine.model_x.predict.side_effect = [[10.0], [20.0], [30.0]]
        self.engine.model_y.predict.side_effect = [[100.0], [200.0], [300.0]]

        with patch.object(self.engine, "grab_features_if_valid", return_value=feats):
            out1 = self.engine.predict_gaze(frame=np.zeros((10, 10, 3), dtype=np.uint8), result={})
            out2 = self.engine.predict_gaze(frame=np.zeros((10, 10, 3), dtype=np.uint8), result={})
            out3 = self.engine.predict_gaze(frame=np.zeros((10, 10, 3), dtype=np.uint8), result={})

        self.assertEqual(out1, (10, 100))
        self.assertEqual(out2, (15, 150))
        self.assertEqual(out3, (20, 200))
        self.assertEqual(len(self.engine.history), 3)

    def test_output_type_and_range(self):
        """Test output type and basic value conversion."""
        self.engine._is_calibrated = True
        feats = np.ones((10,), dtype=np.float32)
        self.engine.model_x.predict.return_value = [123.4]
        self.engine.model_y.predict.return_value = [567.8]

        with patch.object(self.engine, "grab_features_if_valid", return_value=feats):
            out = self.engine.predict_gaze(frame=np.zeros((10, 10, 3), dtype=np.uint8), result={})

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], int)
        self.assertIsInstance(out[1], int)
        self.assertEqual(out, (123, 567))


# ======================================================================
# Summary block
# ======================================================================

class SummaryTextTestRunner(unittest.TextTestRunner):
    def run(self, test):
        start = time.perf_counter()
        result = super().run(test)
        elapsed = time.perf_counter() - start

        failures = len(result.failures)
        errors = len(result.errors)

        skipped = getattr(result, "skipped", [])
        expected_failures = getattr(result, "expectedFailures", [])
        unexpected_successes = getattr(result, "unexpectedSuccesses", [])

        skipped_n = len(skipped)
        exp_fail_n = len(expected_failures)
        unexp_succ_n = len(unexpected_successes)

        successes = result.testsRun - failures - errors - skipped_n - exp_fail_n

        print("\n" + "=" * 70)
        print("\nTEST SUMMARY")
        print("=" * 70 + "\n")
        print(f"Tests run:  {result.testsRun}")
        print(f"Successes:  {successes}")
        print(f"Failures:   {failures}")
        print(f"Errors:     {errors}")

        if failures == 0 and errors == 0:
            print("\n[OK] ALL TESTS PASSED")
        else:
            print("\n[!] SOME TESTS FAILED")

        print("\n" + "=" * 70 + "\n")

        return result