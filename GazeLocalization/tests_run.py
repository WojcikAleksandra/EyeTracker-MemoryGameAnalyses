import unittest
import tests_gaze_localizator
from tests_gaze_localizator import SummaryTextTestRunner

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests_gaze_localizator)
    runner = SummaryTextTestRunner(verbosity=2, descriptions=True, buffer=True)
    runner.run(suite)
