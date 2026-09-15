import importlib.util
import unittest


HAS_IMAGE_DEPENDENCIES = (
    importlib.util.find_spec('cv2') is not None
    and importlib.util.find_spec('numpy') is not None
)

if HAS_IMAGE_DEPENDENCIES:
    from super8_detector import Super8Detector


@unittest.skipUnless(HAS_IMAGE_DEPENDENCIES, 'OpenCV and NumPy are required')
class Super8DetectorCalibrationTests(unittest.TestCase):
    def test_calibration_api_returns_all_complete_candidates(self):
        detector = Super8Detector()
        detector.last_candidates = []

        def fake_registration(_frame):
            detector.last_candidates = [
                {'classification': 'COMPLETE', 'center_y': 400, 'score': 0.9},
                {'classification': 'PARTIAL_TOP', 'center_y': 10, 'score': 0.8},
                {'classification': 'COMPLETE', 'center_y': 1240, 'score': 0.85},
            ]
            return {'mode': 'direct', 'actual_y': 400}

        detector.detect_registration = fake_registration
        candidates = detector.detect_calibration_candidates(object())
        self.assertEqual([item['center_y'] for item in candidates], [400, 1240])


if __name__ == '__main__':
    unittest.main()
