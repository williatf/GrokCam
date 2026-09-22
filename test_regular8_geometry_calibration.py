import unittest

import cv2
import numpy as np

from regular8_geometry_calibration import Regular8GeometryCalibrator


def preview(frame_index, *, pitch=260, support_width=171):
    image = np.zeros((570, 760, 3), dtype=np.uint8)
    image[120:200, 80:80 + support_width] = 240
    image[120 + pitch:200 + pitch, 80:80 + support_width] = 240
    # A changing image region represents successive film content and prevents
    # a stalled transport from being counted as a calibration sample.
    image[245:345, 350:550] = (20 + (frame_index * 17) % 200)
    return image


def sprockets(*, pitch=260, width=130, height=90):
    return [
        (165.0, 160.0, float(width), float(height), 0.0),
        (165.0, 160.0 + pitch, float(width), float(height), 0.0),
    ]


class Regular8GeometryCalibrationTests(unittest.TestCase):
    def test_freezes_after_stable_samples_and_emits_public_schema(self):
        calibrator = Regular8GeometryCalibrator("/capture/project")
        for frame in range(1, 18):
            status = calibrator.add_frame(frame, preview(frame), sprockets())
        self.assertFalse(status["frozen"])

        status = calibrator.add_frame(18, preview(18), sprockets())
        self.assertTrue(status["frozen"])
        geometry = status["capture_geometry"]
        self.assertEqual(geometry["schema_version"], 1)
        self.assertEqual(geometry["film_format"], "regular8")
        self.assertEqual(geometry["calibration_method"], "regular8_capture_geometry_v1")
        self.assertTrue(geometry["frozen"])
        self.assertEqual(geometry["p15_image_support_width_px"], geometry["sprocket_width_px"])
        self.assertIn("capture_lower_top_minus_midpoint", geometry["quality"])
        self.assertIn("capture_lower_x_minus_midpoint", geometry["quality"])
        self.assertAlmostEqual(geometry["sprocket_pitch_px"], 693.333, places=2)

    def test_stationary_frames_are_not_used_as_calibration_samples(self):
        calibrator = Regular8GeometryCalibrator("/capture/project")
        image = preview(1)
        calibrator.add_frame(1, image, sprockets())
        status = calibrator.add_frame(2, image.copy(), sprockets())
        self.assertEqual(status["samples"], 1)
        self.assertEqual(status["reason"], "stationary_frame")
        self.assertEqual(status["rejected_samples"], 1)

    def test_unstable_pitch_does_not_freeze(self):
        calibrator = Regular8GeometryCalibrator("/capture/project")
        for frame in range(1, 25):
            pitch = 240 if frame % 2 else 290
            status = calibrator.add_frame(frame, preview(frame, pitch=pitch),
                                          sprockets(pitch=pitch))
        self.assertFalse(status["frozen"])
        self.assertEqual(calibrator.finalize()["state"], "failed")

    def test_malformed_detection_is_rejected_without_exception(self):
        calibrator = Regular8GeometryCalibrator("/capture/project")
        status = calibrator.add_frame(1, np.zeros((570, 760, 3), dtype=np.uint8), [])
        self.assertEqual(status["samples"], 0)
        self.assertEqual(status["reason"], "no_image_support_pair")


if __name__ == "__main__":
    unittest.main()
