import unittest

import cv2
import numpy as np

from regular8_geometry_calibration import (
    Regular8GeometryCalibrator,
    _choose_pair,
)


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
        status = calibrator.add_frame(
            1, preview(1), [(160.0, 200.0), (160.0, 460.0)]
        )
        self.assertEqual(status["samples"], 0)
        self.assertEqual(status["reason"], "malformed_sprocket_observation")
        self.assertEqual(status["rejected_samples"], 1)

    def test_freeze_uses_only_the_stable_window(self):
        calibrator = Regular8GeometryCalibrator("/capture/project")
        for frame in range(1, 31):
            pitch = 240 if frame % 2 else 290
            calibrator.add_frame(
                frame,
                preview(frame, pitch=pitch),
                sprockets(pitch=pitch, width=180),
            )

        self.assertFalse(calibrator.frozen)
        for frame in range(31, 43):
            status = calibrator.add_frame(
                frame, preview(frame), sprockets(width=130)
            )
            if status["frozen"]:
                break

        self.assertTrue(status["frozen"])
        geometry = status["capture_geometry"]
        stable_width = np.median([
            sample["p24_capture_roi_width"]
            for sample in calibrator.samples[-calibrator.stability_window:]
        ])
        self.assertAlmostEqual(geometry["p24_capture_width_px"], stable_width)
        self.assertAlmostEqual(geometry["p24_capture_width_px"], 346.8947, places=3)
        self.assertEqual(
            geometry["calibration_frames"],
            [sample["frame"] for sample in calibrator.samples[-12:]],
        )

    def test_freeze_rejects_an_invalid_candidate_geometry(self):
        calibrator = Regular8GeometryCalibrator("/capture/project")
        calibrator._build_geometry = lambda samples: {"frozen": True}
        for frame in range(1, 19):
            status = calibrator.add_frame(frame, preview(frame), sprockets())
        self.assertFalse(status["frozen"])
        self.assertEqual(status["reason"], "frozen_geometry_failed_validation")

    def test_pair_scoring_target_scales_with_preview_coordinates(self):
        boxes = [
            (160.0, 57.0, 130.0, 80.0, 0.0),
            (160.0, 62.0, 130.0, 80.0, 0.0),
            (160.0, 357.0, 130.0, 80.0, 0.0),
        ]
        pitch_range = (500.0 / (1520.0 / 570.0), 1000.0 / (1520.0 / 570.0))
        scaled = _choose_pair(
            boxes,
            (570, 760, 3),
            pitch_range=pitch_range,
            expected_pitch=785.0 / (1520.0 / 570.0),
        )
        raw_coordinate = _choose_pair(
            [tuple(value * (1520.0 / 570.0) for value in box) for box in boxes],
            (1520, 2028, 3),
        )
        self.assertEqual([item[1] for item in scaled], [62.0, 357.0])
        self.assertEqual(
            [round(item[1] / (1520.0 / 570.0), 5) for item in raw_coordinate],
            [62.0, 357.0],
        )


if __name__ == "__main__":
    unittest.main()
