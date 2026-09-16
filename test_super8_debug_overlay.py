import unittest
import importlib.util

HAS_IMAGE_DEPENDENCIES = (
    importlib.util.find_spec('cv2') is not None
    and importlib.util.find_spec('numpy') is not None
)

if HAS_IMAGE_DEPENDENCIES:
    import numpy as np
    from super8_debug_overlay import annotate_super8_debug_preview


@unittest.skipUnless(HAS_IMAGE_DEPENDENCIES, 'OpenCV and NumPy are required')
class Super8DebugOverlayTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.candidates = [{
            'center_x': 40.0, 'center_y': 60.0,
            'width': 20.0, 'height': 30.0, 'score': 0.9,
        }, {
            'center_x': 40.0, 'center_y': 90.0,
            'width': 18.0, 'height': 28.0, 'score': 0.8,
        }]

    def test_overlay_does_not_modify_detector_input(self):
        original = self.image.copy()
        annotate_super8_debug_preview(self.image, self.candidates)
        np.testing.assert_array_equal(self.image, original)

    def test_candidate_geometry_and_centers_are_drawn(self):
        output = annotate_super8_debug_preview(self.image, self.candidates)
        self.assertGreater(int(output[60, 30].max()), 0)
        self.assertGreater(int(output[60, 40].max()), 0)
        self.assertGreater(int(output[90, 31].max()), 0)
        self.assertGreater(int(output[90, 40].max()), 0)

    def test_trusted_selected_candidate_is_distinct(self):
        output = annotate_super8_debug_preview(
            self.image, self.candidates, selected_y=60.0, phase_trusted=True,
        )
        self.assertGreater(int(output[60, 40, 1]), int(output[60, 40, 0]))
        self.assertNotEqual(tuple(output[90, 40]), tuple(output[60, 40]))

    def test_phase_markers_render_when_trusted_or_untrusted(self):
        trusted = annotate_super8_debug_preview(
            self.image, selected_y=60.0, phase_trusted=True,
            predicted_y=58.0, registration_target_y=70.0, crop_center_y=60.0,
        )
        untrusted = annotate_super8_debug_preview(
            self.image, predicted_y=58.0, registration_target_y=70.0,
        )
        self.assertGreater(int(trusted[58].max()), 0)
        self.assertGreater(int(trusted[70].max()), 0)
        self.assertGreater(int(untrusted[58].max()), 0)

    def test_missing_candidates_and_prediction_are_safe(self):
        output = annotate_super8_debug_preview(self.image)
        self.assertEqual(output.shape, self.image.shape)


if __name__ == '__main__':
    unittest.main()
