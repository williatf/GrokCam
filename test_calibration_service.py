import importlib.util
import unittest

HAS_IMAGE_DEPENDENCIES = (
    importlib.util.find_spec('cv2') is not None
    and importlib.util.find_spec('numpy') is not None
)

if HAS_IMAGE_DEPENDENCIES:
    from calibration_service import Regular8CalibrationService


@unittest.skipUnless(HAS_IMAGE_DEPENDENCIES, 'OpenCV and NumPy are required')
class Regular8CalibrationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = Regular8CalibrationService(
            camera=None,
            tc=None,
            detector=None,
            settings={
                'calibration_resolution': [2028, 1520],
                'steps_per_pitch': 277,
            },
        )

    @staticmethod
    def valid_sample(pitch, area=100000):
        return {
            'pitch_valid': True,
            'sprocket_pitch_px': pitch,
            'full_sprocket_areas': [area],
        }

    def test_summary_preserves_production_statistics(self):
        summary = self.service.compute_summary([
            self.valid_sample(780, 98000),
            self.valid_sample(782, 102000),
            {'pitch_valid': False, 'full_sprocket_areas': []},
        ])

        self.assertEqual(summary['valid_samples'], 2)
        self.assertEqual(summary['total_samples'], 3)
        self.assertEqual(summary['pitch_mean'], 781.0)
        self.assertEqual(summary['pitch_min'], 780.0)
        self.assertEqual(summary['pitch_max'], 782.0)
        self.assertEqual(summary['area_mean'], 100000.0)

    def test_proposal_uses_filtered_pitch_and_measured_motor_value(self):
        samples = [
            self.valid_sample(pitch)
            for pitch in (779, 780, 780, 780, 781, 1200)
        ]
        proposed, can_save, reason = self.service.build_proposed_calibration(
            samples,
            {'exposure_time': 814},
            {'motor_updated': True, 'motor_steps_per_pitch': 281},
        )

        self.assertTrue(can_save)
        self.assertIsNone(reason)
        self.assertEqual(proposed['calibration_version'], 2)
        self.assertEqual(proposed['calibration_resolution'], [2028, 1520])
        self.assertEqual(proposed['sprocket_pitch_px'], 780.0)
        self.assertEqual(proposed['steps_per_pitch'], 281)
        self.assertAlmostEqual(proposed['steps_per_px'], 281 / 780)

    def test_proposal_requires_five_valid_pitch_samples(self):
        proposed, can_save, reason = self.service.build_proposed_calibration(
            [self.valid_sample(780) for _ in range(4)],
            {'exposure_time': 814},
        )

        self.assertIsNone(proposed)
        self.assertFalse(can_save)
        self.assertEqual(reason, 'need_at_least_5_valid_pitch_samples')


if __name__ == '__main__':
    unittest.main()
