import unittest

from super8_calibration_result import (
    build_super8_calibration_proposal,
    build_super8_client_summary,
)


class Super8CalibrationResultTests(unittest.TestCase):
    @staticmethod
    def result(transition_count=5, disagreement=1.0):
        return {
            'rejected_transitions': 1,
            'measurements': {
                'steps_per_pitch': {'count': transition_count, 'median': 310.2},
                'sprocket_pitch_px': {
                    'count': 12,
                    'mean': 842.9,
                    'median': 842.8,
                    'std': 1.1,
                    'min': 841.0,
                    'max': 845.0,
                },
                'steps_per_px': {'count': 7, 'median': 0.3681},
                'confidence': {'count': 40, 'median': 0.97},
                'area': {
                    'count': 40,
                    'mean': 43100.0,
                    'median': 43000.0,
                    'std': 300.0,
                    'min': 42500.0,
                    'max': 43700.0,
                },
                'cross_check': {'difference_percent': disagreement},
                'steps_per_pitch_values': [309, 310, 311, 310, 311],
                'sprocket_pitch_px_values': [842, 843],
                'steps_per_px_values': [0.367, 0.369],
            },
        }

    def test_minimum_transition_gate(self):
        proposal, can_save, reason = build_super8_calibration_proposal(
            self.result(4),
            {'exposure_time': 814, 'gain': 1.0},
            (2028, 1520),
        )
        self.assertIsNone(proposal)
        self.assertFalse(can_save)
        self.assertEqual(reason, 'need_at_least_5_accepted_transitions')

    def test_constructs_formal_measured_proposal(self):
        proposal, can_save, reason = build_super8_calibration_proposal(
            self.result(),
            {'exposure_time': 814, 'gain': 1.0},
            (2028, 1520),
            timestamp='2026-09-15T12:00:00-0400',
        )
        self.assertTrue(can_save)
        self.assertIsNone(reason)
        self.assertEqual(proposal['film_format'], 'super8')
        self.assertEqual(proposal['status'], 'calibrated')
        self.assertEqual(proposal['steps_per_pitch'], 310)
        self.assertEqual(proposal['calibration_resolution'], [2028, 1520])
        self.assertEqual(proposal['quality']['successful_transitions'], 5)
        self.assertEqual(proposal['sprocket_area_min'], 37840)
        self.assertEqual(proposal['sprocket_area_max'], 48160)

    def test_rejects_large_independent_cross_check_disagreement(self):
        proposal, can_save, reason = build_super8_calibration_proposal(
            self.result(disagreement=15),
            {'exposure_time': 814, 'gain': 1.0},
            (2028, 1520),
        )
        self.assertIsNone(proposal)
        self.assertFalse(can_save)
        self.assertEqual(reason, 'cross_check_disagreement_too_large')

    def test_client_summary_exposes_legacy_gui_fields(self):
        result = self.result()
        result['valid'] = True
        result['target_transitions'] = 8
        summary = build_super8_client_summary(result, 8)

        self.assertEqual(summary['pitch_mean'], 842.9)
        self.assertEqual(summary['area_mean'], 43100.0)
        self.assertEqual(summary['valid_samples'], 5)
        self.assertEqual(summary['total_samples'], 8)


if __name__ == '__main__':
    unittest.main()
