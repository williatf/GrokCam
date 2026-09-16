import unittest

from super8_phase_tracker import PhaseResult, select_super8_crop_guidance


class Super8CropGuidanceTests(unittest.TestCase):
    def result(self, **kwargs):
        values = {
            'trusted': False,
            'reason': 'no_complete_candidates',
            'selected_y': None,
            'predicted_y': 285.5,
            'loss_count': 1,
        }
        values.update(kwargs)
        return PhaseResult(**values)

    def test_trusted_phase_is_centered_and_marked_trusted(self):
        guidance = select_super8_crop_guidance(
            self.result(trusted=True, reason='tracked', selected_y=284.5),
            last_safe_center_y=270.0,
        )
        self.assertEqual(guidance['source'], 'trusted_phase')
        self.assertEqual(guidance['center_y'], 284.5)
        self.assertEqual(guidance['prediction_age'], 0)
        self.assertTrue(guidance['valid'])

    def test_short_detector_loss_uses_prediction_only_for_crop(self):
        guidance = select_super8_crop_guidance(
            self.result(predicted_y=286.25, loss_count=1),
            last_safe_center_y=280.0,
        )
        self.assertEqual(guidance['source'], 'predicted_phase')
        self.assertEqual(guidance['center_y'], 286.25)
        self.assertEqual(guidance['prediction_age'], 1)

    def test_second_loss_frame_remains_bounded_predicted_guidance(self):
        guidance = select_super8_crop_guidance(
            self.result(predicted_y=287.0, loss_count=2),
            last_safe_center_y=280.0,
        )
        self.assertEqual(guidance['source'], 'predicted_phase')
        self.assertEqual(guidance['prediction_age'], 2)

    def test_horizon_exhaustion_holds_last_safe_center(self):
        guidance = select_super8_crop_guidance(
            self.result(predicted_y=289.0, loss_count=3),
            last_safe_center_y=280.0,
        )
        self.assertEqual(guidance['source'], 'held_safe')
        self.assertEqual(guidance['center_y'], 280.0)
        self.assertEqual(guidance['fallback_reason'], 'no_complete_candidates')

    def test_reseed_observations_do_not_become_crop_center(self):
        guidance = select_super8_crop_guidance(
            self.result(
                reason='reseed_confirming',
                predicted_y=320.0,
                selected_y=320.0,
                loss_count=3,
            ),
            last_safe_center_y=280.0,
        )
        self.assertEqual(guidance['source'], 'held_safe')
        self.assertEqual(guidance['center_y'], 280.0)

    def test_no_safe_center_uses_explicit_full_preview(self):
        guidance = select_super8_crop_guidance(
            self.result(reason='reseed_started', predicted_y=None, loss_count=3),
        )
        self.assertEqual(guidance['source'], 'full_preview')
        self.assertIsNone(guidance['center_y'])
        self.assertFalse(guidance['valid'])

    def test_crop_guidance_does_not_change_phase_trust(self):
        result = self.result()
        select_super8_crop_guidance(result, last_safe_center_y=280.0)
        self.assertFalse(result.trusted)
        self.assertIsNone(result.selected_y)

    def test_recovery_candidate_is_display_only_crop_guidance(self):
        guidance = select_super8_crop_guidance(
            self.result(
                reason='recovery_confirming', selected_y=335.0,
                predicted_y=300.0, recovery_age=2,
            ),
            last_safe_center_y=280.0,
        )
        self.assertEqual(guidance['source'], 'recovery_phase')
        self.assertEqual(guidance['center_y'], 335.0)
        self.assertEqual(guidance['prediction_age'], 2)

if __name__ == '__main__':
    unittest.main()
