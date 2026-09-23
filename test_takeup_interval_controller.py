import unittest

from takeup_interval_controller import AdaptiveTakeupIntervalController


class AdaptiveTakeupIntervalControllerTests(unittest.TestCase):
    def controller(self):
        return AdaptiveTakeupIntervalController(filter_window=1)

    def test_initial_interval_and_limits(self):
        c = self.controller()
        self.assertEqual(c.interval_frames, 12)
        self.assertEqual((c.min_interval, c.max_interval), (8, 32))

    def test_low_disturbance_decreases_by_one_and_other_thresholds_remain(self):
        c = self.controller()
        self.assertEqual(c.decide(1, 4.9, trusted=True).adaptation_delta, -1)
        self.assertEqual(c.interval_frames, 11)
        c.reset()
        self.assertEqual(c.decide(2, 7.0, trusted=True).adaptation_delta, 0)
        self.assertEqual(c.decide(3, 15.0, trusted=True).adaptation_delta, 2)
        self.assertEqual(c.decide(4, 20.0, trusted=True).adaptation_delta, 4)

    def test_repeated_low_disturbance_reaches_minimum(self):
        c = self.controller()
        intervals = []
        for sequence in range(1, 6):
            intervals.append(c.decide(sequence, 4.0, trusted=True).interval_after)
        self.assertEqual(intervals, [11, 10, 9, 8, 8])
        self.assertEqual(c.interval_frames, 8)

    def test_low_disturbance_cannot_decrease_below_minimum(self):
        c = self.controller()
        c.interval_frames = 8
        decision = c.decide(1, 0.0, trusted=True)
        self.assertEqual(decision.interval_after, 8)
        self.assertEqual(decision.adaptation_delta, 0)

    def test_upward_adaptation_behavior_remains_unchanged(self):
        c = self.controller()
        self.assertEqual(c.decide(1, 7.0, trusted=True).adaptation_delta, 0)
        self.assertEqual(c.decide(2, 15.0, trusted=True).adaptation_delta, 2)
        self.assertEqual(c.decide(3, 20.0, trusted=True).adaptation_delta, 4)

    def test_plausible_untrusted_loss_increases_four(self):
        c = self.controller()
        decision = c.decide(
            1, 42.0, trusted=False, plausible_phase_loss=True,
        )
        self.assertEqual(decision.adaptation_delta, 4)
        self.assertEqual(decision.adaptation_reason, 'plausible_takeup_phase_loss')

    def test_ambiguous_or_unavailable_evidence_does_not_adapt(self):
        c = self.controller()
        self.assertEqual(c.decide(1, None, trusted=False).adaptation_delta, 0)
        self.assertEqual(c.decide(2, 25.0, trusted=False).adaptation_delta, 0)
        self.assertEqual(c.interval_frames, 12)

    def test_maximum_is_bounded(self):
        c = self.controller()
        c.interval_frames = 31
        decision = c.decide(1, 100.0, trusted=True)
        self.assertEqual(decision.interval_after, 32)
        decision = c.decide(2, 100.0, trusted=True)
        self.assertEqual(decision.interval_after, 32)
        self.assertEqual(decision.adaptation_delta, 0)

    def test_one_decision_per_pulse_and_plus_two_is_diagnostic_only(self):
        c = self.controller()
        first = c.decide(7, 20.0, trusted=True)
        before = first.interval_after
        c.interval_frames = first.interval_after
        duplicate = c.decide(7, 100.0, trusted=True)
        self.assertEqual(duplicate.adaptation_delta, 0)
        self.assertEqual(duplicate.adaptation_reason, 'already_adapted')
        self.assertEqual(c.interval_frames, before)

    def test_reset_starts_a_fresh_capture_at_twelve(self):
        c = self.controller()
        decision = c.decide(1, 20.0, trusted=True)
        c.reset()
        self.assertEqual(c.interval_frames, 12)

    def test_regular8_tuning_starts_near_legacy_frame_cadence(self):
        c = AdaptiveTakeupIntervalController(
            initial_interval=10, min_interval=8, max_interval=32,
        )
        self.assertEqual(c.interval_frames, 10)
        self.assertEqual((c.min_interval, c.max_interval), (8, 32))

    def test_regular8_disturbance_can_lengthen_with_same_controller(self):
        c = AdaptiveTakeupIntervalController(
            initial_interval=10, min_interval=8, max_interval=32,
            filter_window=1,
        )
        decision = c.decide(1, 20.0, trusted=True)
        self.assertEqual(decision.interval_after, 14)
        self.assertEqual(decision.adaptation_delta, 4)
        self.assertLessEqual(c.interval_frames, 32)


if __name__ == '__main__':
    unittest.main()
