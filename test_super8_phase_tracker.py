import unittest

from super8_phase_tracker import Super8PhaseTracker


class Super8PhaseTrackerTests(unittest.TestCase):
    @staticmethod
    def candidate(y, score=0.98):
        return {
            'center_x': 278,
            'center_y': y,
            'width': 190,
            'height': 236,
            'area': 43000,
            'score': score,
            'classification': 'COMPLETE',
        }

    def tracker(self):
        return Super8PhaseTracker(
            pixels_per_step=1.0,
            preview_size=(760, 570),
            motion_direction=-1,
        )

    def test_normal_phase_continuity_uses_applied_steps(self):
        tracker = self.tracker()
        self.assertTrue(tracker.update([self.candidate(500)], 0).trusted)
        result = tracker.update([self.candidate(480), self.candidate(200)], 20)
        self.assertTrue(result.trusted)
        self.assertEqual(result.selected_y, 480)
        self.assertAlmostEqual(result.predicted_y, 480)

    def test_adjacent_hole_is_rejected_by_phase_gate(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 0)
        result = tracker.update([self.candidate(449)], 20)
        self.assertFalse(result.trusted)
        self.assertEqual(result.reason, 'candidate_outside_phase_gate')

    def test_close_candidates_are_ambiguous(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 0)
        result = tracker.update([self.candidate(480), self.candidate(486)], 20)
        self.assertFalse(result.trusted)
        self.assertEqual(result.reason, 'ambiguous_phase')
        self.assertAlmostEqual(result.ambiguity_px, 6.0)

    def test_loss_requires_three_frame_controlled_reseed(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 0)
        for steps in (20, 40, 60):
            result = tracker.update([], steps)
            self.assertFalse(result.trusted)
        self.assertEqual(result.loss_count, 3)

        self.assertEqual(
            tracker.update([self.candidate(300)], 80).reason,
            'reseed_started',
        )
        self.assertFalse(tracker.update([self.candidate(290)], 90).trusted)
        self.assertTrue(
            tracker.update([self.candidate(280)], 100).trusted
        )
        self.assertEqual(tracker.reseed_count, 1)


if __name__ == '__main__':
    unittest.main()
