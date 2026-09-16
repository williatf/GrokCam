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
            expected_sprocket_pitch_px=20.0,
            preview_size=(760, 570),
            motion_direction=-1,
        )

    def test_normal_phase_continuity_uses_applied_steps(self):
        tracker = self.tracker()
        self.assertTrue(tracker.update([self.candidate(500)], 0).trusted)
        result = tracker.update([self.candidate(500), self.candidate(200)], 20)
        self.assertTrue(result.trusted)
        self.assertEqual(result.selected_y, 500)
        self.assertAlmostEqual(result.predicted_y, 500)

    def test_adjacent_hole_is_rejected_by_phase_gate(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 0)
        result = tracker.update([self.candidate(820)], 20)
        self.assertFalse(result.trusted)
        self.assertEqual(result.reason, 'candidate_outside_phase_gate')

    def test_close_candidates_are_ambiguous(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 0)
        result = tracker.update([self.candidate(500), self.candidate(506)], 20)
        self.assertFalse(result.trusted)
        self.assertEqual(result.reason, 'ambiguous_phase')
        self.assertAlmostEqual(result.ambiguity_px, 6.0)

    def test_loss_requires_three_frame_controlled_reseed(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 0)
        for steps in (20, 20, 20):
            result = tracker.update([], steps)
            self.assertFalse(result.trusted)
        self.assertEqual(result.loss_count, 3)

        self.assertEqual(
            tracker.update([self.candidate(300)], 20).reason,
            'reseed_started',
        )
        self.assertFalse(tracker.update([self.candidate(300)], 20).trusted)
        self.assertTrue(
            tracker.update([self.candidate(300)], 20).trusted
        )
        self.assertEqual(tracker.reseed_count, 1)

    def test_untrusted_frame_advances_prediction_without_accumulating_steps(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        lost = tracker.update([], 20)
        self.assertFalse(lost.trusted)
        recovered = tracker.update([self.candidate(500)], 20)
        self.assertTrue(recovered.trusted)
        self.assertAlmostEqual(recovered.predicted_y, 500.0)

    def test_multiple_untrusted_frames_accumulate_prediction_exactly_once(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        first = tracker.update([], 19)
        second = tracker.update([], 19)
        self.assertAlmostEqual(first.predicted_y, 499.0)
        self.assertAlmostEqual(second.predicted_y, 498.0)
        self.assertAlmostEqual(tracker.predicted_y, 498.0)

    def test_coherent_recovery_preserves_epoch_and_delays_trust(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        results = [
            tracker.update([self.candidate(535, score=0.1)], 20),
            tracker.update([self.candidate(536, score=0.1)], 20),
            tracker.update([self.candidate(537, score=0.1)], 20),
        ]
        self.assertEqual(
            [result.reason for result in results],
            ['recovery_started', 'recovery_confirming', 'recovery_established'],
        )
        self.assertTrue(all(not result.trusted for result in results))
        self.assertEqual(tracker.phase_epoch, 0)
        self.assertTrue(tracker.update([self.candidate(537)], 20).trusted)

    def test_recovery_uses_geometry_not_detector_score(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        poor_geometry = self.candidate(560, score=0.999)
        poor_geometry.update({'center_x': 340, 'width': 80, 'height': 120, 'area': 90000})
        result = tracker.update([
            self.candidate(535, score=0.1), poor_geometry,
        ], 20)
        self.assertEqual(result.reason, 'recovery_started')
        self.assertEqual(result.selected_candidate_index, 0)

    def test_ambiguous_recovery_does_not_guess(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        result = tracker.update([
            self.candidate(535, score=0.1), self.candidate(538, score=0.99),
        ], 20)
        self.assertFalse(result.trusted)
        self.assertNotEqual(result.reason, 'recovery_started')

    def test_recovery_beyond_gate_falls_back_to_existing_reseed_path(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        results = [tracker.update([], 20) for _ in range(3)]
        self.assertEqual(tracker.update([self.candidate(700)], 20).reason, 'reseed_started')
        self.assertTrue(all(not result.trusted for result in results))

    def test_genuine_reseed_increments_epoch(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        for _ in range(3):
            tracker.update([], 20)
        tracker.update([self.candidate(700)], 20)
        tracker.update([self.candidate(700)], 20)
        result = tracker.update([self.candidate(700)], 20)
        self.assertTrue(result.trusted)
        self.assertEqual(result.reason, 'reseeded')
        self.assertEqual(result.phase_epoch, 1)

    def test_recovery_horizon_exhaustion_is_untrusted_and_bounded(self):
        tracker = Super8PhaseTracker(
            pixels_per_step=1.0, expected_sprocket_pitch_px=20.0,
            preview_size=(760, 570), motion_direction=-1,
            recovery_horizon=2, reseed_confirmations=3,
        )
        tracker.update([self.candidate(500)], 20)
        self.assertEqual(
            tracker.update([self.candidate(535)], 20).reason,
            'recovery_started',
        )
        result = tracker.update([self.candidate(536)], 20)
        self.assertEqual(result.reason, 'recovery_horizon_exhausted')
        self.assertFalse(result.trusted)

    def test_candidate_diagnostics_include_replay_geometry_and_association(self):
        tracker = self.tracker()
        result = tracker.update([self.candidate(500), self.candidate(540)], 0)
        diagnostic = result.candidate_diagnostics[0]
        self.assertEqual(diagnostic['candidate_index'], 0)
        self.assertEqual(diagnostic['association_status'], 'selected')
        for key in ('center_x', 'center_y', 'x1', 'y1', 'x2', 'y2',
                    'width', 'height', 'area', 'score',
                    'inside_trusted_gate', 'inside_recovery_gate'):
            self.assertIn(key, diagnostic)

    def test_prolonged_loss_without_candidates_remains_safe_and_does_not_crash(self):
        tracker = self.tracker()
        tracker.update([self.candidate(500)], 20)
        results = [tracker.update([], 20) for _ in range(5)]
        self.assertTrue(all(not result.trusted for result in results))
        self.assertEqual(results[-1].reason, 'no_complete_candidates')
        self.assertEqual(results[-1].loss_count, 5)

    def test_formal_calibration_nominal_residual_is_about_point_54_px(self):
        from film_calibration import load_film_calibration, resolve_raw_preview_transport

        calibration = load_film_calibration('super8')
        geometry = resolve_raw_preview_transport(calibration, (760, 570))
        tracker = Super8PhaseTracker(
            pixels_per_step=geometry['preview_pixels_per_step'],
            expected_sprocket_pitch_px=geometry['preview_sprocket_pitch_px'],
            preview_size=(760, 570),
            motion_direction=-1,
        )
        seed_y = 198.7868747
        expected_y = seed_y + (
            308 * geometry['preview_pixels_per_step']
            - geometry['preview_sprocket_pitch_px']
        )
        tracker.update([self.candidate(seed_y)], 308)
        result = tracker.update([self.candidate(expected_y)], 308)
        self.assertTrue(result.trusted)
        self.assertAlmostEqual(result.predicted_y, 199.3276, places=3)
        self.assertAlmostEqual(
            result.error_px, 0.0, places=3,
        )

    def test_adjacent_hole_transition_is_rejected_but_equivalent_candidate_tracks(self):
        tracker = self.tracker()
        tracker.update([self.candidate(200)], 20)
        result = tracker.update(
            [self.candidate(200), self.candidate(520, score=0.99)], 20
        )
        self.assertTrue(result.trusted)
        self.assertEqual(result.selected_y, 200)
        result = tracker.update([self.candidate(520)], 20)
        self.assertFalse(result.trusted)
        self.assertEqual(result.reason, 'candidate_outside_phase_gate')

    def test_repeated_nominal_steps_do_not_accumulate_as_cumulative_motion(self):
        tracker = Super8PhaseTracker(
            pixels_per_step=1.0407885250801314,
            expected_sprocket_pitch_px=320.02211234782806,
            preview_size=(760, 570),
            motion_direction=-1,
        )
        y = 198.7868747
        self.assertTrue(tracker.update([self.candidate(y)], 308).trusted)
        for _ in range(99):
            y += 308 * 1.0407885250801314 - 320.02211234782806
            result = tracker.update([self.candidate(y)], 308)
            self.assertTrue(result.trusted)
        self.assertLess(abs(result.predicted_y - y), 1e-6)

    def test_frames_45_to_50_choose_equivalent_phase_candidates(self):
        tracker = Super8PhaseTracker(
            pixels_per_step=1.0407885250801314,
            expected_sprocket_pitch_px=320.02211234782806,
            preview_size=(760, 570),
            motion_direction=-1,
        )
        tracker.update([self.candidate(407.0)], 308)
        raw_values = (413.42, 98.64, 421.85, 421.75, 421.96, 101.06)
        for raw_y in raw_values:
            candidates = [self.candidate(raw_y)]
            equivalent_y = raw_y + 320.02211234782806
            if equivalent_y <= 570:
                candidates.append(self.candidate(equivalent_y, score=0.5))
            equivalent_y = raw_y - 320.02211234782806
            if equivalent_y >= 0:
                candidates.append(self.candidate(equivalent_y, score=0.5))
            result = tracker.update(candidates, 308)
            self.assertTrue(result.trusted)
            self.assertLess(abs(result.selected_y - result.predicted_y), 30.0)


if __name__ == '__main__':
    unittest.main()
