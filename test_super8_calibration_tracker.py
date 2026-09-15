import json
from collections import defaultdict
from pathlib import Path
import unittest

from super8_calibration_tracker import (
    Super8PerforationTracker,
    TrackObservation,
    crossing_step,
    linear_slope,
    robust_values,
)


class Super8PerforationTrackerTests(unittest.TestCase):
    @staticmethod
    def candidate(y, x=278, width=190, height=236, area=43000, score=0.98):
        return {
            'center_x': x,
            'center_y': y,
            'width': width,
            'height': height,
            'area': area,
            'score': score,
            'classification': 'COMPLETE',
        }

    def test_replays_advancing_experiment_without_identity_substitution(self):
        source = Path(
            'experiments/super8_sprocket/moving_results/candidates.json'
        )
        rows = json.loads(source.read_text())
        frames = defaultdict(list)
        for row in rows:
            if row['classification'] == 'COMPLETE':
                frames[row['index']].append(row)

        tracker = Super8PerforationTracker(registration_y=760)
        for index in sorted(frames):
            result = tracker.update(frames[index], index * 10)
            self.assertTrue(result['accepted'], (index, result))

        measurements = tracker.measurements()
        self.assertEqual(len(tracker.tracks), 3)
        self.assertEqual(len(tracker.tracks[0].observations), 24)
        self.assertEqual(len(tracker.tracks[1].observations), 33)
        self.assertEqual(measurements['ambiguous_track_count'], 0)
        self.assertEqual(measurements['rejected_track_count'], 0)
        self.assertEqual(measurements['steps_per_pitch']['count'], 1)
        self.assertAlmostEqual(
            measurements['steps_per_pitch']['median'], 310.2561, places=3
        )
        self.assertAlmostEqual(
            measurements['sprocket_pitch_px']['median'], 842.852, places=3
        )
        self.assertAlmostEqual(
            measurements['steps_per_px']['median'], 0.368161, places=5
        )
        self.assertLess(
            measurements['cross_check']['difference_percent'], 0.1
        )

    def test_adjacent_hole_is_created_as_a_separate_track(self):
        tracker = Super8PerforationTracker(registration_y=500)
        tracker.update([self.candidate(600)], 0)
        result = tracker.update(
            [self.candidate(570), self.candidate(1413, x=270)], 10
        )

        self.assertTrue(result['accepted'])
        self.assertEqual(result['assignments'][0]['track_id'], 1)
        self.assertEqual(result['created_tracks'], [2])

    def test_ambiguous_assignment_requests_same_position_recapture(self):
        tracker = Super8PerforationTracker(
            registration_y=500,
            ambiguity_margin=8,
        )
        tracker.update([self.candidate(500)], 0)
        result = tracker.update(
            [self.candidate(470), self.candidate(473)], 10
        )

        self.assertFalse(result['accepted'])
        self.assertTrue(result['recapture_required'])
        self.assertEqual(len(tracker.tracks[0].observations), 1)

    def test_implausible_midframe_identity_jump_is_rejected(self):
        tracker = Super8PerforationTracker(registration_y=760)
        tracker.update([self.candidate(700)], 0)
        result = tracker.update([self.candidate(610)], 10)

        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'implausible_identity_jump')

    def test_regression_and_registration_crossing(self):
        observations = [
            TrackObservation(step, 278, 800 - 2.5 * step, 190, 236, 43000, 0.98)
            for step in (0, 10, 20, 30)
        ]
        self.assertAlmostEqual(
            linear_slope([(item.steps, item.center_y) for item in observations]),
            -2.5,
        )
        self.assertAlmostEqual(crossing_step(observations, 760), 16.0)

    def test_robust_filter_rejects_large_outlier(self):
        self.assertEqual(robust_values([309, 310, 310, 311, 700]), [309, 310, 310, 311])


if __name__ == '__main__':
    unittest.main()
