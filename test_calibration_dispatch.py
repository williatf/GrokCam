import os
import tempfile
import unittest

from calibration_dispatch import PendingCalibrationProposal, calibration_route


class CalibrationDispatchTests(unittest.TestCase):
    def test_format_destinations_are_isolated(self):
        regular = calibration_route('regular8', '/tmp/calibration-test')
        super8 = calibration_route('super8', '/tmp/calibration-test')
        self.assertTrue(regular['destination'].endswith('calibration.json'))
        self.assertTrue(super8['destination'].endswith('calibration.super8.json'))
        self.assertNotEqual(regular['destination'], super8['destination'])

    def test_pending_proposal_rejects_stale_project_or_format(self):
        with tempfile.TemporaryDirectory() as directory:
            route = calibration_route('super8', directory)
            pending = PendingCalibrationProposal.create(
                os.path.join(directory, 'project-a'),
                'super8',
                route['destination'],
                route['mode'],
                {'calibration_version': 2},
            )
            valid, reason = pending.validate(
                os.path.join(directory, 'project-b'),
                'super8',
                route['destination'],
                route['mode'],
            )
            self.assertFalse(valid)
            self.assertEqual(reason, 'active_project_changed_since_calibration_sweep')

            regular = calibration_route('regular8', directory)
            valid, reason = pending.validate(
                os.path.join(directory, 'project-a'),
                'regular8',
                regular['destination'],
                regular['mode'],
            )
            self.assertFalse(valid)
            self.assertEqual(reason, 'film_format_changed_since_calibration_sweep')


if __name__ == '__main__':
    unittest.main()
