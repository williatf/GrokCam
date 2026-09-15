import unittest

from super8_calibration_service import Super8CalibrationService


class FakeTransport:
    def __init__(self):
        self.total_steps = 0

    def steps_forward(self, steps):
        self.total_steps += steps


class Super8CalibrationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_crossing_resets_per_transition_watchdog(self):
        transport = FakeTransport()
        service = Super8CalibrationService(
            camera=None,
            tc=transport,
            detector=None,
            calibration_resolution=(2028, 1520),
            search_steps_per_pitch=100,
        )

        def capture_candidates():
            candidates = []
            for offset in range(0, 701, 100):
                y = 900 + offset - transport.total_steps
                if 100 <= y <= 1420:
                    candidates.append({
                        'center_x': 278,
                        'center_y': y,
                        'width': 190,
                        'height': 236,
                        'area': 43000,
                        'score': 0.98,
                    })
            return object(), candidates

        service._capture_candidates = capture_candidates
        progress_events = []

        async def record_progress(progress):
            progress_events.append(progress)

        result = await service.run_calibration(
            target_transitions=5,
            step_size=10,
            max_steps_per_transition=500,
            settle_delay=0,
            progress_callback=record_progress,
        )

        self.assertTrue(result['valid'], result)
        self.assertEqual(result['measurements']['steps_per_pitch']['count'], 5)
        self.assertGreater(result['total_motor_steps'], 150)
        self.assertTrue(any(event['progress_units'] > 0 for event in progress_events))
        self.assertEqual(progress_events[-1]['target_progress_units'], 6)


if __name__ == '__main__':
    unittest.main()
