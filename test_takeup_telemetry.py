import unittest
from unittest import mock


# Importing control remains hardware-free because the transport is only opened
# when tcControl is constructed; these tests exercise the telemetry boundary
# with a manually constructed controller.
import control


class TakeupTelemetryTests(unittest.TestCase):
    def make_control(self):
        tc = control.tcControl.__new__(control.tcControl)
        tc.takeup_pulse_sequence = 0
        tc.feed_steps_taken = 0
        tc._adaptive_takeup_enabled = False
        tc._takeup_frame_counter = 0
        tc._takeup_interval_frames = None
        tc.TAKEUP_REEL_PIN = 107
        tc.TAKEUP_INTERVAL = 2500
        tc.FEED_INTERVAL = 5000
        tc.TAKEUP_PULSE_DURATION = 0.2
        tc.ADVANCE_SETTLE_DELAY = 0
        tc.POST_TAKEUP_SETTLE_DELAY = 0
        tc._last_takeup_telemetry = {}
        tc.pulse_reel = mock.Mock()
        return tc

    def test_takeup_pulse_snapshot_contains_command_and_sequence(self):
        tc = self.make_control()
        with mock.patch.object(control.time, 'sleep'):
            tc._run_deferred_reel_pulses(308, 0, 1)

        telemetry = tc.get_last_takeup_telemetry()
        self.assertTrue(telemetry['takeup_active'])
        self.assertTrue(telemetry['takeup_pulse_started'])
        self.assertEqual(telemetry['takeup_pulse_sequence'], 1)
        self.assertEqual(telemetry['takeup_command'], {
            'pulse_count': 1, 'steps': 2500,
        })
        self.assertEqual(telemetry['takeup_motor_steps'], 2500)
        self.assertIsNone(telemetry['takeup_motor_direction'])
        self.assertIsInstance(telemetry['takeup_timestamp'], int)
        self.assertGreaterEqual(telemetry['takeup_pulse_duration'], 0.0)
        self.assertEqual(tc.pulse_reel.call_args.args[1], tc.TAKEUP_PULSE_DURATION)
        tc.pulse_reel.assert_called_once()

    def test_advance_without_pulse_is_distinguishable(self):
        tc = self.make_control()
        with mock.patch.object(control.time, 'sleep'):
            tc._run_deferred_reel_pulses(308, 0, 0)

        telemetry = tc.get_last_takeup_telemetry()
        self.assertFalse(telemetry['takeup_active'])
        self.assertFalse(telemetry['takeup_pulse_started'])
        self.assertEqual(telemetry['takeup_pulse_sequence'], 0)
        self.assertIsNone(telemetry['takeup_command'])
        self.assertEqual(telemetry['takeup_motor_steps'], 0)

    def test_sequence_advances_once_per_pulse(self):
        tc = self.make_control()
        with mock.patch.object(control.time, 'sleep'):
            tc._run_deferred_reel_pulses(308, 0, 2)

        telemetry = tc.get_last_takeup_telemetry()
        self.assertEqual(telemetry['takeup_pulse_sequence'], 2)
        self.assertEqual(telemetry['takeup_command']['pulse_count'], 2)
        self.assertEqual(tc.pulse_reel.call_count, 2)

    def test_adaptive_takeup_schedule_preserves_feed_schedule(self):
        tc = self.make_control()
        tc.feed_steps_taken = 4999
        tc.begin_takeup_capture(12)
        feed_pulses, takeup_pulses = tc._schedule_reel_pulses(2)
        self.assertEqual(feed_pulses, 1)
        self.assertEqual(takeup_pulses, 0)

    def test_adaptive_interval_counts_capture_frames(self):
        tc = self.make_control()
        tc.begin_takeup_capture(12)
        pulses = [tc._schedule_reel_pulses(308)[1] for _ in range(12)]
        self.assertEqual(sum(pulses), 1)
        self.assertEqual(pulses[-1], 1)

    def test_regular8_interval_counts_ten_capture_frames(self):
        tc = self.make_control()
        tc.takeup_steps_taken = 2_499
        tc.begin_takeup_capture(10)
        pulses = [tc._schedule_reel_pulses(308)[1] for _ in range(10)]
        self.assertEqual(sum(pulses), 1)
        self.assertEqual(pulses[-1], 1)
        self.assertEqual(tc.takeup_steps_taken, 0)

    def test_end_adaptive_capture_clears_legacy_accumulator(self):
        tc = self.make_control()
        tc.takeup_steps_taken = 1_234
        tc.begin_takeup_capture(10)
        tc.end_takeup_capture()
        self.assertEqual(tc.takeup_steps_taken, 0)
        self.assertFalse(tc._adaptive_takeup_enabled)


if __name__ == '__main__':
    unittest.main()
