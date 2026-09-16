import sys
import types
import unittest
from unittest import mock


# The GPIO module is Pi-only; importing the control class with a stub keeps
# this test hardware-free while exercising the telemetry boundary.
sys.modules.setdefault('wiringpi', types.SimpleNamespace())
import control


class TakeupTelemetryTests(unittest.TestCase):
    def make_control(self):
        tc = control.tcControl.__new__(control.tcControl)
        tc.takeup_pulse_sequence = 0
        tc.TAKEUP_REEL_PIN = 107
        tc.TAKEUP_INTERVAL = 2500
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


if __name__ == '__main__':
    unittest.main()
