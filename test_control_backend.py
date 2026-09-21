import unittest
from unittest import mock

import control


class FakeTransportDevice:
    """Logical device fake; transport tests do not model SPI registers."""

    def __init__(self):
        self.configured_outputs = []
        self.states = {}
        self.writes = []
        self.closed = False

    def configure_output(self, pin):
        self.configured_outputs.append(pin)

    def write(self, pin, value):
        pin = int(pin)
        value = int(bool(value))
        self.writes.append((pin, value))
        self.states[pin] = value

    def close(self):
        self.closed = True


class TcControlTransportTests(unittest.TestCase):
    def make_control(self):
        device = FakeTransportDevice()
        with mock.patch.object(control, 'MCP23S17', return_value=device):
            tc = control.tcControl()
        return tc, device

    @staticmethod
    def count_writes(device, pin, value):
        native_pin = pin - 100
        return sum(1 for written_pin, written_value in device.writes
                   if written_pin == native_pin and written_value == value)

    def test_legacy_pins_are_configured_as_native_outputs(self):
        _, device = self.make_control()
        self.assertEqual(device.configured_outputs, list(range(11)))

    def test_initial_outputs_match_controller_defaults(self):
        tc, device = self.make_control()
        self.assertEqual(device.states[100 - 100], 0)  # pusher enable
        self.assertEqual(device.states[102 - 100], 1)  # pusher direction
        self.assertEqual(device.states.get(104 - 100, 0), 0)  # puller step idle
        self.assertEqual(device.states[105 - 100], 0)  # puller enable
        self.assertEqual(device.states[103 - 100], 1)  # puller direction
        self.assertEqual(device.states[108 - 100], 0)  # LED
        self.assertEqual(tc.PUSHER_RATIO, 0.98)

    def test_forward_steps_keep_puller_and_ratioed_pusher_behavior(self):
        tc, device = self.make_control()
        device.writes.clear()
        with mock.patch.object(control.time, 'sleep'):
            tc.steps_forward(50)
        self.assertEqual(self.count_writes(device, 104, 1), 50)
        self.assertEqual(self.count_writes(device, 104, 0), 50)
        self.assertEqual(self.count_writes(device, 101, 1), 48)
        self.assertEqual(self.count_writes(device, 101, 0), 48)

    def test_single_forward_step_has_one_puller_step_and_no_pusher_step(self):
        tc, device = self.make_control()
        device.writes.clear()
        with mock.patch.object(control.time, 'sleep'):
            tc.steps_forward(1)
        self.assertEqual(self.count_writes(device, 104, 1), 1)
        self.assertEqual(self.count_writes(device, 104, 0), 1)
        self.assertEqual(self.count_writes(device, 101, 1), 0)

    def test_normal_reel_schedule_keeps_feed_and_takeup_remainders(self):
        tc, _ = self.make_control()
        tc.feed_steps_taken = 4_999
        tc.takeup_steps_taken = 2_499
        self.assertEqual(tc._schedule_reel_pulses(2), (1, 1))
        self.assertEqual(tc.feed_steps_taken, 1)
        self.assertEqual(tc.takeup_steps_taken, 1)

    def test_back_steps_restore_forward_directions(self):
        tc, device = self.make_control()
        device.writes.clear()
        with mock.patch.object(control.time, 'sleep'):
            tc.steps_back(3)
        self.assertEqual(device.states[102 - 100], 1)
        self.assertEqual(device.states[103 - 100], 1)
        self.assertEqual(self.count_writes(device, 104, 1), 3)
        self.assertEqual(self.count_writes(device, 101, 1), 2)

    def test_cleanup_disables_outputs_and_closes_device(self):
        tc, device = self.make_control()
        tc.clean_up()
        tc.clean_up()
        self.assertTrue(device.closed)
        self.assertEqual(device.states[108 - 100], 0)
        self.assertEqual(device.states[106 - 100], 0)
        self.assertEqual(device.states[107 - 100], 0)
        self.assertEqual(device.states[100 - 100], 0)


if __name__ == '__main__':
    unittest.main()
