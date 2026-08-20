import unittest

from transport_calibration import AdaptiveTransportController


def controller(**overrides):
    values = dict(base_steps=280, pixels_per_step=1.0, correction_gain=1.0,
                  integral_gain=0.1, min_correction=-8, max_correction=8,
                  min_command=246, max_command=313, adaptation_frames=4,
                  warning_frames=3, warning_interval=10)
    values.update(overrides)
    return AdaptiveTransportController(**values)


class TransportCalibrationTests(unittest.TestCase):
    def test_sustained_negative_saturation_adapts_down_but_transient_does_not(self):
        c = controller()
        c.update(-100)
        self.assertEqual(c.adaptive_base_steps, 280)
        for _ in range(3):
            c.update(-100)
        self.assertEqual(c.adaptive_base_steps, 279)

    def test_sustained_positive_saturation_adapts_up(self):
        c = controller()
        for _ in range(4):
            c.update(100)
        self.assertEqual(c.adaptive_base_steps, 281)

    def test_anti_windup_and_safe_commands(self):
        c = controller()
        for _ in range(40):
            result = c.update(-1000)
            self.assertGreaterEqual(result.commanded_steps, 246)
            self.assertLessEqual(result.commanded_steps, 313)
        self.assertEqual(c.integral_steps, 0.0)

    def test_simulated_bias_moves_correction_toward_zero(self):
        c = controller(integral_gain=0.0, adaptation_frames=3)
        corrections = []
        # Plant error is the difference between the true 276-step pitch and base.
        for _ in range(24):
            error = 10 * (276 - c.adaptive_base_steps)
            corrections.append(c.update(error).correction)
        self.assertEqual(c.adaptive_base_steps, 276)
        self.assertEqual(corrections[:3], [-8, -8, -8])
        self.assertEqual(corrections[-1], 0)

    def test_state_restores_only_for_matching_calibration(self):
        c = controller()
        for _ in range(4):
            c.update(-100)
        state = c.state()
        resumed = controller(state=state)
        self.assertEqual(resumed.adaptive_base_steps, 279)
        fresh = controller(base_steps=281, state=state)
        self.assertEqual(fresh.adaptive_base_steps, 281)

    def test_warning_is_thresholded_and_rate_limited(self):
        c = controller()
        warnings = [c.update(-100).warning for _ in range(12)]
        self.assertEqual(sum(warnings), 1)

    def test_diagnostics_include_limits_and_observations(self):
        c = controller()
        c.update(-100)
        data = c.diagnostics()
        for key in ('initial_base_steps', 'final_adaptive_base_steps',
                    'min_correction', 'max_correction', 'min_motor_command',
                    'max_motor_command', 'observed_min_correction',
                    'observed_max_correction', 'negative_saturation_pct'):
            self.assertIn(key, data)


if __name__ == '__main__':
    unittest.main()
