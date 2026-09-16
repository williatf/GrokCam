import unittest

from transport_calibration import calculate_super8_stage1_transport


class Super8Stage1TransportTests(unittest.TestCase):
    def calculate(self, error, trusted=True, min_command=271, max_command=345):
        return calculate_super8_stage1_transport(
            error_px=error,
            nominal_steps=308,
            pixels_per_step=1.0,
            trusted=trusted,
            correction_gain=0.25,
            dead_band_px=3.75,
            min_correction=-8,
            max_correction=8,
            min_command=min_command,
            max_command=max_command,
        )

    def test_positive_error_uses_existing_positive_polarity(self):
        result = self.calculate(20)
        self.assertEqual(result.requested_correction, 5)
        self.assertEqual(result.commanded_steps, 313)
        self.assertTrue(result.eligible)

    def test_deadband_disables_p_correction(self):
        result = self.calculate(3.75)
        self.assertTrue(result.deadband_active)
        self.assertEqual(result.p_contribution, 0.0)
        self.assertEqual(result.commanded_steps, 308)

    def test_correction_saturates_at_eight_steps(self):
        self.assertEqual(self.calculate(100).limited_correction, 8)
        self.assertEqual(self.calculate(-100).limited_correction, -8)
        self.assertEqual(self.calculate(100).requested_correction, 25)
        self.assertTrue(self.calculate(100).saturated)

    def test_untrusted_phase_falls_back_to_nominal_without_eligibility(self):
        result = self.calculate(100, trusted=False)
        self.assertFalse(result.eligible)
        self.assertEqual(result.commanded_steps, 308)
        self.assertEqual(result.applied_correction, 0)
        self.assertIsNone(result.error_px)

    def test_hard_command_bounds_override_correction(self):
        low = self.calculate(-100, min_command=304, max_command=312)
        high = self.calculate(100, min_command=304, max_command=312)
        self.assertEqual(low.commanded_steps, 304)
        self.assertEqual(high.commanded_steps, 312)
        self.assertEqual(low.applied_correction, -4)
        self.assertEqual(high.applied_correction, 4)
        self.assertTrue(low.saturated)
        self.assertTrue(high.saturated)


if __name__ == '__main__':
    unittest.main()
