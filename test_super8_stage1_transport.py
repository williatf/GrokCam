import unittest

from film_calibration import load_film_calibration, resolve_raw_preview_transport
from transport_calibration import calculate_super8_stage1_transport


class Super8Stage1TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calibration = load_film_calibration('super8')
        cls.transport = resolve_raw_preview_transport(calibration, (760, 570))
        cls.nominal = cls.transport['steps_per_pitch']
        cls.pixels_per_step = cls.transport['preview_pixels_per_step']

    def test_formal_calibration_is_version_two_at_308_steps(self):
        self.assertEqual(self.nominal, 308)
        self.assertEqual(load_film_calibration('super8').values['calibration_version'], 2)

    def calculate(self, error, trusted=True, min_command=271, max_command=344):
        return calculate_super8_stage1_transport(
            error_px=error,
            nominal_steps=self.nominal,
            pixels_per_step=self.pixels_per_step,
            trusted=trusted,
            correction_gain=0.25,
            dead_band_px=3.75,
            min_correction=-24,
            max_correction=24,
            min_command=min_command,
            max_command=max_command,
        )

    def test_positive_error_uses_existing_positive_polarity(self):
        result = self.calculate(20)
        self.assertGreater(result.commanded_steps, self.nominal)
        self.assertTrue(result.eligible)

    def test_zero_and_negative_errors_have_expected_commands(self):
        self.assertEqual(self.calculate(0).commanded_steps, self.nominal)
        self.assertLess(self.calculate(-20).commanded_steps, self.nominal)

    def test_deadband_disables_p_correction(self):
        result = self.calculate(3.75)
        self.assertTrue(result.deadband_active)
        self.assertEqual(result.p_contribution, 0.0)
        self.assertEqual(result.commanded_steps, self.nominal)

    def test_correction_saturates_at_24_steps(self):
        self.assertEqual(self.calculate(100).limited_correction, 24)
        self.assertEqual(self.calculate(-100).limited_correction, -24)
        self.assertGreater(self.calculate(150).requested_correction, 24)
        self.assertTrue(self.calculate(150).saturated)

    def test_untrusted_phase_falls_back_to_nominal_without_eligibility(self):
        result = self.calculate(100, trusted=False)
        self.assertFalse(result.eligible)
        self.assertEqual(result.commanded_steps, self.nominal)
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
