import unittest
from transport_calibration import AdaptiveTransportController, merge_calibration_settings, resolve_nominal_steps


def controller(**kw):
    args = dict(base_steps=277, pixels_per_step=1, correction_gain=1, integral_gain=0,
                min_correction=-8, max_correction=8, min_command=243, max_command=310,
                adaptation_frames=10000, bias_window_size=300, bias_cooldown_samples=100,
                warning_frames=15, warning_interval=100, bias_warning_interval=300)
    args.update(kw)
    return AdaptiveTransportController(**args)


class TransportCalibrationTests(unittest.TestCase):
    def test_99_percent_negative_without_limit_runs_adapts_down(self):
        c = controller()
        for i in range(300): result = c.update(1 if i == 150 else (-4-i % 4))
        self.assertEqual((c.adaptive_base_steps, c.rolling_adaptations_down,
                          c.exact_saturation_adaptations), (276, 1, 0))
        self.assertEqual(result.adaptation_reason, 'rolling_negative_bias')

    def test_symmetric_positive_bias_adapts_up(self):
        c = controller()
        for i in range(300): c.update(4+i % 4)
        self.assertEqual(c.adaptive_base_steps, 278)

    def test_centered_stop_band_does_not_adapt(self):
        c = controller()
        for i in range(600): c.update((-1, 0, 1)[i % 3])
        self.assertEqual(c.adaptive_base_steps, 277)

    def test_full_window_and_reliable_samples_required(self):
        c = controller()
        for _ in range(299): c.update(-5)
        for _ in range(50): c.update(-5, trusted=False)
        self.assertEqual((len(c.correction_window), c.adaptive_base_steps), (299, 277))
        c.update(-5)
        self.assertEqual(c.adaptive_base_steps, 276)

    def test_cooldown_prevents_rapid_repeat(self):
        c = controller()
        for _ in range(399): c.update(-5)
        self.assertEqual(c.adaptive_base_steps, 276)
        c.update(-5)
        self.assertEqual(c.adaptive_base_steps, 275)

    def test_safety_bound_and_no_double_adaptation(self):
        double = controller(adaptation_frames=300)
        for _ in range(300): double.update(-100)
        self.assertEqual(double.base_adaptations_down, 1)
        bounded = controller(base_steps=245, min_command=243)
        for _ in range(800): bounded.update(-100)
        self.assertGreaterEqual(bounded.adaptive_base_steps, 243)

    def test_anti_windup_at_output_limits(self):
        c = controller(integral_gain=.5)
        for _ in range(50): c.update(-1000)
        value = c.integral_steps
        for _ in range(50): c.update(-1000)
        self.assertEqual(c.integral_steps, value)
        edge = controller(base_steps=244, min_command=243, integral_gain=.5)
        for _ in range(20): self.assertGreaterEqual(edge.update(-100).commanded_steps, 243)

    def test_resume_nominal_change_and_legacy_state(self):
        c = controller(integral_gain=.1)
        for _ in range(10): c.update(2)
        resumed = controller(integral_gain=.1, state=c.state())
        self.assertEqual((resumed.restore_status, resumed.integral_steps),
                         ('same_nominal_restored', c.integral_steps))
        changed = controller(base_steps=272, integral_gain=.1, state=c.state())
        self.assertEqual((changed.adaptive_base_steps, changed.integral_steps,
                          changed.restore_status), (272, 0, 'configured_nominal_changed_reset'))
        legacy = controller(state={'version': 1, 'adaptive_base_steps': 270, 'integral_steps': -5})
        self.assertEqual((legacy.adaptive_base_steps, legacy.integral_steps,
                          legacy.restore_status), (277, 0, 'legacy_state_missing_nominal_reset'))

    def test_setting_precedence_fallback_and_legacy_field(self):
        merged = merge_calibration_settings({'steps_per_pitch': 277}, {'steps_per_pitch': 272})
        self.assertEqual(resolve_nominal_steps(merged), 272)
        self.assertEqual(resolve_nominal_steps({}), 280)
        self.assertEqual(resolve_nominal_steps({'steps_per_pitch_avg': 281}), 280)

    def test_warning_rate_limit_summary_and_simulation(self):
        c = controller(bias_warning_interval=100)
        warnings = [c.update(-5).bias_warning for _ in range(500)]
        self.assertEqual(sum(warnings), 3)
        self.assertEqual(c.diagnostics()['rolling_window']['sample_count'], 300)
        sim = controller()
        commands, corrections = [], []
        for _ in range(900):
            r = sim.update(-(sim.adaptive_base_steps-272))
            commands.append(r.commanded_steps); corrections.append(r.correction)
        self.assertLess(sim.adaptive_base_steps, 277)
        self.assertLess(abs(corrections[-1]), abs(corrections[0]))
        self.assertTrue(all(243 <= x <= 310 for x in commands))
        self.assertLessEqual(max(abs(b-a) for a, b in zip(commands, commands[1:])), 1)


if __name__ == '__main__': unittest.main()
