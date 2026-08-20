"""Slow, bounded RAW transport calibration.

Only ``steps_per_pitch`` is current. The historical ``steps_per_pitch_avg``
field is intentionally not interpreted as the nominal setting.
"""
from collections import deque
from dataclasses import dataclass
import statistics


def merge_calibration_settings(calibration, config):
    return {**(calibration or {}), **(config or {})}


def resolve_nominal_steps(settings, fallback=280):
    return int(settings.get('steps_per_pitch', fallback))


@dataclass
class TransportResult:
    correction: int
    requested_steps: int
    commanded_steps: int
    saturation_warning: bool = False
    bias_warning: bool = False
    adaptation_reason: str = None
    bias_warning_status: str = None
    active_constraint: str = 'neither'

    @property
    def warning(self):
        return self.saturation_warning


class AdaptiveTransportController:
    STATE_VERSION = 2

    def __init__(self, base_steps, pixels_per_step, correction_gain=0.25,
                 integral_gain=0.02, min_correction=-8, max_correction=8,
                 min_command=None, max_command=None, adaptation_frames=30,
                 warning_frames=15, warning_interval=100, state=None,
                 bias_window_size=300, bias_cooldown_samples=100,
                 bias_median_threshold=4, bias_share_threshold=.80,
                 bias_stop_band=1, bias_warning_interval=300):
        self.configured_nominal_steps = int(base_steps)
        self.initial_base_steps = int(base_steps)
        self.adaptive_base_steps = int(base_steps)
        self.pixels_per_step = float(pixels_per_step)
        self.correction_gain = float(correction_gain)
        self.integral_gain = float(integral_gain)
        self.min_correction, self.max_correction = int(min_correction), int(max_correction)
        self.min_command = int(base_steps if min_command is None else min_command)
        self.max_command = int(base_steps if max_command is None else max_command)
        self.adaptation_frames = max(2, int(adaptation_frames))
        self.warning_frames, self.warning_interval = max(2, int(warning_frames)), max(1, int(warning_interval))
        self.bias_window_size = max(2, int(bias_window_size))
        self.bias_cooldown_samples = max(1, int(bias_cooldown_samples))
        self.bias_median_threshold = abs(float(bias_median_threshold))
        self.bias_share_threshold = float(bias_share_threshold)
        self.bias_stop_band = abs(float(bias_stop_band))
        self.bias_warning_interval = max(1, int(bias_warning_interval))
        self.integral_steps = 0.0
        self.correction_window = deque(maxlen=self.bias_window_size)
        self.cooldown_remaining = 0
        self.negative_saturation_run = self.positive_saturation_run = 0
        self.frames_evaluated = 0
        self.negative_saturated_frames = self.positive_saturated_frames = 0
        self.observed_min_correction = self.observed_max_correction = None
        self.base_adaptations_down = self.base_adaptations_up = 0
        self.rolling_adaptations_down = self.rolling_adaptations_up = 0
        self.exact_saturation_adaptations = self.integral_compensation_events = 0
        self.last_adaptation_reason = None
        self.integral_reset_reason = 'new_controller'
        self.last_warning_frame = self.last_bias_warning_frame = None
        self.restore_status = 'new_capture'
        if state:
            self.restore(state)

    def _limits(self):
        return (max(self.min_correction, self.min_command-self.adaptive_base_steps),
                min(self.max_correction, self.max_command-self.adaptive_base_steps))

    def rolling_statistics(self):
        v, n = list(self.correction_window), len(self.correction_window)
        if not v:
            return dict(sample_count=0, median_correction=None, negative_share=0., positive_share=0.,
                        min_correction=None, max_correction=None, negative_limit_share=0., positive_limit_share=0.)
        return dict(sample_count=n, median_correction=float(statistics.median(v)),
                    negative_share=sum(x < 0 for x in v)/n, positive_share=sum(x > 0 for x in v)/n,
                    min_correction=min(v), max_correction=max(v),
                    negative_limit_share=sum(x == self.min_correction for x in v)/n,
                    positive_limit_share=sum(x == self.max_correction for x in v)/n)

    def _clamp_integral(self, proportional):
        # Bound stored controller memory independently of a potentially huge
        # one-frame proportional term. This avoids back-calculating a large,
        # opposite-signed integral that would kick when the error disappears.
        self.integral_steps = max(float(self.min_correction),
                                  min(float(self.max_correction), self.integral_steps))

    def _adapt(self, direction, reason, proportional):
        new_base = self.adaptive_base_steps + direction
        if not self.min_command <= new_base <= self.max_command:
            self.last_adaptation_reason = reason + '_suppressed_safety_limit'
            return False
        self.adaptive_base_steps = new_base
        self.integral_steps -= direction  # bumpless: base +1, integral -1 (and vice versa)
        self._clamp_integral(proportional)
        self.integral_compensation_events += 1
        self.cooldown_remaining = self.bias_cooldown_samples
        self.last_adaptation_reason = reason
        if direction < 0:
            self.base_adaptations_down += 1
            self.rolling_adaptations_down += int(reason.startswith('rolling_'))
        else:
            self.base_adaptations_up += 1
            self.rolling_adaptations_up += int(reason.startswith('rolling_'))
        self.exact_saturation_adaptations += int(reason.startswith('exact_'))
        return True

    def update(self, error_px, trusted=True):
        if not trusted or error_px is None:
            self.negative_saturation_run = max(0, self.negative_saturation_run-1)
            self.positive_saturation_run = max(0, self.positive_saturation_run-1)
            command = max(self.min_command, min(self.max_command, self.adaptive_base_steps))
            return TransportResult(0, self.adaptive_base_steps, command)
        self.frames_evaluated += 1
        if self.cooldown_remaining:
            self.cooldown_remaining -= 1
        proportional = float(error_px)/self.pixels_per_step*self.correction_gain
        delta = float(error_px)/self.pixels_per_step*self.integral_gain
        low, high = self._limits()
        output = proportional + self.integral_steps
        if not ((output <= low and delta < 0) or (output >= high and delta > 0)):
            self.integral_steps += delta
        self._clamp_integral(proportional)
        raw = int(round(proportional+self.integral_steps))
        correction = max(low, min(high, raw))
        requested = self.adaptive_base_steps + correction
        command = max(self.min_command, min(self.max_command, requested))
        constraint = 'motor_command_limit' if command != requested else 'correction_limit' if correction != raw else 'neither'
        neg, pos = correction == self.min_correction, correction == self.max_correction
        self.negative_saturation_run = self.negative_saturation_run+1 if neg else 0
        self.positive_saturation_run = self.positive_saturation_run+1 if pos else 0
        self.negative_saturated_frames += int(neg); self.positive_saturated_frames += int(pos)
        self.observed_min_correction = correction if self.observed_min_correction is None else min(self.observed_min_correction, correction)
        self.observed_max_correction = correction if self.observed_max_correction is None else max(self.observed_max_correction, correction)
        self.correction_window.append(correction)
        run = max(self.negative_saturation_run, self.positive_saturation_run)
        sat_warning = run >= self.warning_frames and (self.last_warning_frame is None or self.frames_evaluated-self.last_warning_frame >= self.warning_interval)
        if sat_warning: self.last_warning_frame = self.frames_evaluated

        reason = None
        if self.cooldown_remaining == 0:  # shared gate ensures at most one adaptation
            if self.negative_saturation_run >= self.adaptation_frames:
                if self._adapt(-1, 'exact_negative_saturation', proportional):
                    reason, self.negative_saturation_run = self.last_adaptation_reason, 0
            elif self.positive_saturation_run >= self.adaptation_frames:
                if self._adapt(1, 'exact_positive_saturation', proportional):
                    reason, self.positive_saturation_run = self.last_adaptation_reason, 0
            elif len(self.correction_window) == self.bias_window_size:
                s, = (self.rolling_statistics(),)
                median = s['median_correction']
                if abs(median) > self.bias_stop_band:
                    negative = median <= -self.bias_median_threshold or s['negative_share'] >= self.bias_share_threshold
                    positive = median >= self.bias_median_threshold or s['positive_share'] >= self.bias_share_threshold
                    if negative and not positive and self._adapt(-1, 'rolling_negative_bias', proportional): reason = self.last_adaptation_reason
                    elif positive and not negative and self._adapt(1, 'rolling_positive_bias', proportional): reason = self.last_adaptation_reason

        self._clamp_integral(proportional)
        correction = max(self._limits()[0], min(self._limits()[1], int(round(proportional+self.integral_steps))))
        requested = self.adaptive_base_steps + correction
        command = max(self.min_command, min(self.max_command, requested))
        constraint = ('motor_command_limit' if command != requested else
                      'correction_limit' if correction != int(round(proportional+self.integral_steps)) else
                      'neither')
        s = self.rolling_statistics()
        bias = (s['sample_count'] == self.bias_window_size and abs(s['median_correction']) > self.bias_stop_band and
                (abs(s['median_correction']) >= self.bias_median_threshold or s['negative_share'] >= self.bias_share_threshold or s['positive_share'] >= self.bias_share_threshold))
        bias_warning = bias and (self.last_bias_warning_frame is None or self.frames_evaluated-self.last_bias_warning_frame >= self.bias_warning_interval)
        if bias_warning: self.last_bias_warning_frame = self.frames_evaluated
        status = ('adapted' if reason else 'suppressed_by_cooldown' if self.cooldown_remaining else 'suppressed_by_safety_limit') if bias else ('stop_band_or_no_bias' if s['sample_count'] == self.bias_window_size else None)
        return TransportResult(correction, requested, command, sat_warning, bias_warning, reason, status, constraint)

    def state(self):
        payload = {name: getattr(self, name) for name in (
            'configured_nominal_steps','initial_base_steps','adaptive_base_steps','integral_steps','cooldown_remaining',
            'negative_saturation_run','positive_saturation_run','frames_evaluated','negative_saturated_frames',
            'positive_saturated_frames','observed_min_correction','observed_max_correction','base_adaptations_down',
            'base_adaptations_up','rolling_adaptations_down','rolling_adaptations_up','exact_saturation_adaptations',
            'last_adaptation_reason','integral_compensation_events','integral_reset_reason','last_warning_frame','last_bias_warning_frame')}
        payload.update(version=self.STATE_VERSION, correction_window=list(self.correction_window))
        return payload

    def restore(self, state):
        saved = state.get('configured_nominal_steps')
        if saved is None:
            self.restore_status, self.integral_reset_reason = 'legacy_state_missing_nominal_reset', 'legacy_state_missing_nominal'; return False
        if int(saved) != self.configured_nominal_steps:
            self.restore_status, self.integral_reset_reason = 'configured_nominal_changed_reset', 'configured_nominal_changed'; return False
        if int(state.get('version', 0)) != self.STATE_VERSION:
            self.restore_status, self.integral_reset_reason = 'state_version_changed_reset', 'state_version_changed'; return False
        for name in ('adaptive_base_steps','cooldown_remaining','negative_saturation_run','positive_saturation_run','frames_evaluated',
                     'negative_saturated_frames','positive_saturated_frames','base_adaptations_down','base_adaptations_up',
                     'rolling_adaptations_down','rolling_adaptations_up','exact_saturation_adaptations','integral_compensation_events'):
            setattr(self, name, int(state.get(name, getattr(self, name))))
        self.adaptive_base_steps = max(self.min_command, min(self.max_command, self.adaptive_base_steps))
        self.integral_steps = float(state.get('integral_steps', 0.))
        self.correction_window.extend(int(x) for x in state.get('correction_window', []))
        for name in ('observed_min_correction','observed_max_correction','last_adaptation_reason','last_warning_frame','last_bias_warning_frame'):
            setattr(self, name, state.get(name))
        self.integral_reset_reason = state.get('integral_reset_reason')
        self.restore_status = 'same_nominal_restored'
        return True

    def diagnostics(self):
        n = self.frames_evaluated
        return {**self.state(), 'restore_status': self.restore_status, 'final_adaptive_base_steps': self.adaptive_base_steps,
                'min_correction': self.min_correction, 'max_correction': self.max_correction,
                'min_motor_command': self.min_command, 'max_motor_command': self.max_command,
                'rolling_window': self.rolling_statistics(),
                'negative_saturation_pct': round(100*self.negative_saturated_frames/n, 3) if n else 0.,
                'positive_saturation_pct': round(100*self.positive_saturated_frames/n, 3) if n else 0.}
