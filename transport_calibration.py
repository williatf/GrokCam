"""Slow, bounded transport calibration for the RAW capture pipeline."""

from dataclasses import dataclass


@dataclass
class TransportResult:
    correction: int
    requested_steps: int
    commanded_steps: int
    warning: bool = False


class AdaptiveTransportController:
    """PI trim with slow base learning and conditional-integration anti-windup."""

    STATE_VERSION = 1

    def __init__(self, base_steps, pixels_per_step, correction_gain=0.25,
                 integral_gain=0.02, min_correction=-8, max_correction=8,
                 min_command=None, max_command=None, adaptation_frames=30,
                 warning_frames=15, warning_interval=100, state=None):
        self.initial_base_steps = int(base_steps)
        self.adaptive_base_steps = int(base_steps)
        self.pixels_per_step = float(pixels_per_step)
        self.correction_gain = float(correction_gain)
        self.integral_gain = float(integral_gain)
        self.min_correction = int(min_correction)
        self.max_correction = int(max_correction)
        self.min_command = int(min_command if min_command is not None else base_steps)
        self.max_command = int(max_command if max_command is not None else base_steps)
        self.adaptation_frames = max(2, int(adaptation_frames))
        self.warning_frames = max(2, int(warning_frames))
        self.warning_interval = max(1, int(warning_interval))
        self.integral_steps = 0.0
        self.negative_saturation_run = 0
        self.positive_saturation_run = 0
        self.frames_evaluated = 0
        self.negative_saturated_frames = 0
        self.positive_saturated_frames = 0
        self.observed_min_correction = None
        self.observed_max_correction = None
        self.base_adaptations_down = 0
        self.base_adaptations_up = 0
        self.last_warning_frame = None
        if state:
            self.restore(state)

    def _limits_for_base(self):
        return (
            max(self.min_correction, self.min_command - self.adaptive_base_steps),
            min(self.max_correction, self.max_command - self.adaptive_base_steps),
        )

    def update(self, error_px, trusted=True):
        if not trusted or error_px is None:
            self.negative_saturation_run = max(0, self.negative_saturation_run - 1)
            self.positive_saturation_run = max(0, self.positive_saturation_run - 1)
            command = max(self.min_command, min(self.max_command, self.adaptive_base_steps))
            return TransportResult(0, self.adaptive_base_steps, command)

        self.frames_evaluated += 1
        proportional = (float(error_px) / self.pixels_per_step) * self.correction_gain
        candidate_integral = self.integral_steps + (
            (float(error_px) / self.pixels_per_step) * self.integral_gain
        )
        low, high = self._limits_for_base()
        requested_with_candidate = proportional + candidate_integral
        # Conditional integration: reject only an update that drives farther
        # into an active correction or absolute-command saturation.
        if not ((requested_with_candidate < low and error_px < 0) or
                (requested_with_candidate > high and error_px > 0)):
            self.integral_steps = candidate_integral

        requested_correction = int(round(proportional + self.integral_steps))
        correction = max(low, min(high, requested_correction))
        requested_steps = self.adaptive_base_steps + requested_correction
        commanded_steps = max(self.min_command, min(self.max_command, requested_steps))

        at_negative = correction == self.min_correction
        at_positive = correction == self.max_correction
        self.negative_saturation_run = self.negative_saturation_run + 1 if at_negative else 0
        self.positive_saturation_run = self.positive_saturation_run + 1 if at_positive else 0
        if at_negative:
            self.negative_saturated_frames += 1
        if at_positive:
            self.positive_saturated_frames += 1
        self.observed_min_correction = correction if self.observed_min_correction is None else min(self.observed_min_correction, correction)
        self.observed_max_correction = correction if self.observed_max_correction is None else max(self.observed_max_correction, correction)

        saturated_run = max(self.negative_saturation_run, self.positive_saturation_run)
        warning_due = saturated_run >= self.warning_frames
        warning = warning_due and (
            self.last_warning_frame is None or
            self.frames_evaluated - self.last_warning_frame >= self.warning_interval
        )
        if warning:
            self.last_warning_frame = self.frames_evaluated
        if self.negative_saturation_run >= self.adaptation_frames and self.adaptive_base_steps > self.min_command:
            self.adaptive_base_steps -= 1
            self.base_adaptations_down += 1
            self.negative_saturation_run = 0
            requested_steps = self.adaptive_base_steps + correction
            commanded_steps = max(self.min_command, min(self.max_command, requested_steps))
        elif self.positive_saturation_run >= self.adaptation_frames and self.adaptive_base_steps < self.max_command:
            self.adaptive_base_steps += 1
            self.base_adaptations_up += 1
            self.positive_saturation_run = 0
            requested_steps = self.adaptive_base_steps + correction
            commanded_steps = max(self.min_command, min(self.max_command, requested_steps))

        return TransportResult(correction, requested_steps, commanded_steps, warning)

    def state(self):
        return {
            'version': self.STATE_VERSION,
            'initial_base_steps': self.initial_base_steps,
            'adaptive_base_steps': self.adaptive_base_steps,
            'integral_steps': self.integral_steps,
            'negative_saturation_run': self.negative_saturation_run,
            'positive_saturation_run': self.positive_saturation_run,
            'frames_evaluated': self.frames_evaluated,
            'negative_saturated_frames': self.negative_saturated_frames,
            'positive_saturated_frames': self.positive_saturated_frames,
            'observed_min_correction': self.observed_min_correction,
            'observed_max_correction': self.observed_max_correction,
            'base_adaptations_down': self.base_adaptations_down,
            'base_adaptations_up': self.base_adaptations_up,
            'last_warning_frame': self.last_warning_frame,
        }

    def restore(self, state):
        if int(state.get('version', 0)) != self.STATE_VERSION:
            return False
        if int(state.get('initial_base_steps', -1)) != self.initial_base_steps:
            return False
        for name in ('adaptive_base_steps', 'negative_saturation_run',
                     'positive_saturation_run', 'frames_evaluated',
                     'negative_saturated_frames', 'positive_saturated_frames',
                     'base_adaptations_down', 'base_adaptations_up'):
            if name in state:
                setattr(self, name, int(state[name]))
        self.adaptive_base_steps = max(self.min_command, min(self.max_command, self.adaptive_base_steps))
        self.integral_steps = float(state.get('integral_steps', 0.0))
        self.observed_min_correction = state.get('observed_min_correction')
        self.observed_max_correction = state.get('observed_max_correction')
        warning_frame = state.get('last_warning_frame')
        self.last_warning_frame = int(warning_frame) if warning_frame is not None else None
        return True

    def diagnostics(self):
        frames = self.frames_evaluated
        return {
            **self.state(),
            'final_adaptive_base_steps': self.adaptive_base_steps,
            'min_correction': self.min_correction,
            'max_correction': self.max_correction,
            'min_motor_command': self.min_command,
            'max_motor_command': self.max_command,
            'negative_saturation_pct': round(100.0 * self.negative_saturated_frames / frames, 3) if frames else 0.0,
            'positive_saturation_pct': round(100.0 * self.positive_saturated_frames / frames, 3) if frames else 0.0,
        }
