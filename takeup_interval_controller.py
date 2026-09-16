"""Capture-local, one-sided adaptive take-up pulse interval control."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TakeupIntervalDecision:
    pulse_sequence: int
    interval_before: int
    interval_after: int
    adaptation_applied: bool
    adaptation_delta: int
    adaptation_reason: str
    disturbance_filtered_px: float = None


class AdaptiveTakeupIntervalController:
    """Lengthen take-up intervals in response to measured pulse disturbance."""

    def __init__(self, initial_interval=12, min_interval=8, max_interval=32,
                 filter_window=3):
        self.initial_interval = int(initial_interval)
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.filter_window = max(1, int(filter_window))
        if not (0 < self.min_interval <= self.initial_interval <= self.max_interval):
            raise ValueError('take-up interval limits are invalid')
        self.reset()

    def reset(self):
        self.interval_frames = self.initial_interval
        self._recent_disturbances = []
        self._adapted_sequences = set()

    def decide(self, pulse_sequence, disturbance_px=None, trusted=False,
               plausible_phase_loss=False):
        """Make at most one bounded interval decision for a pulse.

        The filter is a short median of available +1 disturbances. It is
        intentionally one-sided: evidence never shortens the interval.
        """
        sequence = int(pulse_sequence)
        before = self.interval_frames
        if sequence in self._adapted_sequences:
            return TakeupIntervalDecision(
                sequence, before, before, False, 0, 'already_adapted',
                self._filtered_disturbance(),
            )
        self._adapted_sequences.add(sequence)

        if disturbance_px is None or not trusted and not plausible_phase_loss:
            return TakeupIntervalDecision(
                sequence, before, before, False, 0,
                'measurement_unavailable_or_not_attributable',
                self._filtered_disturbance(),
            )

        value = abs(float(disturbance_px))
        self._recent_disturbances.append(value)
        del self._recent_disturbances[:-self.filter_window]
        filtered = self._filtered_disturbance()
        if not trusted and plausible_phase_loss:
            delta = 4
            reason = 'plausible_takeup_phase_loss'
        elif filtered < 5.0:
            delta = 1
            reason = 'low_postpulse_disturbance'
        elif filtered < 10.0:
            delta = 0
            reason = 'moderate_postpulse_disturbance'
        elif filtered < 20.0:
            delta = 2
            reason = 'high_postpulse_disturbance'
        else:
            delta = 4
            reason = 'severe_postpulse_disturbance'

        after = min(self.max_interval, before + delta)
        applied_delta = after - before
        self.interval_frames = after
        return TakeupIntervalDecision(
            sequence, before, after, bool(applied_delta), applied_delta,
            reason if applied_delta else 'interval_already_at_maximum' if delta else reason,
            filtered,
        )

    def _filtered_disturbance(self):
        if not self._recent_disturbances:
            return None
        values = sorted(self._recent_disturbances)
        middle = len(values) // 2
        return float(values[middle]) if len(values) % 2 else float(
            (values[middle - 1] + values[middle]) / 2.0
        )
