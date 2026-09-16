"""Production phase association for Super 8 RAW preview candidates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseResult:
    trusted: bool
    reason: str
    selected_y: float = None
    predicted_y: float = None
    error_px: float = None
    ambiguity_px: float = None
    candidate_count: int = 0
    loss_count: int = 0
    reseed_count: int = 0

    def as_dict(self):
        return {
            'phase_trusted': bool(self.trusted),
            'phase_reason': self.reason,
            'phase_selected_y': self.selected_y,
            'phase_predicted_y': self.predicted_y,
            'phase_error_px': self.error_px,
            'phase_ambiguity_px': self.ambiguity_px,
            'phase_candidate_count': int(self.candidate_count),
            'phase_loss_count': int(self.loss_count),
            'phase_reseed_count': int(self.reseed_count),
        }


def select_super8_crop_guidance(phase_result, last_safe_center_y=None,
                                max_prediction_age=2):
    """Select display-only Super 8 crop guidance from a phase result.

    This intentionally does not alter phase trust or registration state.  A
    prediction is allowed only for a short, ordinary detector loss; reseed
    observations and other unsafe losses fall back to the last safe centered
    crop.
    """
    if phase_result.trusted and phase_result.selected_y is not None:
        return {
            'center_y': float(phase_result.selected_y),
            'source': 'trusted_phase',
            'prediction_age': 0,
            'valid': True,
            'fallback_reason': None,
        }

    if (
        phase_result.reason == 'no_complete_candidates'
        and phase_result.predicted_y is not None
        and 0 < int(phase_result.loss_count) <= int(max_prediction_age)
    ):
        return {
            'center_y': float(phase_result.predicted_y),
            'source': 'predicted_phase',
            'prediction_age': int(phase_result.loss_count),
            'valid': True,
            'fallback_reason': None,
        }

    if last_safe_center_y is not None:
        return {
            'center_y': float(last_safe_center_y),
            'source': 'held_safe',
            'prediction_age': None,
            'valid': True,
            'fallback_reason': phase_result.reason,
        }

    return {
        'center_y': None,
        'source': 'full_preview',
        'prediction_age': None,
        'valid': False,
        'fallback_reason': phase_result.reason,
    }


class Super8PhaseTracker:
    """Associate detector candidates with the expected physical phase.

    ``applied_steps`` is the number of motor steps actually applied for the
    transport interval immediately before the current preview.  The tracker
    associates equivalent perforation phase, so a full sprocket pitch is
    removed from that transport before predicting the next Y position.
    """

    def __init__(self, pixels_per_step, expected_sprocket_pitch_px,
                 preview_size, gate_px=30.0,
                 ambiguity_margin_px=8.0, motion_direction=-1,
                 reseed_after_loss=3, reseed_confirmations=3):
        self.pixels_per_step = float(pixels_per_step)
        self.expected_sprocket_pitch_px = float(expected_sprocket_pitch_px)
        if self.pixels_per_step <= 0 or self.expected_sprocket_pitch_px <= 0:
            raise ValueError('phase geometry must be positive')
        self.preview_size = tuple(int(value) for value in preview_size)
        self.gate_px = float(gate_px)
        self.ambiguity_margin_px = float(ambiguity_margin_px)
        self.motion_direction = 1 if float(motion_direction) >= 0 else -1
        self.reseed_after_loss = int(reseed_after_loss)
        self.reseed_confirmations = int(reseed_confirmations)
        self.reset()

    def reset(self):
        self.last_y = None
        self.predicted_y = None
        self.loss_count = 0
        self.reseed_count = 0
        self._reseed_observations = []

    def update(self, candidates, applied_steps):
        candidates = list(candidates or [])
        steps = float(applied_steps)
        if steps < 0:
            return self._lost('motor_steps_negative', len(candidates), steps)

        if self.last_y is None:
            if not candidates:
                return self._lost('no_complete_candidates', 0, steps)
            selected = max(candidates, key=lambda item: float(item.get('score', 0.0)))
            self.last_y = float(selected['center_y'])
            self.predicted_y = self.last_y
            self.loss_count = 0
            return self._result(True, 'seeded', selected_y=self.last_y,
                                candidate_count=len(candidates))

        if (
            self.loss_count >= self.reseed_after_loss
            and candidates
            and not self._reseed_observations
        ):
            # A lost phase is deliberately not recovered from one lucky hole.
            # Start a new controlled sequence, then require confirmations.
            selected = max(candidates, key=lambda item: float(item.get('score', 0.0)))
            selected_y = float(selected['center_y'])
            self._reseed_observations = [(selected_y,)]
            return self._result(False, 'reseed_started', selected_y=selected_y,
                                candidate_count=len(candidates))

        if self._reseed_observations:
            predicted = self._reseed_predict(steps)
        else:
            predicted = self._predict(steps)
        ranked = sorted(
            ((abs(float(item['center_y']) - predicted), item) for item in candidates),
            key=lambda pair: pair[0],
        )
        if not ranked or ranked[0][0] > self.gate_px:
            if self._reseed_observations:
                self._reseed_observations = []
            return self._lost('candidate_outside_phase_gate' if ranked else
                              'no_complete_candidates', len(candidates), steps,
                              predicted_y=predicted)
        if len(ranked) > 1:
            ambiguity = ranked[1][0] - ranked[0][0]
            if ambiguity < self.ambiguity_margin_px:
                if self._reseed_observations:
                    self._reseed_observations = []
                return self._lost('ambiguous_phase', len(candidates), steps,
                                  predicted_y=predicted, ambiguity_px=ambiguity)

        selected_y = float(ranked[0][1]['center_y'])
        if self._reseed_observations:
            self._reseed_observations.append((selected_y,))
            if len(self._reseed_observations) < self.reseed_confirmations:
                return self._result(False, 'reseed_confirming', selected_y=selected_y,
                                    predicted_y=predicted, error_px=selected_y - predicted,
                                    candidate_count=len(candidates))
            self.reseed_count += 1
            self._reseed_observations = []
            reason = 'reseeded'
        else:
            reason = 'tracked'

        self.last_y = selected_y
        self.predicted_y = selected_y
        self.loss_count = 0
        return self._result(True, reason, selected_y=selected_y,
                            predicted_y=predicted, error_px=selected_y - predicted,
                            candidate_count=len(candidates))

    def _predict(self, steps):
        return self._advance(self.predicted_y, steps)

    def _reseed_predict(self, steps):
        return self._advance(self._reseed_observations[-1][0], steps)

    def _advance(self, y, applied_steps):
        # Equivalent phase advances by the difference between the applied
        # transport and one calibrated sprocket pitch.  With the production
        # direction (-1), nominal 308 steps therefore predict about +0.54 px.
        residual_transport = (
            float(applied_steps) * self.pixels_per_step
            - self.expected_sprocket_pitch_px
        )
        return float(y) - self.motion_direction * residual_transport

    def _lost(self, reason, candidate_count, steps, predicted_y=None,
              ambiguity_px=None):
        self.loss_count += 1
        return self._result(False, reason, predicted_y=predicted_y,
                            ambiguity_px=ambiguity_px, candidate_count=candidate_count)

    def _result(self, trusted, reason, selected_y=None, predicted_y=None,
                error_px=None, ambiguity_px=None, candidate_count=0):
        # A lost phase must not become a reseed until a candidate has remained
        # phase-consistent for the configured three frames.
        return PhaseResult(
            trusted=trusted,
            reason=reason,
            selected_y=selected_y,
            predicted_y=predicted_y,
            error_px=error_px,
            ambiguity_px=ambiguity_px,
            candidate_count=candidate_count,
            loss_count=self.loss_count,
            reseed_count=self.reseed_count,
        )
