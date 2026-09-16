"""Production phase association for Super 8 RAW preview candidates."""

from dataclasses import dataclass
import math


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
    phase_epoch: int = 0
    selected_candidate_index: int = None
    recovery_candidate_index: int = None
    recovery_distance_px: float = None
    recovery_trajectory_error_px: float = None
    recovery_age: int = 0
    recovery_confirmations: int = 0
    candidate_diagnostics: tuple = ()

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
            'phase_epoch': int(self.phase_epoch),
            'phase_selected_candidate_index': self.selected_candidate_index,
            'phase_recovery_candidate_index': self.recovery_candidate_index,
            'phase_recovery_distance_px': self.recovery_distance_px,
            'phase_recovery_trajectory_error_px': self.recovery_trajectory_error_px,
            'phase_recovery_age': int(self.recovery_age),
            'phase_recovery_confirmations': int(self.recovery_confirmations),
            'phase_candidate_diagnostics': [dict(item) for item in self.candidate_diagnostics],
        }


def select_super8_crop_guidance(phase_result, last_safe_center_y=None,
                                max_prediction_age=2):
    """Select display-only Super 8 crop guidance from a phase result."""
    if phase_result.trusted and phase_result.selected_y is not None:
        return {
            'center_y': float(phase_result.selected_y),
            'source': 'trusted_phase', 'prediction_age': 0,
            'valid': True, 'fallback_reason': None,
        }
    if phase_result.reason.startswith('recovery_') and phase_result.selected_y is not None:
        return {
            'center_y': float(phase_result.selected_y),
            'source': 'recovery_phase',
            'prediction_age': int(phase_result.recovery_age),
            'valid': True, 'fallback_reason': None,
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
            'valid': True, 'fallback_reason': None,
        }
    if last_safe_center_y is not None:
        return {
            'center_y': float(last_safe_center_y), 'source': 'held_safe',
            'prediction_age': None, 'valid': True,
            'fallback_reason': phase_result.reason,
        }
    return {
        'center_y': None, 'source': 'full_preview', 'prediction_age': None,
        'valid': False, 'fallback_reason': phase_result.reason,
    }


class Super8PhaseTracker:
    """Associate Super 8 candidates with normal and recovery confidence."""

    def __init__(self, pixels_per_step, expected_sprocket_pitch_px,
                 preview_size, gate_px=30.0, ambiguity_margin_px=8.0,
                 motion_direction=-1, reseed_after_loss=3,
                 reseed_confirmations=3, recovery_gate_px=60.0,
                 recovery_motion_tolerance_px=20.0,
                 recovery_geometry_tolerance=0.45, recovery_horizon=3,
                 recovery_ambiguity_margin_px=8.0):
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
        self.recovery_gate_px = float(recovery_gate_px)
        self.recovery_motion_tolerance_px = float(recovery_motion_tolerance_px)
        self.recovery_geometry_tolerance = float(recovery_geometry_tolerance)
        self.recovery_horizon = int(recovery_horizon)
        self.recovery_ambiguity_margin_px = float(recovery_ambiguity_margin_px)
        self.reset()

    def reset(self):
        self.last_y = None
        self.predicted_y = None
        self.loss_count = 0
        self.reseed_count = 0
        self.phase_epoch = 0
        self.last_candidate = None
        self._reseed_observations = []
        self._recovery = None

    def update(self, candidates, applied_steps):
        candidates = list(candidates or [])
        steps = float(applied_steps)
        if steps < 0:
            return self._lost('motor_steps_negative', len(candidates), steps)
        if self.last_y is None:
            if not candidates:
                return self._lost('no_complete_candidates', 0, steps)
            selected_index, selected = self._best_by_score(candidates)
            self.last_y = float(selected['center_y'])
            self.predicted_y = self.last_y
            self.last_candidate = dict(selected)
            self.loss_count = 0
            return self._result(
                True, 'seeded', selected_y=self.last_y,
                candidate_count=len(candidates), selected_candidate_index=selected_index,
                candidate_diagnostics=self._diagnostics(candidates, self.last_y, selected_index),
            )

        predicted = self._predict(steps)
        # Exactly one calibrated advance is committed for every processed
        # frame, including detector-loss and recovery frames.
        self.predicted_y = predicted
        diagnostics = self._diagnostics(candidates, predicted)
        ranked = self._rank_by_distance(candidates, predicted)

        if self._reseed_observations:
            return self._continue_reseed(candidates, predicted, ranked, steps)

        if self._recovery is not None:
            normal = self._unambiguous_normal(ranked)
            if normal is not None:
                self._recovery = None
                return self._trust_selected(
                    normal[1], normal[0], predicted, len(candidates),
                    self._diagnostics(candidates, predicted, normal[0]),
                )

        if ranked and ranked[0][0] <= self.gate_px:
            if len(ranked) > 1:
                ambiguity = ranked[1][0] - ranked[0][0]
                if ambiguity < self.ambiguity_margin_px:
                    self._recovery = None
                    return self._lost(
                        'ambiguous_phase', len(candidates), steps,
                        predicted_y=predicted, ambiguity_px=ambiguity,
                        candidate_diagnostics=diagnostics,
                    )
            self._recovery = None
            return self._trust_selected(
                ranked[0][2], ranked[0][1], predicted, len(candidates),
                self._diagnostics(candidates, predicted, ranked[0][1]),
            )

        recovery = self._recover(candidates, predicted, ranked, steps)
        if recovery is not None:
            return recovery

        if self._recovery is None and (
            self.loss_count >= self.reseed_after_loss and candidates
        ):
            selected_index, selected = self._best_by_score(candidates)
            selected_y = float(selected['center_y'])
            self._reseed_observations = [(selected_y,)]
            return self._result(
                False, 'reseed_started', selected_y=selected_y,
                candidate_count=len(candidates), selected_candidate_index=selected_index,
                candidate_diagnostics=self._diagnostics(candidates, predicted, selected_index),
            )

        return self._lost(
            'candidate_outside_phase_gate' if ranked else 'no_complete_candidates',
            len(candidates), steps, predicted_y=predicted,
            candidate_diagnostics=diagnostics,
        )

    def _continue_reseed(self, candidates, predicted, ranked, steps):
        reseed_predicted = self._advance(
            self._reseed_observations[-1][0],
            steps,
        )
        ranked = self._rank_by_distance(candidates, reseed_predicted)
        if not ranked or ranked[0][0] > self.gate_px:
            self._reseed_observations = []
            return self._lost(
                'candidate_outside_phase_gate' if ranked else 'no_complete_candidates',
                len(candidates), steps, predicted_y=predicted,
                candidate_diagnostics=self._diagnostics(candidates, predicted),
            )
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.ambiguity_margin_px:
            self._reseed_observations = []
            return self._lost(
                'ambiguous_phase', len(candidates), steps,
                predicted_y=predicted,
                ambiguity_px=ranked[1][0] - ranked[0][0],
                candidate_diagnostics=self._diagnostics(candidates, predicted),
            )
        index, selected = ranked[0][1], ranked[0][2]
        selected_y = float(selected['center_y'])
        self._reseed_observations.append((selected_y,))
        diagnostics = self._diagnostics(candidates, predicted, index)
        if len(self._reseed_observations) < self.reseed_confirmations:
            return self._result(
                False, 'reseed_confirming', selected_y=selected_y,
                predicted_y=predicted, error_px=selected_y - predicted,
                candidate_count=len(candidates), selected_candidate_index=index,
                candidate_diagnostics=diagnostics,
            )
        self.reseed_count += 1
        self.phase_epoch += 1
        self._reseed_observations = []
        self.last_y = selected_y
        self.predicted_y = selected_y
        self.last_candidate = dict(selected)
        self.loss_count = 0
        return self._result(
            True, 'reseeded', selected_y=selected_y,
            predicted_y=predicted, error_px=selected_y - predicted,
            candidate_count=len(candidates), selected_candidate_index=index,
            candidate_diagnostics=diagnostics,
        )

    def _recover(self, candidates, predicted, ranked, steps):
        if not candidates:
            if self._recovery is not None:
                self._recovery = None
                return self._lost('recovery_failed', 0, steps, predicted_y=predicted)
            self._recovery = None
            return None
        if self._recovery is None:
            eligible = [item for item in ranked if self.gate_px < item[0] <= self.recovery_gate_px]
            compatible = self._compatible_candidates(eligible, self.last_candidate)
            choice = self._select_recovery_candidate(compatible)
            if choice is None:
                return None
            index, candidate, distance = choice
            self._recovery = {
                'offset': float(candidate['center_y']) - float(predicted),
                'previous_candidate': dict(candidate),
                'age': 1, 'confirmations': 1,
            }
            self.loss_count += 1
            return self._result(
                False, 'recovery_started', selected_y=float(candidate['center_y']),
                predicted_y=predicted, error_px=float(candidate['center_y']) - predicted,
                candidate_count=len(candidates), selected_candidate_index=index,
                recovery_candidate_index=index, recovery_distance_px=distance,
                recovery_age=1, recovery_confirmations=1,
                candidate_diagnostics=self._diagnostics(
                    candidates, predicted, index, True, distance,
                ),
            )

        hypothesis = self._recovery
        expected_y = float(predicted) + float(hypothesis['offset'])
        eligible = [
            (abs(float(item['center_y']) - predicted), index, item)
            for index, item in enumerate(candidates)
            if abs(float(item['center_y']) - predicted) <= self.recovery_gate_px
        ]
        compatible = self._compatible_recovery_candidates(eligible, hypothesis, expected_y)
        choice = self._select_recovery_candidate(compatible)
        if choice is None:
            self._recovery = None
            return self._lost(
                'recovery_failed', len(candidates), steps, predicted_y=predicted,
                candidate_diagnostics=self._diagnostics(candidates, predicted),
            )
        index, candidate, trajectory_error = choice
        candidate_distance = abs(float(candidate['center_y']) - predicted)
        if trajectory_error > self.recovery_motion_tolerance_px:
            self._recovery = None
            return self._lost(
                'recovery_failed', len(candidates), steps, predicted_y=predicted,
                candidate_diagnostics=self._diagnostics(candidates, predicted),
            )
        hypothesis['previous_candidate'] = dict(candidate)
        hypothesis['age'] += 1
        hypothesis['confirmations'] += 1
        diagnostics = self._diagnostics(
            candidates, predicted, index, True, candidate_distance, trajectory_error,
        )
        if (
            hypothesis['age'] >= self.recovery_horizon
            and hypothesis['confirmations'] < self.reseed_confirmations
        ):
            self._recovery = None
            return self._lost(
                'recovery_horizon_exhausted', len(candidates), steps,
                predicted_y=predicted, candidate_diagnostics=diagnostics,
            )
        if hypothesis['confirmations'] >= self.reseed_confirmations:
            self.last_y = float(candidate['center_y'])
            self.predicted_y = self.last_y
            self.last_candidate = dict(candidate)
            self.loss_count = 0
            age = hypothesis['age']
            confirmations = hypothesis['confirmations']
            self._recovery = None
            return self._result(
                False, 'recovery_established', selected_y=self.last_y,
                predicted_y=predicted, error_px=self.last_y - predicted,
                candidate_count=len(candidates), selected_candidate_index=index,
                recovery_candidate_index=index, recovery_distance_px=candidate_distance,
                recovery_trajectory_error_px=trajectory_error,
                recovery_age=age, recovery_confirmations=confirmations,
                candidate_diagnostics=diagnostics,
            )
        return self._result(
            False, 'recovery_confirming', selected_y=float(candidate['center_y']),
            predicted_y=predicted, error_px=float(candidate['center_y']) - predicted,
            candidate_count=len(candidates), selected_candidate_index=index,
            recovery_candidate_index=index, recovery_distance_px=candidate_distance,
            recovery_trajectory_error_px=trajectory_error,
            recovery_age=hypothesis['age'], recovery_confirmations=hypothesis['confirmations'],
            candidate_diagnostics=diagnostics,
        )

    def _compatible_candidates(self, ranked, reference):
        if reference is None:
            return [(distance, index, item) for distance, index, item in ranked]
        return [
            (distance, index, item) for distance, index, item in ranked
            if self._geometry_distance(item, reference) <= self.recovery_geometry_tolerance
        ]

    def _compatible_recovery_candidates(self, eligible, hypothesis, expected_y):
        reference = hypothesis['previous_candidate']
        result = []
        for _, index, candidate in eligible:
            if self._geometry_distance(candidate, reference) <= self.recovery_geometry_tolerance:
                result.append((abs(float(candidate['center_y']) - expected_y), index, candidate))
        return result

    def _select_recovery_candidate(self, candidates):
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda item: item[0])
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.recovery_ambiguity_margin_px:
            return None
        distance, index, candidate = ranked[0]
        return index, candidate, float(distance)

    def _unambiguous_normal(self, ranked):
        if not ranked or ranked[0][0] > self.gate_px:
            return None
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.ambiguity_margin_px:
            return None
        return ranked[0][1], ranked[0][2]

    @staticmethod
    def _best_by_score(candidates):
        return max(enumerate(candidates), key=lambda pair: float(pair[1].get('score', 0.0)))

    @staticmethod
    def _rank_by_distance(candidates, predicted):
        return sorted(
            (abs(float(item['center_y']) - predicted), index, item)
            for index, item in enumerate(candidates)
        )

    def _diagnostics(self, candidates, predicted, selected_index=None,
                     recovery_considered=False, recovery_distance=None,
                     recovery_trajectory_error=None):
        result = []
        for index, candidate in enumerate(candidates):
            cx = float(candidate['center_x'])
            cy = float(candidate['center_y'])
            width = float(candidate['width'])
            height = float(candidate['height'])
            distance = abs(cy - predicted) if predicted is not None else None
            result.append({
                'candidate_index': int(index), 'center_x': cx, 'center_y': cy,
                'x1': cx - width / 2.0, 'y1': cy - height / 2.0,
                'x2': cx + width / 2.0, 'y2': cy + height / 2.0,
                'width': width, 'height': height,
                'area': float(candidate.get('area', 0.0)),
                'score': float(candidate.get('score', 0.0)),
                'classification': candidate.get('classification', 'COMPLETE'),
                'distance_from_predicted_px': distance,
                'inside_trusted_gate': bool(distance is not None and distance <= self.gate_px),
                'inside_recovery_gate': bool(distance is not None and distance <= self.recovery_gate_px),
                'recovery_considered': bool(recovery_considered),
                'selected': bool(index == selected_index),
                'association_status': (
                    'selected' if index == selected_index else
                    'inside_trusted_gate' if distance is not None and distance <= self.gate_px else
                    'inside_recovery_gate' if distance is not None and distance <= self.recovery_gate_px else
                    'outside_recovery_gate'
                ),
                'recovery_distance_px': recovery_distance if index == selected_index else None,
                'recovery_trajectory_error_px': (
                    recovery_trajectory_error if index == selected_index else None
                ),
            })
        return tuple(result)

    @staticmethod
    def _geometry_distance(candidate, reference):
        if reference is None:
            return 0.0
        def relative(a, b):
            return abs(float(a) - float(b)) / max(1.0, abs(float(b)))
        x_term = abs(float(candidate['center_x']) - float(reference['center_x'])) / 40.0
        return math.sqrt(
            (x_term * 0.25) ** 2
            + relative(candidate['width'], reference['width']) ** 2
            + relative(candidate['height'], reference['height']) ** 2
            + relative(candidate.get('area', 0.0), reference.get('area', 0.0)) ** 2
        )

    def _trust_selected(self, selected, selected_index, predicted,
                        candidate_count, diagnostics):
        selected_y = float(selected['center_y'])
        self.last_y = selected_y
        self.predicted_y = selected_y
        self.last_candidate = dict(selected)
        self.loss_count = 0
        return self._result(
            True, 'tracked', selected_y=selected_y, predicted_y=predicted,
            error_px=selected_y - predicted, candidate_count=candidate_count,
            selected_candidate_index=selected_index,
            candidate_diagnostics=diagnostics,
        )

    def _predict(self, steps):
        return self._advance(self.predicted_y, steps)

    def _advance(self, y, applied_steps):
        residual_transport = (
            float(applied_steps) * self.pixels_per_step
            - self.expected_sprocket_pitch_px
        )
        return float(y) - self.motion_direction * residual_transport

    def _lost(self, reason, candidate_count, steps, predicted_y=None,
              ambiguity_px=None, candidate_diagnostics=()):
        self.loss_count += 1
        if predicted_y is None:
            predicted_y = self.predicted_y
        return self._result(
            False, reason, predicted_y=predicted_y, ambiguity_px=ambiguity_px,
            candidate_count=candidate_count, candidate_diagnostics=candidate_diagnostics,
        )

    def _result(self, trusted, reason, selected_y=None, predicted_y=None,
                error_px=None, ambiguity_px=None, candidate_count=0,
                selected_candidate_index=None, recovery_candidate_index=None,
                recovery_distance_px=None, recovery_age=0,
                recovery_trajectory_error_px=None, recovery_confirmations=0,
                candidate_diagnostics=()):
        return PhaseResult(
            trusted=trusted, reason=reason, selected_y=selected_y,
            predicted_y=predicted_y, error_px=error_px,
            ambiguity_px=ambiguity_px, candidate_count=candidate_count,
            loss_count=self.loss_count, reseed_count=self.reseed_count,
            phase_epoch=self.phase_epoch,
            selected_candidate_index=selected_candidate_index,
            recovery_candidate_index=recovery_candidate_index,
            recovery_distance_px=recovery_distance_px,
            recovery_trajectory_error_px=recovery_trajectory_error_px,
            recovery_age=recovery_age,
            recovery_confirmations=recovery_confirmations,
            candidate_diagnostics=candidate_diagnostics,
        )
