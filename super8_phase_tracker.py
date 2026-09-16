"""Production phase association for Super 8 RAW preview candidates."""

from dataclasses import dataclass
import math


def unwrap_super8_y_near(y, reference_y, pitch_px):
    """Return the pitch-equivalent coordinate nearest an unwrapped reference."""
    if y is None or reference_y is None:
        return None
    pitch_px = float(pitch_px)
    if pitch_px <= 0:
        raise ValueError('pitch_px must be positive')
    return float(y) + round((float(reference_y) - float(y)) / pitch_px) * pitch_px


@dataclass(frozen=True)
class PhaseResult:
    trusted: bool
    reason: str
    selected_y: float = None
    predicted_y: float = None
    selected_raw_y: float = None
    selected_unwrapped_y: float = None
    predicted_unwrapped_y: float = None
    pitch_offset: int = 0
    phase_wrapped: bool = False
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
            'phase_selected_raw_y': self.selected_raw_y,
            'phase_selected_unwrapped_y': self.selected_unwrapped_y,
            'phase_predicted_unwrapped_y': self.predicted_unwrapped_y,
            'phase_pitch_offset': int(self.pitch_offset),
            'phase_wrapped': bool(self.phase_wrapped),
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
                 recovery_ambiguity_margin_px=8.0,
                 recovery_confirmations=2):
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
        self.recovery_confirmations = max(2, int(recovery_confirmations))
        self.reset()

    def reset(self):
        self.last_y = None
        self.predicted_y = None
        self.last_unwrapped_y = None
        self.predicted_unwrapped_y = None
        self.last_pitch_offset = 0
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
            self.last_unwrapped_y = self.last_y
            self.predicted_unwrapped_y = self.last_unwrapped_y
            self.predicted_y = self.last_y
            self.last_candidate = dict(selected)
            self.loss_count = 0
            return self._result(
                True, 'seeded', selected_y=self.last_y,
                selected_raw_y=self.last_y,
                selected_unwrapped_y=self.last_unwrapped_y,
                predicted_unwrapped_y=self.predicted_unwrapped_y,
                candidate_count=len(candidates), selected_candidate_index=selected_index,
                candidate_diagnostics=self._diagnostics(candidates, self.last_y, selected_index),
            )

        predicted_unwrapped = self._predict(steps)
        # Exactly one calibrated advance is committed for every processed
        # frame, including detector-loss and recovery frames.
        self.predicted_unwrapped_y = predicted_unwrapped
        predicted = self._physical_y(predicted_unwrapped)
        self.predicted_y = predicted
        diagnostics = self._diagnostics(candidates, predicted_unwrapped)
        ranked = self._rank_by_distance(candidates, predicted_unwrapped)

        if self._reseed_observations:
            return self._continue_reseed(candidates, predicted, ranked, steps)

        if self._recovery is not None:
            normal = self._unambiguous_normal(ranked)
            if normal is not None:
                self._recovery = None
                return self._trust_selected(
                    normal[1], normal[0], predicted_unwrapped, len(candidates),
                    self._diagnostics(candidates, predicted_unwrapped, normal[1]),
                )

        if ranked and ranked[0][0] <= self.gate_px:
            normal = self._unambiguous_normal(ranked)
            if normal is not None:
                self._recovery = None
                return self._trust_selected(
                    normal[1], normal[0], predicted_unwrapped, len(candidates),
                    self._diagnostics(candidates, predicted_unwrapped, normal[0]),
                )
            self._recovery = None
            ambiguity = ranked[1][0] - ranked[0][0] if len(ranked) > 1 else None
            return self._lost(
                'ambiguous_phase' if ambiguity is not None else 'candidate_outside_phase_gate',
                len(candidates), steps, predicted_y=predicted,
                predicted_unwrapped_y=predicted_unwrapped, ambiguity_px=ambiguity,
                candidate_diagnostics=diagnostics,
            )

        recovery = self._recover(candidates, predicted_unwrapped, ranked, steps)
        if recovery is not None:
            return recovery

        if self._recovery is None and (
            self.loss_count >= self.reseed_after_loss and candidates
        ):
            selected_index, selected = self._best_by_score(candidates)
            selected_y = float(selected['center_y'])
            selected_unwrapped = self._nearest_unwrapped(selected_y, predicted_unwrapped)[0]
            self._reseed_observations = [(selected_y, selected_unwrapped, int(
                self._nearest_unwrapped(selected_y, predicted_unwrapped)[1]
            ))]
            return self._result(
                False, 'reseed_started', selected_y=selected_y,
                selected_raw_y=selected_y, selected_unwrapped_y=selected_unwrapped,
                predicted_unwrapped_y=predicted_unwrapped,
                candidate_count=len(candidates), selected_candidate_index=selected_index,
                candidate_diagnostics=self._diagnostics(candidates, predicted_unwrapped, selected_index),
            )

        return self._lost(
            'candidate_outside_phase_gate' if ranked else 'no_complete_candidates',
            len(candidates), steps, predicted_y=predicted,
            predicted_unwrapped_y=predicted_unwrapped,
            candidate_diagnostics=diagnostics,
        )

    def _continue_reseed(self, candidates, predicted, ranked, steps):
        reseed_predicted = self._advance(
            self._reseed_observations[-1][1], steps,
        )
        ranked = self._rank_by_distance(candidates, reseed_predicted)
        if not ranked or ranked[0][0] > self.gate_px:
            self._reseed_observations = []
            return self._lost(
                'candidate_outside_phase_gate' if ranked else 'no_complete_candidates',
                len(candidates), steps, predicted_y=predicted,
                predicted_unwrapped_y=self.predicted_unwrapped_y,
                candidate_diagnostics=self._diagnostics(candidates, self.predicted_unwrapped_y),
            )
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.ambiguity_margin_px:
            prior_offset = self._reseed_observations[-1][2]
            same_offset = [item for item in ranked if item[4] == prior_offset]
            if len(same_offset) == 1:
                ranked = same_offset + [item for item in ranked if item not in same_offset]
            else:
                self._reseed_observations = []
                return self._lost(
                    'ambiguous_phase', len(candidates), steps,
                    predicted_y=predicted,
                    predicted_unwrapped_y=self.predicted_unwrapped_y,
                    ambiguity_px=ranked[1][0] - ranked[0][0],
                    candidate_diagnostics=self._diagnostics(candidates, self.predicted_unwrapped_y),
                )
        index, selected, selected_unwrapped, selected_offset = (
            ranked[0][1], ranked[0][2], ranked[0][3], ranked[0][4]
        )
        selected_y = float(selected['center_y'])
        self._reseed_observations.append((selected_y, selected_unwrapped, selected_offset))
        diagnostics = self._diagnostics(candidates, self.predicted_unwrapped_y, index)
        if len(self._reseed_observations) < self.reseed_confirmations:
            return self._result(
                False, 'reseed_confirming', selected_y=selected_y,
                predicted_y=predicted, selected_raw_y=selected_y,
                selected_unwrapped_y=selected_unwrapped,
                predicted_unwrapped_y=self.predicted_unwrapped_y,
                pitch_offset=selected_offset,
                phase_wrapped=selected_offset != 0,
                error_px=selected_unwrapped - self.predicted_unwrapped_y,
                candidate_count=len(candidates), selected_candidate_index=index,
                candidate_diagnostics=diagnostics,
            )
        preserve_epoch = bool(
            selected_offset != 0
            and self.last_candidate is not None
            and self._geometry_distance(selected, self.last_candidate)
                <= self.recovery_geometry_tolerance
            and abs(selected_unwrapped - self.predicted_unwrapped_y)
                <= self.recovery_gate_px + self.recovery_motion_tolerance_px
        )
        self.reseed_count += 1
        if not preserve_epoch:
            self.phase_epoch += 1
        self._reseed_observations = []
        self.last_y = selected_y
        self.last_unwrapped_y = selected_unwrapped
        self.predicted_unwrapped_y = selected_unwrapped
        self.predicted_y = selected_y
        self.last_pitch_offset = selected_offset
        self.last_candidate = dict(selected)
        self.loss_count = 0
        return self._result(
            True, 'reseeded', selected_y=selected_y,
            predicted_y=predicted, selected_raw_y=selected_y,
            selected_unwrapped_y=selected_unwrapped,
            predicted_unwrapped_y=self.predicted_unwrapped_y,
            pitch_offset=selected_offset, phase_wrapped=selected_offset != 0,
            error_px=selected_unwrapped - self.predicted_unwrapped_y,
            candidate_count=len(candidates), selected_candidate_index=index,
            candidate_diagnostics=diagnostics,
        )

    def _recover(self, candidates, predicted, ranked, steps):
        if not candidates:
            if self._recovery is not None:
                self._recovery = None
                return self._lost('recovery_failed', 0, steps, predicted_y=self._physical_y(predicted), predicted_unwrapped_y=predicted)
            self._recovery = None
            return None
        if self._recovery is None:
            eligible = [item for item in ranked if self.gate_px < item[0] <= self.recovery_gate_px]
            compatible = self._compatible_candidates(eligible, self.last_candidate)
            choice = self._select_recovery_candidate(compatible)
            if choice is None:
                return None
            index, candidate, distance, unwrapped, offset = choice
            self._recovery = {
                'offset': float(unwrapped) - float(predicted),
                'previous_candidate': dict(candidate),
                'previous_unwrapped': float(unwrapped),
                'pitch_offset': int(offset),
                'age': 1, 'confirmations': 1,
            }
            self.loss_count += 1
            return self._result(
                False, 'recovery_started', selected_y=float(candidate['center_y']),
                selected_raw_y=float(candidate['center_y']), selected_unwrapped_y=float(unwrapped),
                predicted_y=self._physical_y(predicted), predicted_unwrapped_y=predicted,
                pitch_offset=offset, phase_wrapped=offset != 0,
                error_px=float(unwrapped) - predicted,
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
            item for item in self._rank_by_distance(candidates, predicted)
            if item[0] <= self.recovery_gate_px
        ]
        compatible = self._compatible_recovery_candidates(eligible, hypothesis, expected_y)
        choice = self._select_recovery_candidate(compatible)
        if choice is None:
            self._recovery = None
            return self._lost(
                'recovery_failed', len(candidates), steps,
                predicted_y=self._physical_y(predicted), predicted_unwrapped_y=predicted,
                candidate_diagnostics=self._diagnostics(candidates, predicted),
            )
        index, candidate, trajectory_error, unwrapped, offset = choice
        candidate_distance = abs(float(unwrapped) - predicted)
        if trajectory_error > self.recovery_motion_tolerance_px:
            self._recovery = None
            return self._lost(
                'recovery_failed', len(candidates), steps,
                predicted_y=self._physical_y(predicted), predicted_unwrapped_y=predicted,
                candidate_diagnostics=self._diagnostics(candidates, predicted),
            )
        hypothesis['previous_candidate'] = dict(candidate)
        hypothesis['previous_unwrapped'] = float(unwrapped)
        hypothesis['pitch_offset'] = int(offset)
        hypothesis['age'] += 1
        hypothesis['confirmations'] += 1
        diagnostics = self._diagnostics(
            candidates, predicted, index, True, candidate_distance, trajectory_error,
        )
        if (
            hypothesis['age'] >= self.recovery_horizon
            and hypothesis['confirmations'] < self.recovery_confirmations
        ):
            self._recovery = None
            return self._lost(
                'recovery_horizon_exhausted', len(candidates), steps,
                predicted_y=self._physical_y(predicted), predicted_unwrapped_y=predicted,
                candidate_diagnostics=diagnostics,
            )
        if hypothesis['confirmations'] >= self.recovery_confirmations:
            self.last_y = float(candidate['center_y'])
            self.last_unwrapped_y = float(unwrapped)
            self.predicted_unwrapped_y = self.last_unwrapped_y
            self.predicted_y = self.last_y
            self.last_pitch_offset = int(offset)
            self.last_candidate = dict(candidate)
            self.loss_count = 0
            age = hypothesis['age']
            confirmations = hypothesis['confirmations']
            self._recovery = None
            return self._result(
                True, 'recovery_established', selected_y=self.last_y,
                selected_raw_y=self.last_y, selected_unwrapped_y=self.last_unwrapped_y,
                predicted_y=self._physical_y(predicted), predicted_unwrapped_y=predicted,
                pitch_offset=offset, phase_wrapped=offset != 0,
                error_px=self.last_unwrapped_y - predicted,
                candidate_count=len(candidates), selected_candidate_index=index,
                recovery_candidate_index=index, recovery_distance_px=candidate_distance,
                recovery_trajectory_error_px=trajectory_error,
                recovery_age=age, recovery_confirmations=confirmations,
                candidate_diagnostics=diagnostics,
            )
        return self._result(
            False, 'recovery_confirming', selected_y=float(candidate['center_y']),
            selected_raw_y=float(candidate['center_y']), selected_unwrapped_y=float(unwrapped),
            predicted_y=self._physical_y(predicted), predicted_unwrapped_y=predicted,
            pitch_offset=offset, phase_wrapped=offset != 0,
            error_px=float(unwrapped) - predicted,
            candidate_count=len(candidates), selected_candidate_index=index,
            recovery_candidate_index=index, recovery_distance_px=candidate_distance,
            recovery_trajectory_error_px=trajectory_error,
            recovery_age=hypothesis['age'], recovery_confirmations=hypothesis['confirmations'],
            candidate_diagnostics=diagnostics,
        )

    def _compatible_candidates(self, ranked, reference):
        if reference is None:
            return list(ranked)
        return [
            item for item in ranked
            if self._geometry_distance(item[2], reference) <= self.recovery_geometry_tolerance
        ]

    def _compatible_recovery_candidates(self, eligible, hypothesis, expected_y):
        reference = hypothesis['previous_candidate']
        result = []
        for _, index, candidate, unwrapped, offset in eligible:
            if self._geometry_distance(candidate, reference) <= self.recovery_geometry_tolerance:
                result.append((abs(float(unwrapped) - expected_y), index, candidate, unwrapped, offset))
        return result

    def _select_recovery_candidate(self, candidates):
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda item: item[0])
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.recovery_ambiguity_margin_px:
            return None
        distance, index, candidate, unwrapped, offset = ranked[0]
        return index, candidate, float(distance), float(unwrapped), int(offset)

    def _unambiguous_normal(self, ranked):
        if not ranked or ranked[0][0] > self.gate_px:
            return None
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.ambiguity_margin_px:
            # A visible perforation pair can straddle the pitch boundary.
            # Preserve the established physical branch only when exactly one
            # candidate is also locally continuous with the prior raw center.
            if self.last_y is not None:
                continuous = [
                    item for item in ranked
                    if abs(float(item[2]['center_y']) - self.last_y) <= self.gate_px
                ]
                if len(continuous) == 1:
                    return continuous[0][1], continuous[0][2], continuous[0][3], continuous[0][4]
            return None
        return ranked[0][1], ranked[0][2], ranked[0][3], ranked[0][4]

    @staticmethod
    def _best_by_score(candidates):
        return max(enumerate(candidates), key=lambda pair: float(pair[1].get('score', 0.0)))

    def _nearest_unwrapped(self, raw_y, reference):
        raw_y = float(raw_y)
        height = float(self.preview_size[1])
        offsets = [0]
        boundary = self.expected_sprocket_pitch_px * 0.35
        if raw_y <= boundary:
            offsets.append(1)
        if raw_y >= height - boundary:
            offsets.append(-1)
        best = min(
            (abs(raw_y + offset * self.expected_sprocket_pitch_px - reference),
             raw_y + offset * self.expected_sprocket_pitch_px, offset)
            for offset in offsets
        )
        return best[1], best[2]

    def _rank_by_distance(self, candidates, predicted):
        ranked = []
        for index, item in enumerate(candidates):
            unwrapped, offset = self._nearest_unwrapped(item['center_y'], predicted)
            ranked.append((abs(float(unwrapped) - predicted), index, item, unwrapped, offset))
        return sorted(ranked, key=lambda value: value[0])

    def _physical_y(self, unwrapped):
        if unwrapped is None:
            return None
        height = float(self.preview_size[1])
        reference = self.last_y if self.last_y is not None else height / 2.0
        values = [float(unwrapped) + k * self.expected_sprocket_pitch_px
                  for k in range(-3, 4)]
        valid = [value for value in values if 0.0 <= value <= height]
        return min(valid or values, key=lambda value: abs(value - reference))

    def _diagnostics(self, candidates, predicted, selected_index=None,
                     recovery_considered=False, recovery_distance=None,
                     recovery_trajectory_error=None):
        result = []
        for index, candidate in enumerate(candidates):
            cx = float(candidate['center_x'])
            cy = float(candidate['center_y'])
            width = float(candidate['width'])
            height = float(candidate['height'])
            unwrapped, offset = self._nearest_unwrapped(cy, predicted) if predicted is not None else (None, 0)
            distance = abs(unwrapped - predicted) if predicted is not None else None
            result.append({
                'candidate_index': int(index), 'center_x': cx, 'center_y': cy,
                'x1': cx - width / 2.0, 'y1': cy - height / 2.0,
                'x2': cx + width / 2.0, 'y2': cy + height / 2.0,
                'width': width, 'height': height,
                'area': float(candidate.get('area', 0.0)),
                'score': float(candidate.get('score', 0.0)),
                'classification': candidate.get('classification', 'COMPLETE'),
                'distance_from_predicted_px': distance,
                'unwrapped_y': unwrapped,
                'pitch_offset': int(offset),
                'wrapped': bool(offset != 0),
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
        selected_unwrapped, pitch_offset = self._nearest_unwrapped(selected_y, predicted)
        self.last_y = selected_y
        self.last_unwrapped_y = selected_unwrapped
        self.predicted_unwrapped_y = selected_unwrapped
        self.predicted_y = selected_y
        self.last_pitch_offset = pitch_offset
        self.last_candidate = dict(selected)
        self.loss_count = 0
        return self._result(
            True, 'tracked', selected_y=selected_y, predicted_y=self._physical_y(predicted),
            selected_raw_y=selected_y, selected_unwrapped_y=selected_unwrapped,
            predicted_unwrapped_y=predicted, pitch_offset=pitch_offset,
            phase_wrapped=pitch_offset != 0,
            error_px=selected_unwrapped - predicted, candidate_count=candidate_count,
            selected_candidate_index=selected_index,
            candidate_diagnostics=diagnostics,
        )

    def _predict(self, steps):
        return self._advance(self.predicted_unwrapped_y, steps)

    def _advance(self, y, applied_steps):
        residual_transport = (
            float(applied_steps) * self.pixels_per_step
            - self.expected_sprocket_pitch_px
        )
        return float(y) - self.motion_direction * residual_transport

    def _lost(self, reason, candidate_count, steps, predicted_y=None,
              predicted_unwrapped_y=None, ambiguity_px=None, candidate_diagnostics=()):
        self.loss_count += 1
        if predicted_unwrapped_y is None:
            predicted_unwrapped_y = self.predicted_unwrapped_y
        if predicted_y is None:
            predicted_y = self._physical_y(predicted_unwrapped_y)
        return self._result(
            False, reason, predicted_y=predicted_y, predicted_unwrapped_y=predicted_unwrapped_y,
            ambiguity_px=ambiguity_px,
            candidate_count=candidate_count, candidate_diagnostics=candidate_diagnostics,
        )

    def _result(self, trusted, reason, selected_y=None, predicted_y=None,
                error_px=None, ambiguity_px=None, candidate_count=0,
                selected_candidate_index=None, recovery_candidate_index=None,
                recovery_distance_px=None, recovery_age=0,
                recovery_trajectory_error_px=None, recovery_confirmations=0,
                candidate_diagnostics=(), selected_raw_y=None,
                selected_unwrapped_y=None, predicted_unwrapped_y=None,
                pitch_offset=0, phase_wrapped=False):
        return PhaseResult(
            trusted=trusted, reason=reason, selected_y=selected_y,
            predicted_y=predicted_y, error_px=error_px,
            selected_raw_y=selected_raw_y if selected_raw_y is not None else selected_y,
            selected_unwrapped_y=selected_unwrapped_y,
            predicted_unwrapped_y=predicted_unwrapped_y,
            pitch_offset=pitch_offset, phase_wrapped=phase_wrapped,
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
