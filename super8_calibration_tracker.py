"""Hardware-independent physical-perforation tracking and calibration metrics."""

from dataclasses import dataclass, field
import statistics


@dataclass
class TrackObservation:
    steps: float
    center_x: float
    center_y: float
    width: float
    height: float
    area: float
    score: float


@dataclass
class PerforationTrack:
    track_id: int
    observations: list = field(default_factory=list)
    missing_frames: int = 0

    @property
    def last(self):
        return self.observations[-1]

    def slope(self):
        return linear_slope([
            (observation.steps, observation.center_y)
            for observation in self.observations
        ])

    def predicted_y(self, steps, fallback_slope=0.0):
        slope = self.slope()
        if slope is None:
            slope = fallback_slope
        return self.last.center_y + slope * (float(steps) - self.last.steps)


def linear_slope(points):
    if len(points) < 2:
        return None
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        return None
    return sum(
        (point[0] - mean_x) * (point[1] - mean_y)
        for point in points
    ) / denominator


def robust_values(values, max_mad_scale=3.5):
    values = [float(value) for value in values]
    if not values:
        return []
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad <= 0:
        return values
    return [
        value for value, deviation in zip(values, deviations)
        if deviation <= mad * float(max_mad_scale)
    ] or values


def summarize(values):
    values = [float(value) for value in values]
    if not values:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
        }
    return {
        'count': len(values),
        'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'std': statistics.pstdev(values),
        'min': min(values),
        'max': max(values),
    }


def crossing_step(observations, registration_y):
    registration_y = float(registration_y)
    ordered = sorted(observations, key=lambda item: item.steps)
    for left, right in zip(ordered, ordered[1:]):
        left_delta = left.center_y - registration_y
        right_delta = right.center_y - registration_y
        if left_delta == 0:
            return float(left.steps)
        if right_delta == 0:
            return float(right.steps)
        if left_delta * right_delta < 0:
            fraction = (registration_y - left.center_y) / (
                right.center_y - left.center_y
            )
            return left.steps + fraction * (right.steps - left.steps)
    return None


class Super8PerforationTracker:
    """Associate complete detector candidates with physical perforations."""

    def __init__(
        self,
        registration_y,
        max_pixels_per_step=5.0,
        min_position_gate=55.0,
        ambiguity_margin=8.0,
        opposite_motion_tolerance=8.0,
        max_missing_frames=2,
    ):
        self.registration_y = float(registration_y)
        self.max_pixels_per_step = float(max_pixels_per_step)
        self.min_position_gate = float(min_position_gate)
        self.ambiguity_margin = float(ambiguity_margin)
        self.opposite_motion_tolerance = float(opposite_motion_tolerance)
        self.max_missing_frames = int(max_missing_frames)
        self.tracks = []
        self.next_track_id = 1
        self.last_steps = None
        self.motion_direction = None
        self.ambiguous_frames = 0
        self.rejected_assignments = 0
        self.detector_loss_frames = 0
        self.pitch_observations = []
        self.confidences = []

    def update(self, candidates, cumulative_steps):
        steps = float(cumulative_steps)
        normalized = [self._observation(candidate, steps) for candidate in candidates]
        normalized.sort(key=lambda item: item.center_y)

        if self.last_steps is not None and steps < self.last_steps:
            return {'accepted': False, 'reason': 'motor_steps_moved_backwards'}
        if not normalized:
            self.detector_loss_frames += 1
            for track in self.tracks:
                track.missing_frames += 1
            self.last_steps = steps
            return {'accepted': False, 'reason': 'no_complete_candidates'}

        if not self.tracks:
            created = [self._new_track(observation) for observation in normalized]
            self._record_frame_measurements(normalized)
            self.last_steps = steps
            return {'accepted': True, 'created_tracks': created, 'assignments': []}

        active = [
            track for track in self.tracks
            if track.missing_frames <= self.max_missing_frames
        ]
        global_slope = self._global_slope()
        possible = []
        per_track = {}
        for track in active:
            step_delta = max(0.0, steps - track.last.steps)
            gate = max(
                self.min_position_gate,
                step_delta * self.max_pixels_per_step,
            )
            predicted_y = track.predicted_y(steps, global_slope or 0.0)
            options = []
            for index, observation in enumerate(normalized):
                if not self._geometry_consistent(track.last, observation):
                    continue
                position_error = abs(observation.center_y - predicted_y)
                if position_error > gate:
                    continue
                if self._opposes_motion(track.last, observation):
                    continue
                secondary = (
                    abs(observation.center_x - track.last.center_x) * 0.08
                    + abs(observation.width - track.last.width) * 0.04
                    + abs(observation.height - track.last.height) * 0.04
                )
                cost = position_error + secondary + (1.0 - observation.score) * 4.0
                options.append((cost, index, predicted_y))
                possible.append((cost, track.track_id, index, predicted_y))
            options.sort()
            per_track[track.track_id] = options

        for options in per_track.values():
            if len(options) >= 2 and options[1][0] - options[0][0] < self.ambiguity_margin:
                self.ambiguous_frames += 1
                return {
                    'accepted': False,
                    'reason': 'ambiguous_candidate_assignment',
                    'recapture_required': True,
                }

        assigned_tracks = set()
        assigned_candidates = set()
        assignments = []
        selected_pairs = []
        for cost, track_id, candidate_index, predicted_y in sorted(possible):
            if track_id in assigned_tracks or candidate_index in assigned_candidates:
                continue
            track = next(item for item in active if item.track_id == track_id)
            observation = normalized[candidate_index]
            assigned_tracks.add(track_id)
            assigned_candidates.add(candidate_index)
            assignment = {
                'track_id': track_id,
                'center_y': observation.center_y,
                'predicted_y': predicted_y,
                'position_error': abs(observation.center_y - predicted_y),
                'cost': cost,
            }
            assignments.append(assignment)
            selected_pairs.append((track, observation))
        unmatched = [
            observation
            for index, observation in enumerate(normalized)
            if index not in assigned_candidates
        ]
        frame_height = self.registration_y * 2.0
        for observation in unmatched:
            near_boundary = (
                observation.center_y <= frame_height * 0.20
                or observation.center_y >= frame_height * 0.80
            )
            if not near_boundary:
                self.rejected_assignments += 1
                return {
                    'accepted': False,
                    'reason': 'implausible_identity_jump',
                }

        for track, observation in selected_pairs:
            track.observations.append(observation)
            track.missing_frames = 0
        for track in active:
            if track.track_id not in assigned_tracks:
                track.missing_frames += 1
        created = [self._new_track(observation) for observation in unmatched]

        if not assignments and not created:
            self.rejected_assignments += 1
            self.last_steps = steps
            return {'accepted': False, 'reason': 'no_plausible_track_assignment'}

        self._update_direction(assignments)
        self._record_frame_measurements(normalized)
        self.last_steps = steps
        return {
            'accepted': True,
            'assignments': assignments,
            'created_tracks': created,
        }

    def measurements(self):
        pitch_values = robust_values(self.pitch_observations)
        steps_per_px_values = []
        crossings = []
        tracks_payload = []
        for track in self.tracks:
            slope = track.slope()
            step_span = (
                track.observations[-1].steps - track.observations[0].steps
                if len(track.observations) >= 2 else 0.0
            )
            if slope is not None and abs(slope) > 1e-9 and len(track.observations) >= 4:
                steps_per_px_values.append(1.0 / abs(slope))
            crossing = crossing_step(track.observations, self.registration_y)
            if crossing is not None:
                crossings.append((crossing, track.track_id))
            tracks_payload.append({
                'track_id': track.track_id,
                'observations': len(track.observations),
                'step_span': step_span,
                'slope_px_per_step': slope,
                'crossing_step': crossing,
            })

        steps_per_px_values = robust_values(steps_per_px_values)
        crossings.sort()
        transition_values = robust_values([
            right[0] - left[0]
            for left, right in zip(crossings, crossings[1:])
            if right[0] > left[0]
        ])
        pitch_summary = summarize(pitch_values)
        steps_per_px_summary = summarize(steps_per_px_values)
        transition_summary = summarize(transition_values)
        cross_check = None
        if (
            pitch_summary['median'] is not None
            and steps_per_px_summary['median'] is not None
            and transition_summary['median'] is not None
        ):
            derived = pitch_summary['median'] * steps_per_px_summary['median']
            direct = transition_summary['median']
            cross_check = {
                'pitch_times_steps_per_px': derived,
                'direct_steps_per_pitch': direct,
                'difference_steps': derived - direct,
                'difference_percent': abs(derived - direct) / direct * 100.0,
            }

        return {
            'sprocket_pitch_px_values': pitch_values,
            'sprocket_pitch_px': pitch_summary,
            'steps_per_px_values': steps_per_px_values,
            'steps_per_px': steps_per_px_summary,
            'crossings': [
                {'steps': crossing, 'track_id': track_id}
                for crossing, track_id in crossings
            ],
            'steps_per_pitch_values': transition_values,
            'steps_per_pitch': transition_summary,
            'confidence': summarize(self.confidences),
            'tracks': tracks_payload,
            'ambiguous_track_count': self.ambiguous_frames,
            'rejected_track_count': self.rejected_assignments,
            'detector_loss_frames': self.detector_loss_frames,
            'cross_check': cross_check,
        }

    def _global_slope(self):
        slopes = [track.slope() for track in self.tracks]
        slopes = [slope for slope in slopes if slope is not None]
        return statistics.median(slopes) if slopes else None

    def _opposes_motion(self, previous, current):
        if self.motion_direction is None:
            return False
        movement = current.center_y - previous.center_y
        return movement * self.motion_direction < -self.opposite_motion_tolerance

    def _update_direction(self, assignments):
        movements = []
        assigned_ids = {assignment['track_id'] for assignment in assignments}
        for track in self.tracks:
            if track.track_id in assigned_ids and len(track.observations) >= 2:
                movements.append(
                    track.observations[-1].center_y
                    - track.observations[-2].center_y
                )
        significant = [value for value in movements if abs(value) > 1.0]
        if significant:
            direction = -1 if statistics.median(significant) < 0 else 1
            if self.motion_direction is None:
                self.motion_direction = direction

    def _record_frame_measurements(self, observations):
        self.confidences.extend(item.score for item in observations)
        for upper, lower in zip(observations, observations[1:]):
            self.pitch_observations.append(lower.center_y - upper.center_y)

    def _new_track(self, observation):
        track = PerforationTrack(self.next_track_id, [observation])
        self.next_track_id += 1
        self.tracks.append(track)
        return track.track_id

    @staticmethod
    def _geometry_consistent(previous, current):
        if abs(current.center_x - previous.center_x) > 45.0:
            return False
        if previous.width and abs(current.width - previous.width) / previous.width > 0.35:
            return False
        if previous.height and abs(current.height - previous.height) / previous.height > 0.35:
            return False
        if previous.area and abs(current.area - previous.area) / previous.area > 0.50:
            return False
        return True

    @staticmethod
    def _observation(candidate, steps):
        return TrackObservation(
            steps=float(steps),
            center_x=float(candidate['center_x']),
            center_y=float(candidate['center_y']),
            width=float(candidate['width']),
            height=float(candidate['height']),
            area=float(candidate['area']),
            score=float(candidate.get('score', candidate.get('confidence', 0.0))),
        )
