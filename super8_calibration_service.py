"""Production Super 8 calibration strategy."""

import asyncio
import io

from calibration_persistence import save_calibration_atomic
from super8_calibration_result import build_super8_calibration_proposal
from super8_calibration_tracker import Super8PerforationTracker


class Super8CalibrationService:
    MODE = 'super8_physical_track'
    MIN_TRANSITIONS = 5
    TARGET_TRANSITIONS = 8
    MIN_CANDIDATE_SCORE = 0.55
    PREVIEW_INTERVAL_INCREMENTS = 8

    def __init__(
        self,
        camera,
        tc,
        detector,
        calibration_resolution=(2028, 1520),
        save_path='calibration.super8.json',
        search_steps_per_pitch=None,
    ):
        self.camera = camera
        self.tc = tc
        self.detector = detector
        self.calibration_resolution = tuple(calibration_resolution)
        self.save_path = save_path
        self.search_steps_per_pitch = search_steps_per_pitch

    def capture_calibration_preview(self, debug_scale=1.0):
        import cv2

        frame = self._capture_jpeg_frame()
        if frame is None:
            raise RuntimeError('Failed to decode Super 8 calibration preview')
        self._validate_frame_resolution(frame)
        candidates = self.detector.detect_calibration_candidates(frame)
        debug_frame = frame.copy()
        for candidate in candidates:
            self._draw_candidate(debug_frame, candidate)
        debug_frame = cv2.flip(debug_frame, 0)
        if float(debug_scale) > 0 and float(debug_scale) != 1.0:
            debug_frame = cv2.resize(
                debug_frame,
                (
                    max(1, int(round(debug_frame.shape[1] * float(debug_scale)))),
                    max(1, int(round(debug_frame.shape[0] * float(debug_scale)))),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
        ok, encoded = cv2.imencode(
            '.jpg', debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )
        if not ok:
            raise RuntimeError('Failed to encode Super 8 calibration preview')
        return {
            'complete_candidate_count': len(candidates),
            'candidates': candidates,
            'calibration_resolution': list(self.calibration_resolution),
        }, encoded.tobytes()

    async def run_calibration(
        self,
        target_transitions=TARGET_TRANSITIONS,
        step_size=10,
        max_steps_per_transition=500,
        settle_delay=0.05,
        progress_callback=None,
        debug_scale=1.0,
    ):
        target_transitions = max(self.MIN_TRANSITIONS, int(target_transitions))
        step_size = max(1, min(10, int(step_size)))
        safety_bound = 500
        if self.search_steps_per_pitch is not None:
            safety_bound = min(
                safety_bound,
                max(step_size, int(round(float(self.search_steps_per_pitch) * 1.5))),
            )
        max_steps_per_transition = max(
            step_size,
            min(safety_bound, int(max_steps_per_transition)),
        )
        registration_y = self.calibration_resolution[1] / 2.0
        tracker = Super8PerforationTracker(registration_y=registration_y)
        total_steps = 0
        rejected_trials = 0
        consecutive_losses = 0

        frame, candidates = self._capture_candidates()
        if frame is None:
            return self._failed_result('failed_to_decode_start_frame', tracker, 0)
        while not candidates and total_steps < max_steps_per_transition:
            self.tc.steps_forward(step_size)
            total_steps += step_size
            await asyncio.sleep(settle_delay)
            frame, candidates = self._capture_candidates()
            if frame is None:
                continue
        if not candidates:
            return self._failed_result(
                'did_not_find_complete_perforation', tracker, total_steps, 1
            )
        initial = tracker.update(candidates, total_steps)
        if not initial['accepted']:
            return self._failed_result(initial['reason'], tracker, total_steps)

        last_transition_steps = total_steps
        last_preview_crossings = 0
        last_preview_transitions = 0
        increments_since_preview = self.PREVIEW_INTERVAL_INCREMENTS

        async def report_progress(frame, candidates, update, preview_reason=None):
            nonlocal last_preview_crossings, last_preview_transitions
            nonlocal increments_since_preview

            measurements = tracker.measurements()
            transitions = measurements['steps_per_pitch']['count']
            crossings = measurements['crossings']
            latest_crossing_steps = last_transition_steps
            if crossings:
                latest_crossing_steps = max(
                    latest_crossing_steps,
                    max(crossing['steps'] for crossing in crossings),
                )
            progress_fraction = min(
                0.99,
                max(0.0, total_steps - latest_crossing_steps)
                / max_steps_per_transition,
            )
            payload = {
                'completed_transitions': transitions,
                'observed_crossings': len(crossings),
                'target_transitions': target_transitions,
                'progress_units': len(crossings) + progress_fraction,
                'target_progress_units': target_transitions + 1,
                'total_motor_steps': total_steps,
                'complete_candidates': len(candidates),
                'track_count': len(tracker.tracks),
                'tracking_accepted': bool(update.get('accepted')),
                'tracking_reason': update.get('reason'),
            }
            crossing_changed = len(crossings) != last_preview_crossings
            transition_changed = transitions != last_preview_transitions
            new_track = bool(update.get('created_tracks'))
            routine_preview = (
                increments_since_preview >= self.PREVIEW_INTERVAL_INCREMENTS
            )
            should_preview = (
                frame is not None
                and (
                    preview_reason is not None
                    or routine_preview
                    or crossing_changed
                    or transition_changed
                    or new_track
                    or not update.get('accepted')
                )
            )
            if should_preview:
                reason = preview_reason
                if reason is None:
                    if not update.get('accepted'):
                        reason = update.get('reason') or 'tracking_rejected'
                    elif transition_changed:
                        reason = 'transition'
                    elif crossing_changed:
                        reason = 'crossing'
                    elif new_track:
                        reason = 'new_track'
                    else:
                        reason = 'periodic'
                payload['preview_reason'] = reason
                payload['preview_jpeg'] = self._encode_tracking_preview(
                    frame,
                    candidates,
                    update,
                    tracker,
                    total_steps,
                    measurements,
                    debug_scale,
                )
                increments_since_preview = 0
                last_preview_crossings = len(crossings)
                last_preview_transitions = transitions
            if progress_callback is not None:
                await progress_callback(payload)

        await report_progress(frame, candidates, initial, 'initial')
        total_safety_limit = (
            total_steps + (target_transitions + 1) * max_steps_per_transition
        )
        while total_steps < total_safety_limit:
            self.tc.steps_forward(step_size)
            total_steps += step_size
            await asyncio.sleep(settle_delay)

            increments_since_preview += 1
            frame, candidates = self._capture_candidates()
            update = tracker.update(candidates, total_steps)
            if update.get('recapture_required'):
                frame, candidates = self._capture_candidates()
                update = tracker.update(candidates, total_steps)
                if update.get('recapture_required'):
                    rejected_trials += 1
                    await report_progress(
                        frame, candidates, update, 'unresolved_candidate_ambiguity'
                    )
                    return self._failed_result(
                        'unresolved_candidate_ambiguity',
                        tracker,
                        total_steps,
                        rejected_trials,
                    )

            measurements = tracker.measurements()
            crossings = measurements['crossings']
            if crossings:
                latest_crossing = max(crossing['steps'] for crossing in crossings)
                last_transition_steps = max(last_transition_steps, latest_crossing)
            await report_progress(frame, candidates, update)

            if not update['accepted']:
                if update['reason'] == 'no_complete_candidates':
                    consecutive_losses += 1
                    if consecutive_losses >= 3:
                        return self._failed_result(
                            'sustained_detector_loss',
                            tracker,
                            total_steps,
                            rejected_trials + 1,
                        )
                else:
                    rejected_trials += 1
                    return self._failed_result(
                        update['reason'], tracker, total_steps, rejected_trials
                    )
            else:
                consecutive_losses = 0

            transitions = measurements['steps_per_pitch']['count']
            if transitions >= target_transitions:
                return self._completed_result(
                    tracker, total_steps, rejected_trials, target_transitions
                )
            if total_steps - last_transition_steps > max_steps_per_transition:
                return self._failed_result(
                    'movement_bound_exceeded_without_transition',
                    tracker,
                    total_steps,
                    rejected_trials + 1,
                )

        return self._failed_result(
            'total_movement_safety_limit_exceeded',
            tracker,
            total_steps,
            rejected_trials + 1,
        )

    def build_proposal(self, result, exposure_result):
        return build_super8_calibration_proposal(
            result,
            exposure_result,
            self.calibration_resolution,
            minimum_transitions=self.MIN_TRANSITIONS,
        )

    def save_calibration(self, calibration):
        if calibration.get('film_format') != 'super8':
            raise ValueError('Refusing to save non-Super 8 calibration')
        if calibration.get('status') != 'calibrated':
            raise ValueError('Refusing to save provisional Super 8 calibration')
        return save_calibration_atomic(calibration, self.save_path)

    def _completed_result(self, tracker, total_steps, rejected, target):
        return {
            'valid': True,
            'reason': None,
            'target_transitions': target,
            'total_motor_steps': total_steps,
            'rejected_transitions': rejected,
            'measurements': tracker.measurements(),
        }

    @staticmethod
    def _failed_result(reason, tracker, total_steps, rejected=0):
        return {
            'valid': False,
            'reason': reason,
            'total_motor_steps': total_steps,
            'rejected_transitions': rejected,
            'measurements': tracker.measurements(),
        }

    def _capture_candidates(self):
        frame = self._capture_jpeg_frame()
        if frame is None:
            return None, []
        self._validate_frame_resolution(frame)
        candidates = self.detector.detect_calibration_candidates(frame)
        return frame, [
            candidate for candidate in candidates
            if float(candidate.get('score', 0.0)) >= self.MIN_CANDIDATE_SCORE
        ]

    def _capture_jpeg_frame(self):
        import cv2
        import numpy as np

        buffer = io.BytesIO()
        self.camera.capture_file(buffer, format='jpeg')
        return cv2.imdecode(
            np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR
        )

    def _validate_frame_resolution(self, frame):
        actual = (int(frame.shape[1]), int(frame.shape[0]))
        if actual != self.calibration_resolution:
            raise RuntimeError(
                f'Super 8 calibration expected {self.calibration_resolution[0]}x'
                f'{self.calibration_resolution[1]}, received {actual[0]}x{actual[1]}'
            )

    @staticmethod
    def _draw_candidate(frame, candidate):
        import cv2

        cx = float(candidate['center_x'])
        cy = float(candidate['center_y'])
        width = float(candidate['width'])
        height = float(candidate['height'])
        cv2.rectangle(
            frame,
            (int(round(cx - width / 2)), int(round(cy - height / 2))),
            (int(round(cx + width / 2)), int(round(cy + height / 2))),
            (0, 255, 0),
            2,
        )

    def _encode_tracking_preview(
        self,
        frame,
        candidates,
        update,
        tracker,
        total_steps,
        measurements,
        debug_scale,
    ):
        import cv2

        debug_frame = frame.copy()
        registration_y = int(round(tracker.registration_y))
        cv2.line(
            debug_frame,
            (0, registration_y),
            (debug_frame.shape[1] - 1, registration_y),
            (0, 0, 255),
            2,
        )
        for candidate in candidates:
            self._draw_candidate(debug_frame, candidate)

        assignments = {
            int(item['track_id']): item for item in update.get('assignments', [])
        }
        track_labels = []
        global_slope = tracker._global_slope() or 0.0
        for track in tracker.tracks:
            predicted_y = track.predicted_y(total_steps, global_slope)
            marker_y = int(round(predicted_y))
            cv2.line(debug_frame, (8, marker_y), (60, marker_y), (0, 255, 255), 2)
            if track.last.steps == float(total_steps):
                label_x = max(4, int(round(track.last.center_x + track.last.width / 2 + 8)))
                label_y = max(18, int(round(track.last.center_y)))
                assignment = assignments.get(track.track_id)
                suffix = '' if assignment is None else f" err={assignment['position_error']:.1f}"
                track_labels.append((f'T{track.track_id}{suffix}', label_x, label_y))

        transitions = measurements['steps_per_pitch']['count']
        if not update.get('accepted'):
            status = update.get('reason') or 'rejected'
        elif update.get('created_tracks'):
            status = 'accepted/new-track'
        else:
            status = 'accepted'
        debug_frame = cv2.flip(debug_frame, 0)
        frame_height = debug_frame.shape[0]
        for label, label_x, source_y in track_labels:
            cv2.putText(
                debug_frame,
                label,
                (label_x, max(18, frame_height - 1 - source_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        lines = (
            'Super 8 calibration',
            f'steps={total_steps} status={status}',
            f"crossings={len(measurements['crossings'])} transitions={transitions}",
            f'candidates={len(candidates)} tracks={len(tracker.tracks)}',
        )
        for index, line in enumerate(lines):
            y = 28 + index * 25
            cv2.putText(
                debug_frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA,
            )

        scale = float(debug_scale)
        if scale > 0 and scale != 1.0:
            debug_frame = cv2.resize(
                debug_frame,
                (
                    max(1, int(round(debug_frame.shape[1] * scale))),
                    max(1, int(round(debug_frame.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
        ok, encoded = cv2.imencode(
            '.jpg', debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )
        if not ok:
            raise RuntimeError('Failed to encode Super 8 tracking preview')
        return encoded.tobytes()
