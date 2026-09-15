"""Production Super 8 calibration strategy."""

import asyncio
import io

import cv2
import numpy as np

from calibration_persistence import save_calibration_atomic
from super8_calibration_result import build_super8_calibration_proposal
from super8_calibration_tracker import Super8PerforationTracker


class Super8CalibrationService:
    MODE = 'super8_physical_track'
    MIN_TRANSITIONS = 5
    TARGET_TRANSITIONS = 8
    MIN_CANDIDATE_SCORE = 0.55

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
        total_safety_limit = (
            total_steps + (target_transitions + 1) * max_steps_per_transition
        )
        while total_steps < total_safety_limit:
            self.tc.steps_forward(step_size)
            total_steps += step_size
            await asyncio.sleep(settle_delay)

            _, candidates = self._capture_candidates()
            update = tracker.update(candidates, total_steps)
            if update.get('recapture_required'):
                _, candidates = self._capture_candidates()
                update = tracker.update(candidates, total_steps)
                if update.get('recapture_required'):
                    rejected_trials += 1
                    return self._failed_result(
                        'unresolved_candidate_ambiguity',
                        tracker,
                        total_steps,
                        rejected_trials,
                    )

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

            measurements = tracker.measurements()
            transitions = measurements['steps_per_pitch']['count']
            if progress_callback is not None:
                await progress_callback({
                    'completed_transitions': transitions,
                    'target_transitions': target_transitions,
                    'total_motor_steps': total_steps,
                    'complete_candidates': len(candidates),
                    'track_count': len(tracker.tracks),
                })
            if transitions:
                latest_crossing = max(
                    crossing['steps'] for crossing in measurements['crossings']
                )
                last_transition_steps = max(last_transition_steps, latest_crossing)
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
