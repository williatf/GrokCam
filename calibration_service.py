"""Production Regular 8 calibration strategy used by the WebSocket workflow."""

import asyncio
import cv2
import io

import numpy as np

from calibration_persistence import save_calibration_atomic


class Regular8CalibrationService:
    """Production Regular 8 calibration measurements and motor procedures."""
    def __init__(self, camera, tc, detector, settings, save_path="calibration.json"):
        self.camera = camera
        self.tc = tc
        self.detector = detector
        self.settings = settings
        self.save_path = save_path

    def capture_sprocket_preview(self, debug_scale=1.0):
        buffer = io.BytesIO()
        self.camera.capture_file(buffer, format="jpeg")

        frame = cv2.imdecode(
            np.frombuffer(buffer.getvalue(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise RuntimeError("Failed to decode captured JPEG frame.")

        sprockets = self.detector.detect(frame, mode="profile") or []
        classified_sprockets = self.detector.classify_sprockets(sprockets, frame.shape)
        debug_frame = frame.copy()
        roi_x1, roi_y1, roi_x2, roi_y2 = self.detector.roi_bounds(frame.shape)
        cv2.rectangle(
            debug_frame,
            (roi_x1, roi_y1),
            (roi_x2 - 1, roi_y2 - 1),
            (0, 255, 255),
            2,
        )

        for item in classified_sprockets:
            cx, cy, width, height, area = item["sprocket"]
            status = item["status"]
            x1 = int(round(cx - width / 2))
            y1 = int(round(cy - height / 2))
            x2 = int(round(cx + width / 2))
            y2 = int(round(cy + height / 2))
            color = (0, 255, 0) if status == "full" else (0, 255, 255)

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(
                debug_frame,
                (int(round(cx)), int(round(cy))),
                4,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                debug_frame,
                f"{status.upper()} cy={cy:.1f} area={area:.0f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        debug_frame = cv2.flip(debug_frame, 0)

        scale = float(debug_scale)
        if scale > 0.0 and scale != 1.0:
            debug_width = max(1, int(round(debug_frame.shape[1] * scale)))
            debug_height = max(1, int(round(debug_frame.shape[0] * scale)))
            debug_frame = cv2.resize(
                debug_frame,
                (debug_width, debug_height),
                interpolation=cv2.INTER_LINEAR,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            debug_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            raise RuntimeError("Failed to encode sprocket preview JPEG.")

        measurement_dict = self._build_measurements(classified_sprockets)
        return measurement_dict, encoded.tobytes()

    async def seek_two_full_sprockets(self, step_size=10, max_steps=500, settle_delay=0.05):
        total_steps = 0

        while total_steps <= max_steps:
            frame_bgr = self._capture_jpeg_frame()
            if frame_bgr is None:
                return {
                    'valid': False,
                    'reason': 'failed_to_decode_seek_frame',
                    'steps': total_steps,
                }

            sprockets = self.detector.detect(frame_bgr, mode='profile') or []
            classified = self.detector.classify_sprockets(sprockets, frame_bgr.shape)
            full_sprocket_count = sum(
                1 for item in classified if item.get('status') == 'full'
            )

            if full_sprocket_count == 2:
                return {
                    'valid': True,
                    'steps': total_steps,
                    'full_sprocket_count': 2,
                    'sprocket_count': len(sprockets),
                }

            if total_steps >= max_steps:
                break

            self.tc.steps_forward(step_size)
            total_steps += step_size
            await asyncio.sleep(settle_delay)

        return {
            'valid': False,
            'reason': 'did_not_find_two_full_sprockets',
            'steps': total_steps,
        }

    async def measure_steps_per_pitch(self, sprocket_pitch_px, step_chunk=20, max_steps=500):
        def choose_reference_y(frame_bgr, sprockets):
            # Motor calibration must follow one physical sprocket, not a pair midpoint.
            classified = self.detector.classify_sprockets(sprockets, frame_bgr.shape)
            full_sprockets = [
                item['sprocket'] for item in classified if item.get('status') == 'full'
            ]
            if full_sprockets:
                center_y = frame_bgr.shape[0] / 2.0
                anchor = min(
                    full_sprockets,
                    key=lambda sprocket: abs(sprocket[1] - center_y),
                )
                return float(anchor[1])
            return None

        frame_bgr = self._capture_jpeg_frame()
        if frame_bgr is None:
            return {'valid': False, 'reason': 'failed_to_decode_start_frame'}

        sprockets = self.detector.detect(frame_bgr, mode='profile') or []
        if not sprockets:
            return {'valid': False, 'reason': 'no_sprockets_in_start_frame'}

        start_y = choose_reference_y(frame_bgr, sprockets)
        if start_y is None:
            return {'valid': False, 'reason': 'no_stable_registration_reference'}

        total_steps = 0
        threshold = float(sprocket_pitch_px) * 0.85

        while total_steps < max_steps:
            self.tc.steps_forward(step_chunk)
            total_steps += step_chunk
            await asyncio.sleep(0.05)

            frame_bgr = self._capture_jpeg_frame()
            if frame_bgr is None:
                continue

            sprockets = self.detector.detect(frame_bgr, mode='profile') or []
            if not sprockets:
                continue

            current_y = choose_reference_y(frame_bgr, sprockets)
            if current_y is None:
                continue

            delta_y = abs(float(current_y) - float(start_y))
            if delta_y >= threshold:
                steps_per_px = total_steps / delta_y
                steps_per_pitch = steps_per_px * float(sprocket_pitch_px)
                return {
                    'steps_per_pitch': int(round(steps_per_pitch)),
                    'steps_per_px': float(steps_per_px),
                    'total_steps': int(total_steps),
                    'delta_y': float(delta_y),
                    'valid': True,
                }

        return {
            'valid': False,
            'reason': 'did_not_reach_pitch_threshold',
            'total_steps': int(total_steps),
        }

    def compute_summary(self, samples):
        pitch_values = [
            float(sample['sprocket_pitch_px'])
            for sample in samples
            if sample.get('pitch_valid') and sample.get('sprocket_pitch_px') is not None
        ]
        area_values = [
            float(area)
            for sample in samples
            for area in sample.get('full_sprocket_areas', [])
        ]
        summary = {
            'pitch_mean': None,
            'pitch_min': None,
            'pitch_max': None,
            'pitch_std': None,
            'area_mean': None,
            'area_min': None,
            'area_max': None,
            'area_std': None,
            'valid_samples': len(pitch_values),
            'total_samples': len(samples),
        }
        if pitch_values:
            values = np.array(pitch_values, dtype=float)
            summary.update({
                'pitch_mean': float(np.mean(values)),
                'pitch_min': float(np.min(values)),
                'pitch_max': float(np.max(values)),
                'pitch_std': float(np.std(values)),
            })
        if area_values:
            values = np.array(area_values, dtype=float)
            summary.update({
                'area_mean': float(np.mean(values)),
                'area_min': float(np.min(values)),
                'area_max': float(np.max(values)),
                'area_std': float(np.std(values)),
            })
        return summary

    def build_proposed_calibration(self, samples, exposure_result, motor_calibration=None):
        valid_pitch_values = self.filter_robust_values([
            float(sample['sprocket_pitch_px'])
            for sample in samples
            if sample.get('pitch_valid') and sample.get('sprocket_pitch_px') is not None
        ])
        trusted_areas = self.filter_robust_values([
            float(area)
            for sample in samples
            for area in sample.get('full_sprocket_areas', [])
        ])

        if len(valid_pitch_values) < 5:
            return None, False, 'need_at_least_5_valid_pitch_samples'
        if not trusted_areas:
            return None, False, 'need_trusted_full_sprocket_area_samples'

        trusted_pitch = float(np.median(np.array(valid_pitch_values, dtype=float)))
        trusted_area = float(np.median(np.array(trusted_areas, dtype=float)))
        area_mad = float(np.median(
            np.abs(np.array(trusted_areas, dtype=float) - trusted_area)
        )) if trusted_areas else 0.0
        if trusted_area <= 0:
            return None, False, 'invalid_trusted_area'

        area_spread_frac = 0.05
        if area_mad > 0:
            area_spread_frac = max(
                0.05,
                min(0.12, (2.5 * area_mad) / trusted_area),
            )

        if motor_calibration and motor_calibration.get('motor_updated'):
            steps_per_pitch = motor_calibration.get('motor_steps_per_pitch')
        else:
            steps_per_pitch = self.settings.get('steps_per_pitch', 280)
        if steps_per_pitch is None:
            return None, False, 'missing_steps_per_pitch'

        steps_per_pitch = float(steps_per_pitch)
        calibration_resolution = self.settings.get(
            'calibration_resolution', [2028, 1520]
        )
        proposed = {
            'calibration_version': 2,
            'calibration_resolution': list(calibration_resolution),
            'exposure_time': int(exposure_result['exposure_time']),
            'gain': 1.0,
            'sprocket_pitch_px': trusted_pitch,
            'steps_per_pitch': int(round(steps_per_pitch)),
            'steps_per_px': steps_per_pitch / trusted_pitch,
            'sprocket_area_min': int(round(trusted_area * (1.0 - area_spread_frac))),
            'sprocket_area_max': int(round(trusted_area * (1.0 + area_spread_frac))),
        }
        return proposed, True, None

    def _capture_jpeg_frame(self):
        buffer = io.BytesIO()
        self.camera.capture_file(buffer, format='jpeg')
        return cv2.imdecode(
            np.frombuffer(buffer.getvalue(), np.uint8),
            cv2.IMREAD_COLOR,
        )

    @staticmethod
    def filter_robust_values(values, max_mad_scale=3.5):
        if not values:
            return []
        array = np.array(values, dtype=float)
        median = float(np.median(array))
        deviations = np.abs(array - median)
        mad = float(np.median(deviations))
        if mad <= 0:
            return array.tolist()
        filtered = array[deviations <= mad * max_mad_scale]
        return filtered.tolist() if filtered.size else array.tolist()

    def save_calibration(self, calibration_dict):
        declared_format = calibration_dict.get('film_format')
        if declared_format not in (None, 'regular8'):
            raise ValueError('Refusing to save non-Regular 8 calibration')
        return save_calibration_atomic(calibration_dict, self.save_path)

    def _build_measurements(self, classified_sprockets):
        measurements = {
            "sprocket_count": len(classified_sprockets),
            "full_sprocket_count": 0,
            "partial_sprocket_count": 0,
            "full_sprocket_areas": [],
            "sprocket_pitch_px": None,
            "pitch_valid": False,
            "pitch_reason": "need_exactly_two_full_sprockets",
            "sprocket_area_nominal": None,
            "sprocket_area_min": None,
            "sprocket_area_max": None,
        }

        if not classified_sprockets:
            return measurements

        full_sprockets = [
            item["sprocket"]
            for item in classified_sprockets
            if item["status"] == "full"
        ]
        partial_sprockets = [
            item["sprocket"]
            for item in classified_sprockets
            if item["status"] == "partial"
        ]

        measurements["full_sprocket_count"] = len(full_sprockets)
        measurements["partial_sprocket_count"] = len(partial_sprockets)
        measurements["full_sprocket_areas"] = [float(sprocket[4]) for sprocket in full_sprockets]

        areas = np.array(measurements["full_sprocket_areas"], dtype=float)

        expected_pitch = self.settings.get("sprocket_pitch_px")
        if expected_pitch is None:
            expected_pitch = self.detector.expected_pitch
        if expected_pitch is None:
            expected_pitch = 814

        expected_pitch = float(expected_pitch)
        pitch_min = expected_pitch * 0.93
        pitch_max = expected_pitch * 1.07

        if len(full_sprockets) == 2:
            full_sorted = sorted(full_sprockets, key=lambda sprocket: sprocket[1])
            measured_pitch = float(abs(full_sorted[1][1] - full_sorted[0][1]))
            if pitch_min <= measured_pitch <= pitch_max:
                measurements["sprocket_pitch_px"] = measured_pitch
                measurements["pitch_valid"] = True
                measurements["pitch_reason"] = "exactly_two_full_sprockets"
            else:
                measurements["sprocket_pitch_px"] = None
                measurements["pitch_valid"] = False
                measurements["pitch_reason"] = "pitch_out_of_expected_range"

        if areas.size > 0:
            nominal_area = float(np.mean(areas))
            measurements["sprocket_area_nominal"] = nominal_area
            measurements["sprocket_area_min"] = nominal_area * 0.8
            measurements["sprocket_area_max"] = nominal_area * 1.2

        return measurements
