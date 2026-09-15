"""Super 8 perforation detector for RAW capture registration.

The detector intentionally owns its geometry.  Values are defined at the
2028x1520 experiment resolution and scaled to the incoming preview.
"""

import math
import statistics

import cv2
import numpy as np


class Super8Detector:
    REFERENCE_SIZE = (2028, 1520)

    def __init__(self, reference_size=REFERENCE_SIZE):
        self.reference_size = tuple(reference_size)
        self.reset()

    def reset(self):
        self.last_failure = None
        self.last_threshold = None
        self.last_confidence = 0.0
        self.last_candidate_count = 0
        self.last_viable_count = 0
        self.last_partial_count = 0
        self.last_selected = None

    @staticmethod
    def _gaussian(value, target, sigma):
        if sigma <= 0:
            return 0.0
        return float(math.exp(-0.5 * ((value - target) / sigma) ** 2))

    @staticmethod
    def _scaled_odd_kernel(reference_kernel, scale):
        value = max(3, int(round(reference_kernel * scale)))
        if value % 2 == 0:
            value += 1
        return value

    def _geometry(self, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        ref_w, ref_h = self.reference_size
        sx = frame_w / float(ref_w)
        sy = frame_h / float(ref_h)
        area_scale = sx * sy
        morphology_scale = min(sx, sy)
        return {
            'sx': sx,
            'sy': sy,
            'area_scale': area_scale,
            'roi': (
                max(0, int(round(100 * sx))),
                0,
                min(frame_w, int(round(450 * sx))),
                frame_h,
            ),
            'width': (150 * sx, 235 * sx),
            'height': (190 * sy, 285 * sy),
            'aspect': (0.60 * sx / sy, 1.05 * sx / sy),
            'area': (28000 * area_scale, 58000 * area_scale),
            'expected_x': 278 * sx,
            'x_tolerance': 55 * sx,
            'boundary_margin': max(1, int(round(10 * sy))),
            'close_kernel': self._scaled_odd_kernel(9, morphology_scale),
            'open_kernel': self._scaled_odd_kernel(5, morphology_scale),
        }

    def detect_registration(self, frame_bgr):
        self.reset()
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_failure = 'empty_frame'
            return self._result(None, [])

        geometry = self._geometry(frame_bgr.shape)
        x1, y1, x2, y2 = geometry['roi']
        if x2 <= x1 or y2 <= y1:
            self.last_failure = 'empty_roi'
            return self._result(None, [])

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        crop = cv2.GaussianBlur(gray[y1:y2, x1:x2], (5, 5), 0)
        otsu, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        percentile = float(np.percentile(crop, 88.0))
        threshold = max(float(otsu), percentile, 150.0)
        self.last_threshold = threshold

        mask = np.uint8(crop >= threshold) * 255
        close_kernel = np.ones((geometry['close_kernel'], geometry['close_kernel']), np.uint8)
        open_kernel = np.ones((geometry['open_kernel'], geometry['open_kernel']), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for label in range(1, count):
            local_x, local_y, width, height, pixel_area = [int(value) for value in stats[label]]
            if pixel_area < max(20, int(round(100 * geometry['area_scale']))):
                continue
            x = x1 + local_x
            y = y1 + local_y
            cx = x1 + float(centroids[label][0])
            cy = y1 + float(centroids[label][1])
            component = labels[local_y:local_y + height, local_x:local_x + width] == label
            pixels = gray[y:y + height, x:x + width][component]
            mean_brightness = float(np.mean(pixels)) if pixels.size else 0.0
            aspect = width / max(1.0, float(height))

            classification = 'COMPLETE'
            margin = geometry['boundary_margin']
            if y <= margin:
                classification = 'PARTIAL_TOP'
            elif y + height >= frame_bgr.shape[0] - margin:
                classification = 'PARTIAL_BOTTOM'
            elif not geometry['width'][0] <= width <= geometry['width'][1]:
                classification = 'REJECT_GEOMETRY'
            elif not geometry['height'][0] <= height <= geometry['height'][1]:
                classification = 'REJECT_GEOMETRY'
            elif not geometry['aspect'][0] <= aspect <= geometry['aspect'][1]:
                classification = 'REJECT_GEOMETRY'
            elif not geometry['area'][0] <= pixel_area <= geometry['area'][1]:
                classification = 'REJECT_GEOMETRY'
            elif abs(cx - geometry['expected_x']) > geometry['x_tolerance']:
                classification = 'REJECT_X'
            elif mean_brightness < 170.0:
                classification = 'REJECT_BRIGHTNESS'

            geometry_score = statistics.mean([
                self._gaussian(width, 192 * geometry['sx'], 42 * geometry['sx']),
                self._gaussian(height, 237 * geometry['sy'], 47 * geometry['sy']),
                self._gaussian(aspect, 0.81 * geometry['sx'] / geometry['sy'], 0.22 * geometry['sx'] / geometry['sy']),
                self._gaussian(pixel_area, 43000 * geometry['area_scale'], 15000 * geometry['area_scale']),
            ])
            x_score = self._gaussian(cx, geometry['expected_x'], geometry['x_tolerance'])
            completeness = 1.0 if classification == 'COMPLETE' else 0.05 if classification.startswith('PARTIAL') else 0.2
            brightness_score = max(0.0, min(1.0, (mean_brightness - 140.0) / 100.0))
            center_y_score = self._gaussian(cy, frame_bgr.shape[0] / 2.0, frame_bgr.shape[0] * 0.30)
            score = (
                0.35 * geometry_score
                + 0.25 * x_score
                + 0.20 * completeness
                + 0.10 * brightness_score
                + 0.10 * center_y_score
            )
            candidates.append({
                'center_x': cx,
                'center_y': cy,
                'width': width,
                'height': height,
                'area': float(pixel_area),
                'mean_brightness': mean_brightness,
                'classification': classification,
                'score': score,
            })

        candidates.sort(key=lambda item: item['score'], reverse=True)
        viable = [item for item in candidates if item['classification'] == 'COMPLETE']
        self.last_candidate_count = len(candidates)
        self.last_viable_count = len(viable)
        self.last_partial_count = sum(
            1 for item in candidates if item['classification'].startswith('PARTIAL')
        )
        selected = viable[0] if viable else None
        if selected is None:
            self.last_failure = 'no_complete_sprocket'
            return self._result(None, candidates)

        second = viable[1] if len(viable) > 1 else None
        margin = selected['score'] - second['score'] if second else selected['score']
        confidence = selected['score'] * (0.65 + 0.35 * min(1.0, margin / 0.15))
        self.last_confidence = float(confidence)
        self.last_selected = dict(selected)
        return self._result(selected, candidates)

    def _result(self, selected, candidates):
        sprockets = []
        if selected is not None:
            sprockets.append((
                float(selected['center_x']),
                float(selected['center_y']),
                float(selected['width']),
                float(selected['height']),
                float(selected['area']),
            ))
        return {
            'mode': 'direct' if selected is not None else 'none',
            'actual_y': float(selected['center_y']) if selected is not None else None,
            'sprockets': sprockets,
            'candidate_count': int(self.last_candidate_count),
            'viable_count': int(self.last_viable_count),
            'partial_count': int(self.last_partial_count),
            'confidence': float(self.last_confidence),
            'threshold': float(self.last_threshold) if self.last_threshold is not None else None,
            'failure_reason': self.last_failure,
        }
