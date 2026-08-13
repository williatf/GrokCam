import cv2
import numpy as np


class FastSprocketDetector:
    """Track two bright sprockets in small predicted regions."""

    def __init__(self, reference_size=(2028, 1520), expected_pitch=795.5):
        self.reference_size = tuple(reference_size)
        self.expected_pitch = float(expected_pitch)
        self.previous = None

    def reset(self):
        self.previous = None

    def detect(self, frame_bgr):
        centers = self._predicted_centers(frame_bgr.shape)
        found = [self._detect_one(frame_bgr, center) for center in centers]
        if any(item is None for item in found):
            return None
        found.sort(key=lambda item: item[1])

        frame_h, frame_w = frame_bgr.shape[:2]
        sx = frame_w / float(self.reference_size[0])
        sy = frame_h / float(self.reference_size[1])
        separation = found[1][1] - found[0][1]
        if abs(separation - self.expected_pitch * sy) > 90.0 * sy:
            return None
        if abs(found[1][0] - found[0][0]) > 90.0 * sx:
            return None

        self.previous = found
        return found

    def _predicted_centers(self, frame_shape):
        if self.previous and len(self.previous) == 2:
            return [(self.previous[0][0], self.previous[0][1]),
                    (self.previous[1][0], self.previous[1][1])]
        frame_h, frame_w = frame_shape[:2]
        return [(0.213 * frame_w, 0.267 * frame_h),
                (0.218 * frame_w, 0.783 * frame_h)]

    def _detect_one(self, frame_bgr, center):
        frame_h, frame_w = frame_bgr.shape[:2]
        sx = frame_w / float(self.reference_size[0])
        sy = frame_h / float(self.reference_size[1])
        half_w = max(80, int(round(270 * sx)))
        half_h = max(60, int(round(190 * sy)))
        center_x, center_y = center
        x1 = max(0, int(center_x - half_w))
        x2 = min(frame_w, int(center_x + half_w))
        y1 = max(0, int(center_y - half_h))
        y2 = min(frame_h, int(center_y + half_h))
        roi_bgr = frame_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        peak = float(np.percentile(gray, 99.5))
        threshold = int(max(180, min(250, peak * 0.86)))
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            aspect = width / float(height) if height else 0.0
            if not (180 * sx <= width <= 500 * sx and 120 * sy <= height <= 360 * sy):
                continue
            if not (1.05 <= aspect <= 2.2 and area >= 15000 * sx * sy):
                continue
            touches_edge = (
                x <= 1 or y <= 1
                or x + width >= gray.shape[1] - 1
                or y + height >= gray.shape[0] - 1
            )
            if touches_edge:
                continue
            fill = area / float(width * height)
            candidate_x = x1 + x + width / 2.0
            candidate_y = y1 + y + height / 2.0
            distance = abs(candidate_x - center_x) + abs(candidate_y - center_y)
            candidates.append((fill - 0.001 * distance,
                               candidate_x, candidate_y,
                               float(width), float(height), float(width * height)))

        if not candidates:
            return None
        best = max(candidates, key=lambda item: item[0])
        return best[1:]
