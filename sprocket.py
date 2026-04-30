import cv2 as cv
import numpy as np


class SprocketDetector:
    def __init__(self, side="left", auto_roi=0.25,
                 min_area=3000, max_area=6000,
                 ar_min=1.2, ar_max=1.6,
                 solidity_min=0.3, edge_margin_frac=1.0,
                 blur=5, open_k=7, close_k=3,
                 adaptive_block=51, adaptive_C=5,
                 method="adaptive", inv=True):

        self.side = side
        self.auto_roi = auto_roi
        self.min_area = min_area
        self.max_area = max_area
        self.ar_min = ar_min
        self.ar_max = ar_max
        self.solidity_min = solidity_min
        self.edge_margin_frac = edge_margin_frac
        self.blur = blur
        self.open_k = open_k
        self.close_k = close_k
        self.adaptive_block = adaptive_block
        self.adaptive_C = adaptive_C
        self.method = method
        self.inv = inv

        # Calibration hints
        self.expected_width = None
        self.expected_pitch = None
        self.width_tol = 0.25
        self.pitch_tol = 12
        self.expected_ar = 1.5

    def update_pitch(self, pitch_px):
        """Update expected sprocket pitch from external measurement."""
        if self.expected_pitch is None:
            self.expected_pitch = int(pitch_px)
            print(f"[SPROCKET] Initialized expected_pitch={self.expected_pitch}px")
        else:
            self.expected_pitch = int(0.8 * self.expected_pitch + 0.2 * pitch_px)
            print(f"[SPROCKET] Updated expected_pitch={self.expected_pitch}px")

    def detect(self, frame_bgr, debug_prefix=None, mode="profile"):
        """Detect plausible sprocket candidates in the configured side ROI."""
        frame_h, frame_w = frame_bgr.shape[:2]
        strip_w = max(1, int(frame_w * self.auto_roi))

        if self.side == "left":
            roi = frame_bgr[:, :strip_w]
            roi_offset = (0, 0)
        else:
            roi = frame_bgr[:, -strip_w:]
            roi_offset = (frame_w - strip_w, 0)

        if mode in ("profile", "contour"):
            sprockets = self._detect_contours_scored(roi, roi_offset, frame_bgr, debug_prefix)
            if len(sprockets) < 2:
                profile_sprockets = self._detect_profile(roi, roi_offset, frame_bgr, None)
                sprockets = self._merge_sprocket_lists(sprockets, profile_sprockets, frame_bgr.shape)
            return sorted(sprockets, key=lambda sprocket: sprocket[1])

        return self._detect_profile(roi, roi_offset, frame_bgr, debug_prefix)

    def classify_sprockets(self, sprockets, frame_shape, edge_margin_px=None):
        frame_h = frame_shape[0]
        if edge_margin_px is None:
            edge_margin_px = max(10, int(frame_h * 0.02))

        classified = []
        for sprocket in sorted(sprockets, key=lambda item: item[1]):
            cx, cy, width, height, area = sprocket
            status = "partial" if self._touches_vertical_edge(
                cy,
                height,
                frame_h,
                edge_margin_px=edge_margin_px,
            ) else "full"
            classified.append({
                "sprocket": sprocket,
                "status": status,
                "edge_margin_px": edge_margin_px,
                "bbox": (
                    float(cx) - float(width) / 2.0,
                    float(cy) - float(height) / 2.0,
                    float(cx) + float(width) / 2.0,
                    float(cy) + float(height) / 2.0,
                ),
                "area": float(area),
            })

        return classified

    def choose_registration_pair(self, sprockets, frame_shape, expected_pitch=None):
        if len(sprockets) < 2:
            return None

        frame_h = frame_shape[0]
        center_y = frame_h / 2.0
        pitch_target = expected_pitch if expected_pitch is not None else self.expected_pitch
        pitch_tol = None
        if pitch_target is not None:
            pitch_tol = max(float(self.pitch_tol), float(pitch_target) * 0.35)

        best_pair = None
        best_score = -1.0
        sprockets_sorted = sorted(sprockets, key=lambda sprocket: sprocket[1])

        for top_index in range(len(sprockets_sorted) - 1):
            top = sprockets_sorted[top_index]
            for bottom in sprockets_sorted[top_index + 1:]:
                sep = float(bottom[1] - top[1])
                if sep <= 0:
                    continue

                if pitch_target is not None and abs(sep - pitch_target) > pitch_tol:
                    continue

                if pitch_target is None:
                    if sep < frame_h * 0.08 or sep > frame_h * 0.9:
                        continue
                    pitch_score = 0.5
                else:
                    pitch_score = self._clamp01(1.0 - (abs(sep - pitch_target) / (pitch_tol + 1.0)))

                midpoint = (top[1] + bottom[1]) / 2.0
                midpoint_score = self._clamp01(1.0 - (abs(midpoint - center_y) / (frame_h / 2.0 + 1.0)))
                top_score = self._score_sprocket_tuple(top, frame_shape)
                bottom_score = self._score_sprocket_tuple(bottom, frame_shape)

                pair_score = (
                    0.45 * pitch_score
                    + 0.30 * midpoint_score
                    + 0.125 * top_score
                    + 0.125 * bottom_score
                )

                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (top, bottom)

        if best_pair is None or best_score < 0.30:
            return None

        return best_pair

    def choose_anchor_sprocket(self, sprockets, frame_shape):
        if not sprockets:
            return None

        best_sprocket = None
        best_score = -1.0
        for sprocket in sprockets:
            score = self._score_sprocket_tuple(sprocket, frame_shape)
            if score > best_score:
                best_score = score
                best_sprocket = sprocket

        if best_sprocket is None or best_score < 0.20:
            return None

        return best_sprocket

    def choose_registration(self, sprockets, frame_shape, expected_pitch=None):
        frame_h = frame_shape[0]
        target_y = frame_h / 2.0

        pair = self.choose_registration_pair(sprockets, frame_shape, expected_pitch=expected_pitch)
        if pair is not None:
            actual_y = (pair[0][1] + pair[1][1]) / 2.0
            return {
                "mode": "pair",
                "pair": pair,
                "anchor": None,
                "target_y": target_y,
                "actual_y": actual_y,
                "error_px": target_y - actual_y,
            }

        anchor = self.choose_anchor_sprocket(sprockets, frame_shape)
        if anchor is not None:
            actual_y = float(anchor[1])
            return {
                "mode": "single",
                "pair": None,
                "anchor": anchor,
                "target_y": target_y,
                "actual_y": actual_y,
                "error_px": target_y - actual_y,
            }

        return {
            "mode": "none",
            "pair": None,
            "anchor": None,
            "target_y": None,
            "actual_y": None,
            "error_px": None,
        }

    def _detect_contours_scored(self, roi, roi_offset, frame_bgr, debug_prefix=None):
        roi_h, roi_w = roi.shape[:2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        blurred = gray

        blur_k = self._odd_kernel(self.blur)
        if blur_k > 1:
            blurred = cv.GaussianBlur(gray, (blur_k, blur_k), 0)

        block_size = max(3, self._odd_kernel(self.adaptive_block))
        binary_flag = cv.THRESH_BINARY if self.inv else cv.THRESH_BINARY_INV
        adaptive = cv.adaptiveThreshold(
            blurred,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            binary_flag,
            block_size,
            self.adaptive_C,
        )
        _, otsu = cv.threshold(blurred, 0, 255, binary_flag | cv.THRESH_OTSU)
        mask = cv.bitwise_or(adaptive, otsu)

        if self.open_k > 1:
            open_kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE,
                (self._odd_kernel(self.open_k), self._odd_kernel(self.open_k)),
            )
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, open_kernel)

        if self.close_k > 1:
            close_kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE,
                (self._odd_kernel(self.close_k), self._odd_kernel(self.close_k)),
            )
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidate_dicts = []
        rejected_dicts = []

        for contour in contours:
            contour_area = float(cv.contourArea(contour))
            if contour_area <= 0:
                continue

            x, y, w, h = cv.boundingRect(contour)
            if min(w, h) < 4:
                continue

            bbox = (
                x + roi_offset[0],
                y + roi_offset[1],
                x + roi_offset[0] + w,
                y + roi_offset[1] + h,
            )

            if not self._passes_dimension_gate(w, h):
                if debug_prefix is not None:
                    rejected_dicts.append({"bbox": bbox, "reason": "dims"})
                continue

            bbox_area = float(w * h)
            if bbox_area < self.min_area * 0.20 or bbox_area > self.max_area * 2.25:
                if debug_prefix is not None:
                    rejected_dicts.append({"bbox": bbox, "reason": "area"})
                continue

            ar = float(w) / float(h) if h > 0 else 0.0
            if ar < self.ar_min or ar > self.ar_max:
                if debug_prefix is not None:
                    rejected_dicts.append({"bbox": bbox, "reason": "aspect"})
                continue

            hull = cv.convexHull(contour)
            hull_area = float(cv.contourArea(hull)) if hull is not None else 0.0
            solidity = contour_area / hull_area if hull_area > 0 else 0.0
            if solidity < max(0.15, self.solidity_min * 0.5):
                if debug_prefix is not None:
                    rejected_dicts.append({"bbox": bbox, "reason": "solidity"})
                continue

            cx = x + (w / 2.0) + roi_offset[0]
            cy = y + (h / 2.0) + roi_offset[1]
            sprocket_tuple = (cx, cy, float(w), float(h), bbox_area)
            score = self._score_sprocket_geometry(
                cx,
                cy,
                float(w),
                float(h),
                bbox_area,
                solidity,
                frame_bgr.shape,
            )
            if score < 0.12:
                if debug_prefix is not None:
                    rejected_dicts.append({"bbox": bbox, "reason": "score"})
                continue

            candidate_dicts.append({
                "tuple": sprocket_tuple,
                "score": score,
                "bbox": bbox,
            })

        candidate_dicts = self._dedupe_candidates(candidate_dicts)
        candidate_dicts.sort(key=lambda item: item["tuple"][1])
        sprockets = [item["tuple"] for item in candidate_dicts]

        if debug_prefix is not None:
            dbg = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            edge_margin = self._edge_margin_pixels(roi_h)
            cv.line(dbg, (0, edge_margin), (roi_w - 1, edge_margin), (80, 80, 255), 1)
            cv.line(dbg, (0, roi_h - edge_margin), (roi_w - 1, roi_h - edge_margin), (80, 80, 255), 1)
            for item in rejected_dicts:
                x1, y1, x2, y2 = item["bbox"]
                cv.rectangle(
                    dbg,
                    (int(round(x1 - roi_offset[0])), int(round(y1 - roi_offset[1]))),
                    (int(round(x2 - roi_offset[0])), int(round(y2 - roi_offset[1]))),
                    (0, 0, 255),
                    1,
                )
            for item in candidate_dicts:
                cx, cy, w, h, _ = item["tuple"]
                local_cx = cx - roi_offset[0]
                local_cy = cy - roi_offset[1]
                color = (0, 255, 255)
                if self._touches_vertical_edge(local_cy, h, roi_h):
                    color = (0, 140, 255)
                cv.rectangle(
                    dbg,
                    (int(round(local_cx - w / 2.0)), int(round(local_cy - h / 2.0))),
                    (int(round(local_cx + w / 2.0)), int(round(local_cy + h / 2.0))),
                    color,
                    2,
                )
                cv.putText(
                    dbg,
                    f"{item['score']:.2f}",
                    (int(round(local_cx - w / 2.0)), int(round(local_cy - h / 2.0)) - 4),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            cv.imwrite(f"{debug_prefix}_contour_dbg.jpg", dbg)

        return sprockets

    def _detect_profile(self, roi, roi_offset, frame_bgr, debug_prefix=None):
        """Legacy profile-based sprocket detection used as a fallback."""
        roi_h, roi_w = roi.shape[:2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        mid_x = roi_w // 2
        column = gray[:, mid_x]
        col_norm = column / 255.0
        thresh_val = np.max(col_norm) * 0.95 if col_norm.size else 1.0
        mask_y = np.where(col_norm > thresh_val)[0]

        if len(mask_y) > 0:
            splits = np.where(np.diff(mask_y) > 5)[0] + 1
            bands = np.split(mask_y, splits)
        else:
            bands = []

        sprockets = []
        for band in bands:
            if len(band) < 5:
                continue

            y_top = int(band[0])
            y_bot = int(band[-1])
            h = max(1, y_bot - y_top)
            cy = (y_top + y_bot) / 2.0 + roi_offset[1]

            row_y = int(y_top + 0.25 * (y_bot - y_top))
            row_y = max(0, min(roi_h - 1, row_y))
            row = gray[row_y, :]
            peak_val = int(np.max(row)) if row.size else 0
            if peak_val < 180:
                continue

            thresh_row = peak_val * 0.95
            x_left = mid_x
            while x_left > 0 and row[x_left] > thresh_row:
                x_left -= 1
            x_right = mid_x
            while x_right < roi_w - 1 and row[x_right] > thresh_row:
                x_right += 1

            w = max(1, x_right - x_left)
            cx = (x_left + x_right) / 2.0 + roi_offset[0]
            area = float(w * h)
            ar = float(w) / float(h) if h > 0 else 0.0

            if not self._passes_dimension_gate(w, h):
                continue

            if ar < self.ar_min or ar > self.ar_max:
                continue

            if not (self.ar_min <= ar <= self.ar_max) and self.expected_ar:
                h = int(max(1, w / self.expected_ar))
                cy = (y_top + (y_top + h)) / 2.0 + roi_offset[1]
                area = float(w * h)

            local_cy = cy - roi_offset[1]
            if self._touches_vertical_edge(local_cy, h, roi_h) and area < self.min_area * 0.6:
                continue
            if area < self.min_area * 0.35 or area > self.max_area * 1.5:
                continue

            score = self._score_sprocket_geometry(cx, cy, float(w), float(h), area, 0.75, frame_bgr.shape)
            if score >= 0.10:
                sprockets.append((cx, cy, float(w), float(h), area))

        sprockets = sorted(sprockets, key=lambda sprocket: sprocket[1])

        if debug_prefix is not None:
            dbg = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            dbg[:, mid_x] = (0, 0, 255)
            for cx, cy, w, h, _ in sprockets:
                local_cx = cx - roi_offset[0]
                local_cy = cy - roi_offset[1]
                color = (0, 255, 255)
                if self._touches_vertical_edge(local_cy, h, roi_h):
                    color = (0, 140, 255)
                cv.rectangle(
                    dbg,
                    (int(round(local_cx - w / 2.0)), int(round(local_cy - h / 2.0))),
                    (int(round(local_cx + w / 2.0)), int(round(local_cy + h / 2.0))),
                    color,
                    2,
                )
                cv.circle(dbg, (int(round(local_cx)), int(round(local_cy))), 3, (255, 0, 0), -1)
            cv.imwrite(f"{debug_prefix}_profile_dbg.jpg", dbg)

        return sprockets

    def _merge_sprocket_lists(self, primary, secondary, frame_shape):
        merged = list(primary)
        for sprocket in secondary:
            duplicate = False
            for existing in merged:
                if self._tuple_overlap(existing, sprocket) > 0.45:
                    duplicate = True
                    break
                if abs(existing[1] - sprocket[1]) < max(8.0, min(existing[3], sprocket[3]) * 0.4):
                    duplicate = True
                    break
            if not duplicate:
                merged.append(sprocket)

        merged.sort(
            key=lambda sprocket: (
                sprocket[1],
                -self._score_sprocket_tuple(sprocket, frame_shape),
            )
        )
        return merged

    def _dedupe_candidates(self, candidates):
        if not candidates:
            return []

        kept = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            is_duplicate = False
            for existing in kept:
                if self._bbox_iou(candidate["bbox"], existing["bbox"]) > 0.45:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)
        return kept

    def _score_sprocket_tuple(self, sprocket, frame_shape):
        cx, cy, w, h, area = sprocket
        return self._score_sprocket_geometry(cx, cy, w, h, area, max(self.solidity_min, 0.75), frame_shape)

    def _score_sprocket_geometry(self, cx, cy, w, h, area, solidity, frame_shape):
        frame_h = frame_shape[0]
        ar = float(w) / float(h) if h > 0 else 0.0

        area_score = self._range_closeness(area, float(self.min_area), float(self.max_area))
        ar_tol = max(0.25, self.expected_ar * 0.35)
        ar_score = self._clamp01(1.0 - (abs(ar - self.expected_ar) / ar_tol))
        solidity_score = self._clamp01((float(solidity) - max(0.15, self.solidity_min * 0.5)) / 0.5)
        vertical_score = self._clamp01(1.0 - (abs(float(cy) - (frame_h / 2.0)) / (frame_h / 2.0 + 1.0)))

        score = (
            0.30 * area_score
            + 0.25 * ar_score
            + 0.20 * solidity_score
            + 0.25 * vertical_score
        )

        if self._touches_vertical_edge(cy, h, frame_h):
            score -= 0.35

        return self._clamp01(score)

    def _passes_dimension_gate(self, w, h):
        if w <= 0 or h <= 0:
            return False
        if float(h) > float(w) * 1.25:
            return False
        return True

    def _touches_vertical_edge(self, cy, h, frame_h, edge_margin_px=None):
        margin = edge_margin_px if edge_margin_px is not None else self._edge_margin_pixels(frame_h)
        y1 = float(cy) - (float(h) / 2.0)
        y2 = float(cy) + (float(h) / 2.0)
        return y1 <= margin or y2 >= (frame_h - margin)

    def _edge_margin_pixels(self, frame_h):
        base_margin = max(2, int(frame_h * 0.01))
        return max(2, int(base_margin * max(0.25, self.edge_margin_frac)))

    def _range_closeness(self, value, low, high):
        if high <= low:
            return 1.0
        center = (low + high) / 2.0
        half_range = (high - low) / 2.0
        if half_range <= 0:
            return 1.0
        return self._clamp01(1.0 - (abs(float(value) - center) / (half_range * 1.5 + 1.0)))

    def _tuple_overlap(self, left, right):
        return self._bbox_iou(self._tuple_to_bbox(left), self._tuple_to_bbox(right))

    def _tuple_to_bbox(self, sprocket):
        cx, cy, w, h, _ = sprocket
        return (
            float(cx) - float(w) / 2.0,
            float(cy) - float(h) / 2.0,
            float(cx) + float(w) / 2.0,
            float(cy) + float(h) / 2.0,
        )

    def _bbox_iou(self, left_box, right_box):
        x1 = max(left_box[0], right_box[0])
        y1 = max(left_box[1], right_box[1])
        x2 = min(left_box[2], right_box[2])
        y2 = min(left_box[3], right_box[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        left_area = max(0.0, (left_box[2] - left_box[0]) * (left_box[3] - left_box[1]))
        right_area = max(0.0, (right_box[2] - right_box[0]) * (right_box[3] - right_box[1]))
        denom = left_area + right_area - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    def _odd_kernel(self, value):
        kernel = max(1, int(value))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def _clamp01(self, value):
        return max(0.0, min(1.0, float(value)))
