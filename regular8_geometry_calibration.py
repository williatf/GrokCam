"""Automatic, frozen per-capture Regular 8 geometry calibration.

The calibrator consumes the same full-sprocket detections and preview image
already used by RAW capture.  It estimates two deliberately separate domains:
the image-support measurements used by post-processing P15 and the capture ROI
measurements used by P24.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import statistics

import cv2
import numpy as np


SCHEMA_VERSION = 1
CALIBRATION_METHOD = "regular8_capture_geometry_v1"
MIN_VALID_SAMPLES = 18
STABILITY_WINDOW = 12

# Broad optical sanity bounds.  These deliberately include both the historical
# ~796 px pitch setup and the validated ~732 px setup.
SANITY_BOUNDS = {
    "sprocket_pitch_px": (500.0, 1000.0),
    "p15_image_support_width_px": (100.0, 500.0),
    "p15_image_support_height_px": (100.0, 400.0),
    "sprocket_x_px": (100.0, 700.0),
    "p24_capture_roi_pitch_px": (500.0, 1000.0),
    "p24_capture_roi_width_px": (200.0, 500.0),
    "p24_capture_roi_height_px": (150.0, 400.0),
    "p24_pair_x_displacement_px": (-100.0, 100.0),
}


def is_valid_frozen_geometry(value):
    """Return whether metadata contains a complete frozen capture geometry.

    Capture deliberately performs only a small local check here; the
    postprocess package remains authoritative for strict schema validation.
    """
    if not isinstance(value, dict):
        return False
    if value.get("schema_version") != SCHEMA_VERSION:
        return False
    if value.get("film_format") != "regular8" or value.get("frozen") is not True:
        return False
    required = (
        "calibration_frames", "sprocket_pitch_px", "sprocket_width_px",
        "sprocket_height_px", "sprocket_x_px", "p24_capture_pitch_px",
        "p24_capture_width_px", "p24_capture_height_px",
        "p24_capture_x_displacement_px",
    )
    if (not isinstance(value.get("calibration_frames"), list) or
            not value["calibration_frames"]):
        return False
    for key in required[1:]:
        if not isinstance(value.get(key), (int, float)) or isinstance(value[key], bool):
            return False
        if not math.isfinite(float(value[key])):
            return False
    for key, (low, high) in {
        "sprocket_pitch_px": SANITY_BOUNDS["sprocket_pitch_px"],
        "sprocket_width_px": SANITY_BOUNDS["p15_image_support_width_px"],
        "sprocket_height_px": SANITY_BOUNDS["p15_image_support_height_px"],
        "sprocket_x_px": SANITY_BOUNDS["sprocket_x_px"],
        "p24_capture_pitch_px": SANITY_BOUNDS["p24_capture_roi_pitch_px"],
        "p24_capture_width_px": SANITY_BOUNDS["p24_capture_roi_width_px"],
        "p24_capture_height_px": SANITY_BOUNDS["p24_capture_roi_height_px"],
        "p24_capture_x_displacement_px": SANITY_BOUNDS["p24_pair_x_displacement_px"],
    }.items():
        if not low <= float(value[key]) <= high:
            return False
    return True


def _mad(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return None
    center = statistics.median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _percentiles(values):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return {}
    return {
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values[0]),
        "max": float(values[-1]),
    }


def _robust(values, max_mad_scale=3.5):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return []
    center = statistics.median(values)
    mad = _mad(values) or 0.0
    limit = max(1.0, float(max_mad_scale) * mad)
    filtered = [value for value in values if abs(value - center) <= limit]
    return filtered or values


def _summary(values):
    filtered = _robust(values)
    if not filtered:
        return {"count": 0, "median": None, "mad": None, "percentiles": {}}
    return {
        "count": len(filtered),
        "median": float(statistics.median(filtered)),
        "mad": _mad(filtered),
        "percentiles": _percentiles(filtered),
    }


def _scale_box(box, preview_size, raw_size):
    preview_width, preview_height = preview_size
    raw_width, raw_height = raw_size
    scale_x = float(raw_width) / float(preview_width)
    scale_y = float(raw_height) / float(preview_height)
    return (
        float(box[0]) * scale_x,
        float(box[1]) * scale_y,
        float(box[2]) * scale_x,
        float(box[3]) * scale_y,
        float(box[4]) * scale_x * scale_y,
    )


def _choose_pair(sprockets, frame_shape, pitch_range=(500.0, 1000.0)):
    if len(sprockets) < 2:
        return None
    frame_height = float(frame_shape[0])
    frame_width = float(frame_shape[1])
    center_y = frame_height / 2.0
    candidates = []
    ordered = sorted(sprockets, key=lambda item: item[1])
    for index, upper in enumerate(ordered[:-1]):
        for lower in ordered[index + 1:]:
            pitch = float(lower[1]) - float(upper[1])
            if not pitch_range[0] <= pitch <= pitch_range[1]:
                continue
            midpoint = (float(upper[1]) + float(lower[1])) / 2.0
            score = abs(pitch - 785.0) / 300.0 + abs(midpoint - center_y) / max(frame_height, 1.0)
            score += abs(float(upper[0]) - float(lower[0])) / max(frame_width, 1.0)
            candidates.append((score, upper, lower))
    if not candidates:
        return None
    _, upper, lower = min(candidates, key=lambda item: item[0])
    return upper, lower


def _image_support_pair(frame_bgr):
    """Find the bright horizontal support used by P15 in a capture preview."""
    height, width = frame_bgr.shape[:2]
    # The production P15 experiment searches raw-coordinate columns 100..570
    # and requires at least 100 supporting columns. Apply that same geometry
    # after scaling the camera preview rather than using a looser percentage
    # ROI that can absorb picture content.
    raw_width = 2028.0
    # Match the established image-derived calibration, which uses mean RGB
    # luminance rather than introducing a new channel-weighted threshold.
    gray = frame_bgr.astype(np.float32).mean(axis=2)
    preview_scale_x = width / raw_width
    x0 = max(0, int(round(100.0 * preview_scale_x)))
    x1 = min(width, int(round(570.0 * preview_scale_x)))
    strip = gray[:, x0:x1]
    if strip.size == 0:
        return None
    threshold = float(np.percentile(strip, 99.0) * 0.90)
    bright = strip > threshold
    minimum_rows = max(20, int(round(130.0 * preview_scale_x)))
    row_mask = bright.sum(axis=1) > minimum_rows
    runs = []
    start = None
    for index, value in enumerate(row_mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(row_mask) - 1):
            end = index if not value else index + 1
            if height * 0.10 <= end - start <= height * 0.30:
                runs.append((start, end))
            start = None

    candidates = []
    for upper, lower in zip(runs, runs[1:]):
        upper_y = sum(upper) / 2.0
        lower_y = sum(lower) / 2.0
        pitch = lower_y - upper_y
        if not height * 0.32 <= pitch <= height * 0.72:
            continue
        boxes = []
        for top, bottom in (upper, lower):
            columns = np.flatnonzero(
                bright[top:bottom].sum(axis=0) > (bottom - top) * 0.55
            )
            if len(columns) < max(24, int(round(100.0 * preview_scale_x))):
                break
            left, right = int(columns[0]), int(columns[-1])
            boxes.append((
                (left + right) / 2.0 + x0,
                (top + bottom) / 2.0,
                right - left + 1,
                bottom - top,
                0.0,
            ))
        if len(boxes) == 2:
            score = abs(pitch - height * 0.48) / height
            score += abs(boxes[0][0] - boxes[1][0]) / max(width, 1)
            candidates.append((score, boxes))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


@dataclass
class Regular8GeometryCalibrator:
    source: str
    raw_size: tuple[int, int] = (2028, 1520)
    min_valid_samples: int = MIN_VALID_SAMPLES
    stability_window: int = STABILITY_WINDOW

    def __post_init__(self):
        self.samples = []
        self.rejected_samples = []
        self.total_samples = 0
        self.frozen_geometry = None
        self._last_signature = None

    @property
    def frozen(self):
        return self.frozen_geometry is not None

    def status(self):
        values = self._series(self.samples, "sprocket_pitch")
        state = "calibrated" if self.frozen else "calibrating"
        return {
            "state": state,
            "film_format": "regular8",
            "frozen": bool(self.frozen),
            "samples": len(self.samples),
            "valid_samples": len(self.samples),
            "rejected_samples": len(self.rejected_samples),
            "total_samples": self.total_samples,
            "minimum_samples": self.min_valid_samples,
            "stability_window": self.stability_window,
            "pitch_px": None if not values else float(statistics.median(_robust(values))),
            "pitch_mad_px": _mad(_robust(values)),
            "stability": self._stability_message(),
            "reason": None,
            "capture_geometry": self.frozen_geometry,
        }

    def add_frame(self, frame_id, frame_bgr, sprockets, *, transport_steps=None):
        self.total_samples += 1
        if self.frozen:
            return self.status()

        measurement, reason = self._measure(frame_id, frame_bgr, sprockets)
        if measurement is None:
            self.rejected_samples.append({"frame": int(frame_id), "reason": reason})
            result = self.status()
            result["reason"] = reason
            return result

        signature = self._signature(frame_bgr)
        if self._last_signature is not None and np.max(np.abs(signature - self._last_signature)) < 0.5:
            self.rejected_samples.append({"frame": int(frame_id), "reason": "stationary_frame"})
            result = self.status()
            result["reason"] = "stationary_frame"
            return result
        self._last_signature = signature
        measurement["frame"] = int(frame_id)
        measurement["transport_steps"] = None if transport_steps is None else int(transport_steps)
        self.samples.append(measurement)

        if self._is_stable():
            self.frozen_geometry = self._build_geometry()
        return self.status()

    def finalize(self):
        if self.frozen:
            return self.status()
        result = self.status()
        result["state"] = "failed"
        result["reason"] = self._failure_reason()
        return result

    def _measure(self, frame_id, frame_bgr, sprockets):
        if frame_bgr is None or not hasattr(frame_bgr, "shape"):
            return None, "invalid_preview_frame"
        support_pair = _image_support_pair(frame_bgr)
        if support_pair is None:
            return None, "no_image_support_pair"
        preview_scale_y = float(self.raw_size[1]) / float(frame_bgr.shape[0])
        capture_pair = _choose_pair(
            sprockets, frame_bgr.shape,
            pitch_range=(500.0 / preview_scale_y, 1000.0 / preview_scale_y),
        )
        if capture_pair is None:
            return None, "no_capture_pair"
        support_upper, support_lower = support_pair
        capture_upper, capture_lower = capture_pair
        preview_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        raw_upper = _scale_box(support_upper, preview_size, self.raw_size)
        raw_lower = _scale_box(support_lower, preview_size, self.raw_size)
        raw_capture_upper = _scale_box(capture_upper, preview_size, self.raw_size)
        raw_capture_lower = _scale_box(capture_lower, preview_size, self.raw_size)
        support_midpoint_x = (raw_upper[0] + raw_lower[0]) / 2.0
        support_midpoint_y = (raw_upper[1] + raw_lower[1]) / 2.0
        capture_midpoint_x = (raw_capture_upper[0] + raw_capture_lower[0]) / 2.0
        capture_midpoint_y = (raw_capture_upper[1] + raw_capture_lower[1]) / 2.0
        return {
            "sprocket_pitch": raw_lower[1] - raw_upper[1],
            "p15_image_support_width": raw_lower[2],
            "p15_image_support_height": raw_lower[3],
            "sprocket_x": support_midpoint_x,
            "lower_x_minus_midpoint": raw_lower[0] - support_midpoint_x,
            "lower_top_minus_midpoint": raw_lower[1] - raw_lower[3] / 2.0 - support_midpoint_y,
            "p24_capture_roi_pitch": raw_capture_lower[1] - raw_capture_upper[1],
            "p24_capture_roi_width": (raw_capture_upper[2] + raw_capture_lower[2]) / 2.0,
            "p24_capture_roi_height": (raw_capture_upper[3] + raw_capture_lower[3]) / 2.0,
            "p24_pair_x_displacement": raw_capture_lower[0] - raw_capture_upper[0],
            "p24_capture_lower_top_minus_midpoint": (
                raw_capture_lower[1] - raw_capture_lower[3] / 2.0 - capture_midpoint_y
            ),
            "p24_capture_lower_x_minus_midpoint": raw_capture_lower[0] - capture_midpoint_x,
        }, None

    @staticmethod
    def _signature(frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def _series(samples, key):
        return [sample[key] for sample in samples if key in sample]

    def _is_stable(self):
        if len(self.samples) < self.min_valid_samples:
            return False
        window = self.samples[-self.stability_window:]
        requirements = {
            "sprocket_pitch": 4.0,
            "p15_image_support_width": 8.0,
            "p15_image_support_height": 8.0,
            "lower_top_minus_midpoint": 4.0,
            "p24_capture_roi_pitch": 6.0,
            "p24_capture_roi_width": 12.0,
            "p24_capture_roi_height": 12.0,
            "p24_pair_x_displacement": 6.0,
        }
        for key, mad_limit in requirements.items():
            # Use the raw stability window for the MAD test. If a quantized
            # detector reports a zero MAD, aggressively filtering at a
            # one-pixel limit can incorrectly discard otherwise stable
            # measurements before the robust final median is formed.
            values = self._series(window, key)
            if len(values) < max(8, int(self.stability_window * 0.75)):
                return False
            if (_mad(values) or 0.0) > mad_limit:
                return False
        bound_series = {
            "sprocket_pitch_px": "sprocket_pitch",
            "p15_image_support_width_px": "p15_image_support_width",
            "p15_image_support_height_px": "p15_image_support_height",
            "sprocket_x_px": "sprocket_x",
            "p24_capture_roi_pitch_px": "p24_capture_roi_pitch",
            "p24_capture_roi_width_px": "p24_capture_roi_width",
            "p24_capture_roi_height_px": "p24_capture_roi_height",
            "p24_pair_x_displacement_px": "p24_pair_x_displacement",
        }
        return all(
            low <= statistics.median(self._series(window, key)) <= high
            for bound, (low, high) in SANITY_BOUNDS.items()
            for key in [bound_series.get(bound)]
            if key is not None
        )

    def _stability_message(self):
        if self.frozen:
            return "frozen"
        if len(self.samples) < self.min_valid_samples:
            return "not_yet_sufficient_samples"
        return "geometry_variation_above_threshold"

    def _failure_reason(self):
        if len(self.samples) < self.min_valid_samples:
            return "insufficient_stable_sprocket_measurements"
        return "geometry_variation_above_threshold"

    def _build_geometry(self):
        def median(key):
            return float(statistics.median(_robust(self._series(self.samples, key))))

        def mad(key):
            return float(_mad(_robust(self._series(self.samples, key))) or 0.0)

        quality = {
            key: _summary(self._series(self.samples, key))
            for key in (
                "sprocket_pitch", "p15_image_support_width",
                "p15_image_support_height", "sprocket_x",
                "lower_x_minus_midpoint", "lower_top_minus_midpoint",
                "p24_capture_roi_pitch", "p24_capture_roi_width",
                "p24_capture_roi_height", "p24_pair_x_displacement",
                "p24_capture_lower_top_minus_midpoint",
                "p24_capture_lower_x_minus_midpoint",
            )
        }
        # Postprocess accepts these stable public quality names. Keep the
        # P24-prefixed names as additional diagnostic detail.
        quality["capture_lower_top_minus_midpoint"] = quality[
            "p24_capture_lower_top_minus_midpoint"
        ]
        quality["capture_lower_x_minus_midpoint"] = quality[
            "p24_capture_lower_x_minus_midpoint"
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "film_format": "regular8",
            "calibration_method": CALIBRATION_METHOD,
            "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "frozen": True,
            "calibration_frames": [sample["frame"] for sample in self.samples],
            "sprocket_pitch_px": median("sprocket_pitch"),
            "sprocket_pitch_mad_px": mad("sprocket_pitch"),
            "sprocket_width_px": median("p15_image_support_width"),
            "sprocket_width_mad_px": mad("p15_image_support_width"),
            "sprocket_height_px": median("p15_image_support_height"),
            "sprocket_height_mad_px": mad("p15_image_support_height"),
            "sprocket_x_px": median("sprocket_x"),
            "sprocket_x_mad_px": mad("sprocket_x"),
            "p15_image_support_width_px": median("p15_image_support_width"),
            "p15_image_support_height_px": median("p15_image_support_height"),
            "lower_x_minus_midpoint_px": median("lower_x_minus_midpoint"),
            "lower_top_minus_midpoint_px": median("lower_top_minus_midpoint"),
            "lower_top_residual_mad_px": mad("lower_top_minus_midpoint"),
            "p24_capture_pitch_px": median("p24_capture_roi_pitch"),
            "p24_capture_pitch_mad_px": mad("p24_capture_roi_pitch"),
            "p24_capture_width_px": median("p24_capture_roi_width"),
            "p24_capture_width_mad_px": mad("p24_capture_roi_width"),
            "p24_capture_height_px": median("p24_capture_roi_height"),
            "p24_capture_height_mad_px": mad("p24_capture_roi_height"),
            "p24_capture_x_displacement_px": median("p24_pair_x_displacement"),
            "p24_capture_x_displacement_mad_px": mad("p24_pair_x_displacement"),
            "quality": quality,
            "sanity_bounds": {key: list(value) for key, value in SANITY_BOUNDS.items()},
        }
