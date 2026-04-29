import cv2
import json
import os
import time
import shutil
import io

import numpy as np


class CalibrationService:
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

    def save_calibration(self, calibration_dict):
        backup_path = None
        if os.path.exists(self.save_path):
            root, ext = os.path.splitext(self.save_path)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_ext = ext or ".json"
            backup_path = f"{root}.backup.{timestamp}{backup_ext}"
            shutil.copy2(self.save_path, backup_path)

        with open(self.save_path, "w", encoding="utf-8") as handle:
            json.dump(calibration_dict, handle, indent=2)
            handle.write("\n")

        return backup_path

    def _build_measurements(self, classified_sprockets):
        measurements = {
            "sprocket_count": len(classified_sprockets),
            "full_sprocket_count": 0,
            "partial_sprocket_count": 0,
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

        accepted_sprockets = [item["sprocket"] for item in classified_sprockets]
        areas = np.array([float(sprocket[4]) for sprocket in accepted_sprockets], dtype=float)

        if len(full_sprockets) == 2:
            full_sorted = sorted(full_sprockets, key=lambda sprocket: sprocket[1])
            measurements["sprocket_pitch_px"] = float(abs(full_sorted[1][1] - full_sorted[0][1]))
            measurements["pitch_valid"] = True
            measurements["pitch_reason"] = "exactly_two_full_sprockets"

        nominal_area = float(np.mean(areas))
        measurements["sprocket_area_nominal"] = nominal_area
        measurements["sprocket_area_min"] = nominal_area * 0.8
        measurements["sprocket_area_max"] = nominal_area * 1.2

        return measurements