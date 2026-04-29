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
        debug_frame = frame.copy()

        for cx, cy, width, height, _area in sprockets:
            x1 = int(round(cx - width / 2))
            y1 = int(round(cy - height / 2))
            x2 = int(round(cx + width / 2))
            y2 = int(round(cy + height / 2))

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(
                debug_frame,
                (int(round(cx)), int(round(cy))),
                4,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                debug_frame,
                f"cy={cy:.1f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
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

        measurement_dict = self._build_measurements(sprockets)
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

    def _build_measurements(self, sprockets):
        measurements = {
            "sprocket_count": len(sprockets),
            "sprocket_pitch_px": None,
            "sprocket_area_nominal": None,
            "sprocket_area_min": None,
            "sprocket_area_max": None,
        }

        if not sprockets:
            return measurements

        sprockets_sorted = sorted(sprockets, key=lambda sprocket: sprocket[1])
        centers_y = np.array([float(sprocket[1]) for sprocket in sprockets_sorted], dtype=float)
        areas = np.array([float(sprocket[4]) for sprocket in sprockets_sorted], dtype=float)

        if centers_y.size >= 2:
            measurements["sprocket_pitch_px"] = float(np.mean(np.diff(centers_y)))

        nominal_area = float(np.mean(areas))
        measurements["sprocket_area_nominal"] = nominal_area
        measurements["sprocket_area_min"] = nominal_area * 0.8
        measurements["sprocket_area_max"] = nominal_area * 1.2

        return measurements