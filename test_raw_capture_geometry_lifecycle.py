import ast
import asyncio
import contextlib
import io
import json
import os
import re
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from film_calibration import FILM_FORMAT_REGULAR8, FILM_FORMAT_SUPER8
from registration import RegistrationTracker
from regular8_geometry_calibration import (
    Regular8GeometryCalibrator,
    is_valid_frozen_geometry,
)
from sprocket import SprocketDetector
from super8_debug_overlay import annotate_super8_debug_preview
from super8_phase_tracker import (
    Super8PhaseTracker,
    select_super8_crop_guidance,
    unwrap_super8_y_near,
)
from takeup_interval_controller import AdaptiveTakeupIntervalController
from test_regular8_geometry_calibration import preview, sprockets
from transport_calibration import (
    AdaptiveTransportController,
    calculate_super8_stage1_transport,
)


async def _no_sleep(*args, **kwargs):
    return None


class _Request:
    def __init__(self, camera, frame_index):
        self.camera = camera
        self.frame_index = frame_index
        self.released = False

    def get_metadata(self):
        return {
            "SensorTimestamp": time.monotonic_ns() + 1_000_000_000,
            "ExposureTime": 3300,
            "AnalogueGain": 1.0,
        }

    def make_array(self, _stream):
        return preview(self.frame_index)

    def save_dng(self, path):
        Path(path).write_bytes(b"fixture-dng")
        if self.frame_index == self.camera.fail_dng_frame:
            raise OSError("injected DNG writer failure")

    def release(self):
        self.released = True


class _Camera:
    def __init__(self, fail_dng_frame=None):
        self.fail_dng_frame = fail_dng_frame
        self.requests = []

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def capture_request(self):
        request = _Request(self, len(self.requests) + 1)
        self.requests.append(request)
        return request


class _Transport:
    TAKEUP_PULSE_DURATION = 0.2

    def __init__(self):
        self.commands = []
        self.cleaned = False
        self.takeup_begin_intervals = []
        self.takeup_interval_updates = []
        self.takeup_ended = 0

    def light_on(self):
        pass

    def steps_forward(self, steps):
        self.commands.append(int(steps))

    def get_last_takeup_telemetry(self):
        return {}

    def begin_takeup_capture(self, _interval):
        self.takeup_begin_intervals.append(int(_interval))

    def set_takeup_interval_frames(self, interval):
        self.takeup_interval_updates.append(int(interval))

    def end_takeup_capture(self):
        self.takeup_ended += 1

    def clean_up(self):
        self.cleaned = True


class _FastDetector:
    last_failure = None

    def reset(self):
        self.last_failure = None

    def detect(self, _frame):
        return sprockets()

    def seed(self, *_args):
        pass


class _Super8Detector:
    def reset(self):
        pass

    def detect_candidates(self, _frame):
        self.last_result = {
            "sprockets": [], "partial_count": 0,
            "failure_reason": "no_candidates", "confidence": 0.0,
            "threshold": 120, "candidate_count": 0, "viable_count": 0,
            "actual_y": None,
        }
        return []


class _WebSocket:
    def __init__(self, *, fail_sample=None, fail_final=False):
        self.events = []
        self.fail_sample = fail_sample
        self.fail_final = fail_final

    async def send(self, payload):
        if isinstance(payload, bytes):
            return
        event = json.loads(payload)
        if (event.get("event") == "regular8_geometry_status"
                and event.get("samples") == self.fail_sample):
            raise RuntimeError("injected per-frame telemetry failure")
        if (self.fail_final and event.get("event") == "regular8_geometry_status"
                and event.get("state") == "failed"):
            raise RuntimeError("injected final telemetry failure")
        self.events.append(event)


def _capture_namespace(project_path, *, film_format=FILM_FORMAT_REGULAR8,
                       fail_dng_frame=None):
    app_path = Path(__file__).with_name("app.py")
    module = ast.parse(app_path.read_text(encoding="utf-8"))
    definitions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    namespace = globals().copy()
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(app_path), "exec"), namespace)

    camera = _Camera(fail_dng_frame)
    transport = _Transport()
    namespace.update({
        "active_project_path": str(project_path),
        "RAW_CAPTURE_MODE": "raw_dng_v1",
        "RAW_SENSOR_SIZE": (2028, 1520),
        "RAW_PREVIEW_SIZE": (760, 570),
        "FILM_FORMAT_REGULAR8": FILM_FORMAT_REGULAR8,
        "FILM_FORMAT_SUPER8": FILM_FORMAT_SUPER8,
        "RAW_SAFE_DEFAULT_EXPOSURE_TIME": 1114,
        "DEFAULT_CAMERA_EXPOSURE_TIME": 3300,
        "GAIN": 1.0,
        "raw_preview_scale": 570 / 1520,
        "settings": {},
        "camera": camera,
        "tc": transport,
        "asyncio": SimpleNamespace(sleep=_no_sleep, get_running_loop=asyncio.get_running_loop),
        "configure_raw_camera": lambda: None,
        "configure_legacy_camera": lambda: None,
        "apply_raw_capture_camera_controls": lambda: {
            "ExposureTime": 3300, "AnalogueGain": 1.0,
        },
        "get_project_film_format": lambda _path: film_format,
        "load_film_calibration": lambda _format: SimpleNamespace(
            source_name="test_fixture", status="formal",
        ),
        "resolve_raw_preview_transport": lambda *_args: {
            "steps_per_pitch": 277,
            "preview_sprocket_pitch_px": 294.4,
            "preview_pixels_per_step": 294.4 / 277,
            "pixels_per_step_source": "test_fixture",
        },
        "raw_fast_detector": _FastDetector(),
        "raw_super8_detector": _Super8Detector(),
        "raw_fallback_detector": SprocketDetector(),
        "RegistrationTracker": RegistrationTracker,
        "Super8PhaseTracker": Super8PhaseTracker,
        "AdaptiveTakeupIntervalController": AdaptiveTakeupIntervalController,
        "AdaptiveTransportController": AdaptiveTransportController,
        "calculate_super8_stage1_transport": calculate_super8_stage1_transport,
        "select_super8_crop_guidance": select_super8_crop_guidance,
        "unwrap_super8_y_near": unwrap_super8_y_near,
        "annotate_super8_debug_preview": annotate_super8_debug_preview,
        "get_scaled_relative_crop_rect": lambda *_args, **_kwargs: (
            (0, 0, 760, 570),
            {"crop_center_y": 285, "crop_clamped": False},
        ),
        "camera_transport": transport,
        "is_valid_frozen_geometry": is_valid_frozen_geometry,
    })
    return namespace, camera, transport


class RawCaptureGeometryLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="grokcam-geometry-lifecycle-")
        self.addCleanup(self.tempdir.cleanup)

    def make_project(self, name):
        project_path = Path(self.tempdir.name) / name
        (project_path / "metadata.json").parent.mkdir(parents=True, exist_ok=True)
        (project_path / "metadata.json").write_text(json.dumps({
            "film_format": FILM_FORMAT_REGULAR8,
            "custom_field": {"preserved": True},
        }), encoding="utf-8")
        return project_path

    async def run_capture(self, name, *, frames=20, fail_sample=None,
                          fail_final=False, fail_dng_frame=None,
                          film_format=FILM_FORMAT_REGULAR8,
                          calibrator_factory=Regular8GeometryCalibrator):
        project_path = self.make_project(name)
        namespace, camera, transport = _capture_namespace(
            project_path,
            film_format=film_format,
            fail_dng_frame=fail_dng_frame,
        )
        namespace["Regular8GeometryCalibrator"] = calibrator_factory
        websocket = _WebSocket(fail_sample=fail_sample, fail_final=fail_final)
        stop_event = asyncio.Event()
        error = None
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                await namespace["run_raw_capture"](websocket, frames, stop_event)
            except Exception as exc:  # surfaced to assert capture result
                error = exc
        metadata = json.loads((project_path / "metadata.json").read_text(encoding="utf-8"))
        committed = [
            path for path in (project_path / "raw").glob("frame_*.dng")
            if ".partial." not in path.name
        ]
        partial = list((project_path / "raw").glob("*.partial.dng"))
        return {
            "error": error, "metadata": metadata, "committed": committed,
            "partial": partial, "websocket": websocket,
            "camera": camera, "transport": transport,
            "requests_released": all(request.released for request in camera.requests),
        }

    async def test_dng_failure_is_not_a_calibration_sample(self):
        result = await self.run_capture("dng-failure", frames=20, fail_dng_frame=18)
        self.assertIsInstance(result["error"], OSError)
        self.assertEqual(len(result["committed"]), 17)
        self.assertEqual(len(result["partial"]), 1)
        status = result["metadata"]["capture_geometry_status"]
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["samples"], 17)
        self.assertNotIn(18, status["capture_geometry"]["calibration_frames"]
                         if result["metadata"].get("capture_geometry") else [])
        self.assertNotIn("capture_geometry", result["metadata"])
        self.assertTrue(result["requests_released"])
        self.assertTrue(result["transport"].cleaned)
        self.assertTrue(result["camera"].stopped)

    async def test_per_frame_geometry_send_failure_does_not_interrupt_capture(self):
        baseline = await self.run_capture("per-frame-baseline", frames=20)
        failed_send = await self.run_capture(
            "per-frame-send-failure", frames=20, fail_sample=3,
        )
        self.assertIsNone(failed_send["error"])
        self.assertEqual(len(failed_send["committed"]), 20)
        self.assertEqual(failed_send["partial"], [])
        self.assertTrue(failed_send["requests_released"])
        self.assertEqual(
            failed_send["metadata"]["transport_calibration_state"],
            baseline["metadata"]["transport_calibration_state"],
        )
        failed_geometry = failed_send["metadata"]["capture_geometry"]
        baseline_geometry = baseline["metadata"]["capture_geometry"]
        for key in (
            "calibration_frames", "sprocket_pitch_px", "sprocket_width_px",
            "sprocket_height_px", "sprocket_x_px", "p24_capture_pitch_px",
        ):
            self.assertEqual(failed_geometry[key], baseline_geometry[key])

    async def test_final_geometry_send_failure_does_not_skip_transport_save(self):
        baseline = await self.run_capture("final-baseline", frames=3)
        failed_send = await self.run_capture(
            "final-send-failure", frames=3, fail_final=True,
        )
        self.assertIsNone(failed_send["error"])
        self.assertEqual(
            failed_send["metadata"]["capture_geometry_status"],
            baseline["metadata"]["capture_geometry_status"],
        )
        self.assertEqual(
            failed_send["metadata"]["transport_calibration_state"],
            baseline["metadata"]["transport_calibration_state"],
        )
        self.assertTrue(failed_send["transport"].cleaned)
        self.assertTrue(failed_send["requests_released"])
        self.assertTrue(failed_send["camera"].stopped)

    async def test_super8_does_not_construct_regular8_calibrator(self):
        def fail_if_created(*_args, **_kwargs):
            raise AssertionError("Super 8 must not create a Regular 8 calibrator")

        result = await self.run_capture(
            "super8", frames=3, film_format=FILM_FORMAT_SUPER8,
            calibrator_factory=fail_if_created,
        )
        self.assertIsNone(result["error"])
        self.assertFalse(any(
            event.get("event") == "regular8_geometry_status"
            for event in result["websocket"].events
        ))
        self.assertNotIn("capture_geometry_status", result["metadata"])

    async def test_regular8_uses_capture_local_adaptive_takeup(self):
        result = await self.run_capture("regular8-takeup", frames=3)
        self.assertIsNone(result["error"])
        self.assertEqual(result["transport"].takeup_begin_intervals, [10])
        self.assertEqual(result["transport"].takeup_ended, 1)

    async def test_super8_retains_existing_adaptive_takeup_interval(self):
        result = await self.run_capture(
            "super8-takeup", frames=3, film_format=FILM_FORMAT_SUPER8,
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["transport"].takeup_begin_intervals, [12])
        self.assertEqual(result["transport"].takeup_ended, 1)


if __name__ == "__main__":
    unittest.main()
