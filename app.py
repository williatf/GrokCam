import asyncio
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import picamera2
import cv2
import numpy as np
import io
import time
import base64
import json
from control import tcControl
from registration import RegistrationTracker
from sprocket import SprocketDetector
from calibration_service import CalibrationService
from fast_sprocket import FastSprocketDetector
from transport_calibration import (
    AdaptiveTransportController,
    merge_calibration_settings,
    resolve_nominal_steps,
)
import socket
import os
import re
from datetime import datetime
from collections import deque

RAW_CAPTURE_MODE = 'raw_dng_v1'
RAW_SENSOR_SIZE = (2028, 1520)
RAW_PREVIEW_SIZE = (760, 570)
RAW_SAFE_DEFAULT_EXPOSURE_TIME = 1114

async def troubleshoot_sprocket_detection(camera, websocket, tc, detector,
                                          step_size=None, delay=0.05):
    """
    Bi-directional sprocket troubleshooting loop.
    The client can send:
      - {"event": "next_step"} → move + capture + send debug frame
            - {"event": "prev_step"} → move backward + capture + send debug frame
      - {"event": "stop_troubleshoot"} → exit loop
    """
    print("[TROUBLE] Entering interactive sprocket troubleshooting mode")

    if step_size is None:
        step_size = max(1, int(round(STEPS_PER_PITCH * 0.25)))

    active = True
    frame_counter = 0

    # ensure lighting and camera are on
    tc.light_on()
    camera.start()
    await asyncio.sleep(0.5)

    while active:
        msg = await websocket.recv()
        data = json.loads(msg)
        evt = data.get("event")

        if evt == "next_step" or evt == "prev_step":
            frame_counter += 1
            moving_forward = evt == "next_step"
            direction_label = "forward" if moving_forward else "backward"
            print(f"[TROUBLE] Step {frame_counter}: moving {direction_label} {step_size} steps")
            if moving_forward:
                tc.steps_forward(step_size)
            else:
                tc.steps_back(step_size)
            await asyncio.sleep(delay)

            # --- capture & detect ---
            buffer = io.BytesIO()
            camera.capture_file(buffer, format="jpeg")
            frame = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            sprockets = detector.detect(frame, mode="profile")

            print(f"[TROUBLE] Detected {len(sprockets)} sprockets")

            # --- draw debug overlay ---
            dbg = frame.copy()
            roi_x1, roi_y1, roi_x2, roi_y2 = detector.roi_bounds(frame.shape)
            cv2.rectangle(dbg, (roi_x1, roi_y1), (roi_x2 - 1, roi_y2 - 1), (0, 255, 255), 2)
            for (cx, cy, w, h, area) in sprockets:
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(dbg, f"cy={cy:.1f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            dbg = cv2.flip(dbg, 0)

            debug_scale = float(data.get("debug_scale", 1.0))
            if debug_scale != 1.0:
                dbg_w = max(1, int(dbg.shape[1] * debug_scale))
                dbg_h = max(1, int(dbg.shape[0] * debug_scale))
                dbg = cv2.resize(dbg, (dbg_w, dbg_h), interpolation=cv2.INTER_LINEAR)
            print(f"[TROUBLE] Frame {frame_counter} debug size: {dbg.shape[1]}x{dbg.shape[0]}")

            # --- send to client ---
            # ok, jpg = cv2.imencode(".jpg", dbg)
            ok, jpg = cv2.imencode('.jpg', dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            encoded_bytes = jpg.tobytes() if ok else b""
            print(f"[TROUBLE] Encoded debug length: {len(encoded_bytes)} bytes")
            encoded_shape = tuple(jpg.shape) if ok and hasattr(jpg, "shape") else "unknown"
            print(f"[TROUBLE] Encoded debug shape: {encoded_shape}")
            if ok:
                header = json.dumps({
                    "event": "troubleshoot_frame",
                    "frame": frame_counter,
                    "sprocket_count": len(sprockets),
                    "direction": direction_label
                })
                await websocket.send(header)
                await websocket.send(jpg.tobytes())

        elif evt == "stop_troubleshoot":
            print("[TROUBLE] Stopping troubleshooting mode")
            active = False
            await websocket.send(json.dumps({
                "event": "troubleshoot_complete",
                "message": "Stopped troubleshooting mode"
            }))
        else:
            print(f"[TROUBLE] Ignored unexpected event: {evt}")

    tc.clean_up()
    camera.stop()
    print("[TROUBLE] Troubleshooting mode exited cleanly")

def draw_sprockets_debug(frame, sprockets, registration_y=None, crop_rect=None, crop_clamped=False):
    """
    Draw bounding boxes, centers, and labels for detected sprockets.
    Always returns a valid flipped debug frame.
    """
    debug_frame = frame.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = detector.roi_bounds(frame.shape)
    roi_left = roi_x1 if roi_x1 > 0 else 1
    roi_right = roi_x2 - 1 if roi_x2 < frame.shape[1] else frame.shape[1] - 2
    cv2.line(debug_frame, (roi_left, roi_y1), (roi_left, roi_y2 - 1), (0, 255, 255), 2)
    cv2.line(debug_frame, (roi_right, roi_y1), (roi_right, roi_y2 - 1), (0, 255, 255), 2)
    cv2.rectangle(debug_frame, (max(0, roi_x1), roi_y1), (max(0, roi_x2 - 1), roi_y2 - 1), (0, 255, 255), 1)

    if crop_rect is None and registration_y is not None:
        try:
            crop_rect = get_relative_crop_rect(frame, registration_y)
        except Exception:
            crop_rect = None

    if crop_rect is not None:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_rect
        crop_color = (0, 165, 255) if crop_clamped else (255, 255, 0)
        cv2.rectangle(debug_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), crop_color, 2)

    # draw sprocket boxes if any
    if sprockets:
        for (cx, cy, w, h, area) in sprockets:
            # Draw rectangle
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw center
            cv2.circle(debug_frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            print(f"[Crop-Debug] cy is {int(cy)}")

            # Label coordinates
            cv2.putText(debug_frame, f"cy={cy:.1f}",
                        (int(cx) + 10, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
    else:
        print("[Crop-Debug] No sprockets detected for debug draw.")

    # always define flipped even if sprockets == []
    flipped = cv2.flip(debug_frame, 0)
    return flipped

def crop_film_frame(frame, anchor, pitch_px=None):
    """
    Crop relative to sprocket anchor:
    - Full width
    - Height = 120% of sprocket pitch
    - Anchor sits 10% from the top
    """
    if pitch_px is None:
        pitch_px = SPROCKET_PITCH_PX

    if anchor is None or pitch_px is None:
        return frame

    H, W = frame.shape[:2]
    cx, cy = int(anchor[0]), int(anchor[1])
    print(f"[CROP] cy is {cy}")

    crop_h = int(pitch_px * 1.2)
    offset = int(0.1 * crop_h)

    y1 = max(0, cy - offset)
    y2 = min(H, y1 + crop_h)

    # If bottom is clipped, shift up
    if y2 - y1 < crop_h and y1 > 0:
        y1 = max(0, y2 - crop_h)
        print("[APP] Cropping has moved up since the bottom is clipped.")

    cropped = frame[y1:y2, 0:W]

    # translate cy into cropped coordinates
    cy_local = cy - y1
    cx_local = cx  # x doesn’t shift because we keep full width

    # draw a marker on cy
    cv2.circle(cropped, (cx_local, cy_local), 8, (0, 0, 255), -1)
    cv2.line(cropped, (0, cy_local), (W, cy_local), (0, 0, 255), 2)

    # 🔄 Rotate 180° (flip vertically + horizontally)
    flipped = cv2.flip(cropped, 0)

    return flipped


def get_raw_registration(frame_bgr, sprockets):
    registration = detector.choose_registration(
        sprockets,
        frame_bgr.shape,
        expected_pitch=SPROCKET_PITCH_PX,
    )
    mode = registration.get('mode', 'none')
    actual_y = registration.get('actual_y')
    return {
        'registration_y': float(actual_y) if actual_y is not None else None,
        'mode': mode if mode in ('pair', 'single') else 'none',
        'target_y': registration.get('target_y'),
        'error_px': registration.get('error_px'),
    }


def get_registration_y(frame_bgr, sprockets):
    return get_raw_registration(frame_bgr, sprockets).get('registration_y')


def get_capture_registration(frame_bgr, sprockets):
    return get_raw_registration(frame_bgr, sprockets)


async def reacquire_pair_registration(camera, tc, detector, target_y, step_size=10, max_steps=300, reacquire_tolerance_px=80):
    total_steps = 0

    while total_steps <= max_steps:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return {
                'valid': False,
                'reason': 'failed_to_decode_reacquire_frame',
                'steps': total_steps,
            }

        sprockets = detector.detect(frame_bgr, mode='profile') or []
        classified = detector.classify_sprockets(sprockets, frame_bgr.shape)
        full_sprockets = [
            item['sprocket'] for item in classified
            if item.get('status') == 'full'
        ]
        full_count = len(full_sprockets)
        partial_count = sum(1 for item in classified if item.get('status') == 'partial')
        registration_choice = detector.choose_registration(
            full_sprockets,
            frame_bgr.shape,
            expected_pitch=detector.expected_pitch,
        )
        registration = {
            'mode': registration_choice.get('mode', 'none'),
            'registration_y': registration_choice.get('actual_y'),
        }

        if registration.get('mode') == 'pair' and registration.get('registration_y') is not None:
            error_px = float(target_y) - float(registration.get('registration_y'))
            print(
                f"[APP] Reacquire candidate: steps={total_steps}, "
                f"registration_y={registration.get('registration_y'):.1f}, error={error_px:+.1f}"
            )
            if abs(error_px) > float(reacquire_tolerance_px):
                if total_steps >= max_steps:
                    break
                tc.steps_forward(step_size)
                total_steps += step_size
                await asyncio.sleep(0.05)
                continue
            return {
                'valid': True,
                'registration_y': registration.get('registration_y'),
                'mode': 'pair',
                'steps': total_steps,
                'sprocket_count': len(sprockets),
                'full_count': full_count,
                'partial_count': partial_count,
            }

        if total_steps >= max_steps:
            break

        tc.steps_forward(step_size)
        total_steps += step_size
        await asyncio.sleep(0.05)

    return {
        'valid': False,
        'reason': 'pair_registration_not_found_within_tolerance',
        'steps': total_steps,
    }


def get_project_metadata_path(project_path=None):
    target_path = project_path or active_project_path
    return os.path.join(target_path, 'metadata.json') if target_path else None


def load_project_metadata(project_path=None):
    metadata_path = get_project_metadata_path(project_path)
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        print(f"[APP] Project metadata read failed: {exc}")
        return {}


def save_project_metadata(payload, project_path=None):
    metadata_path = get_project_metadata_path(project_path)
    if not metadata_path:
        raise RuntimeError('No active project selected')
    temporary_path = f"{metadata_path}.partial"
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    os.replace(temporary_path, metadata_path)


def get_effective_crop_settings(project_path=None):
    project_metadata = load_project_metadata(project_path)
    project_crop = project_metadata.get('crop')
    if isinstance(project_crop, dict):
        return project_crop
    global_crop = settings.get('crop')
    return global_crop if isinstance(global_crop, dict) else None


def get_relative_crop_rect(frame_bgr, registration_y, return_metadata=False):
    frame_h, frame_w = frame_bgr.shape[:2]
    crop_settings = get_effective_crop_settings()
    crop_meta = {
        'crop_y1': 0,
        'crop_y2': int(frame_h),
        'crop_clamped': False,
    }

    if not isinstance(crop_settings, dict) or registration_y is None:
        rect = (0, 0, frame_w, frame_h)
        if return_metadata:
            return rect, crop_meta
        return rect

    try:
        x1 = int(crop_settings['x1'])
        x2 = int(crop_settings['x2'])
        y_offset = float(crop_settings['y_offset'])
        height = int(crop_settings['height'])
    except (KeyError, TypeError, ValueError):
        rect = (0, 0, frame_w, frame_h)
        if return_metadata:
            return rect, crop_meta
        return rect

    if height <= 0:
        rect = (0, 0, frame_w, frame_h)
        if return_metadata:
            return rect, crop_meta
        return rect

    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(x1 + 1, min(frame_w, x2))

    y1 = int(round(float(registration_y) + y_offset))
    y2 = y1 + height

    if y1 < 0:
        crop_meta['crop_clamped'] = True
        y2 = min(frame_h, y2 - y1)
        y1 = 0
    if y2 > frame_h:
        crop_meta['crop_clamped'] = True
        y1 = max(0, y1 - (y2 - frame_h))
        y2 = frame_h

    if y2 <= y1:
        crop_meta['crop_clamped'] = True
        rect = (0, 0, frame_w, frame_h)
        if return_metadata:
            return rect, crop_meta
        return rect

    rect = (int(x1), int(y1), int(x2), int(y2))
    crop_meta['crop_y1'] = int(y1)
    crop_meta['crop_y2'] = int(y2)
    if return_metadata:
        return rect, crop_meta
    return rect


def crop_frame_relative_to_registration(frame_bgr, registration_y):
    x1, y1, x2, y2 = get_relative_crop_rect(frame_bgr, registration_y)
    return frame_bgr[y1:y2, x1:x2]


def get_scaled_relative_crop_rect(frame_bgr, registration_y, source_size=None):
    """Apply the calibrated full-resolution crop to a smaller preview stream."""
    frame_h, frame_w = frame_bgr.shape[:2]
    if source_size is None:
        source_size = CALIBRATION_RES
    source_w, source_h = source_size
    crop_settings = get_effective_crop_settings()
    if not isinstance(crop_settings, dict) or registration_y is None:
        return (0, 0, frame_w, frame_h), {'crop_clamped': False}

    scale_x = frame_w / float(source_w)
    scale_y = frame_h / float(source_h)
    try:
        x1 = int(round(float(crop_settings['x1']) * scale_x))
        x2 = int(round(float(crop_settings['x2']) * scale_x))
        y1 = int(round(float(registration_y) + float(crop_settings['y_offset']) * scale_y))
        height = max(1, int(round(float(crop_settings['height']) * scale_y)))
    except (KeyError, TypeError, ValueError):
        return (0, 0, frame_w, frame_h), {'crop_clamped': False}

    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = y1 + height
    clamped = False
    if y1 < 0:
        y2 -= y1
        y1 = 0
        clamped = True
    if y2 > frame_h:
        y1 = max(0, y1 - (y2 - frame_h))
        y2 = frame_h
        clamped = True
    return (x1, y1, x2, y2), {
        'crop_y1': y1,
        'crop_y2': y2,
        'crop_clamped': clamped,
    }


def get_scaled_saved_registration_target(frame_bgr, default_y=None):
    """Scale the project crop-calibration registration point to this stream."""
    frame_h, frame_w = frame_bgr.shape[:2]
    crop_settings = get_effective_crop_settings()
    if isinstance(crop_settings, dict):
        saved_y = crop_settings.get('registration_y')
        source_size = crop_settings.get('source_resolution')
        try:
            source_h = float(source_size[1])
            if saved_y is not None and source_h > 0:
                return float(saved_y) * frame_h / source_h
        except (TypeError, ValueError, IndexError):
            pass

    if calibrated_baseline_registration_y is not None and CALIBRATION_RES[1] > 0:
        return float(calibrated_baseline_registration_y) * frame_h / float(CALIBRATION_RES[1])
    return float(default_y) if default_y is not None else frame_h / 2.0


def save_crop_settings(rect, registration_y, frame_size):
    frame_w, frame_h = frame_size
    x1, y1, x2, y2 = rect
    crop_data = {
        'reference': 'registration_y',
        'x1': int(x1),
        'x2': int(x2),
        'y_offset': float(y1 - registration_y),
        'height': int(y2 - y1),
        'registration_y': float(registration_y),
        'source_resolution': [int(frame_w), int(frame_h)],
    }

    config_path = 'config.json'
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as handle:
            config_data = json.load(handle)

    config_data['crop'] = crop_data
    config_data['crop_updated_timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(config_path, 'w', encoding='utf-8') as handle:
        json.dump(config_data, handle, indent=2)
        handle.write('\n')

    if active_project_path:
        project_metadata = load_project_metadata(active_project_path)
        project_metadata['crop'] = dict(crop_data)
        project_metadata['crop_source'] = 'project_saved'
        project_metadata['crop_updated_timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_project_metadata(project_metadata, active_project_path)

    return crop_data

async def encode_frame_async(frame_cropped, frame_num):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, encode_frame, frame_cropped, frame_num)

def encode_frame(frame_cropped, frame_num):
    _, encoded = cv2.imencode('.jpg', frame_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    jpg_bytes = encoded.tobytes()
    header = json.dumps({
        'event': 'new_image',
        'frame': frame_num,
        'size': len(jpg_bytes)
    })

    return header, jpg_bytes


def compute_calibration_summary(samples):
    pitch_values = [
        float(sample['sprocket_pitch_px'])
        for sample in samples
        if sample.get('pitch_valid') and sample.get('sprocket_pitch_px') is not None
    ]
    area_values = []
    for sample in samples:
        for area in sample.get('full_sprocket_areas', []):
            area_values.append(float(area))

    valid_samples = sum(
        1 for sample in samples
        if sample.get('pitch_valid') and sample.get('sprocket_pitch_px') is not None
    )

    summary = {
        'pitch_mean': None,
        'pitch_min': None,
        'pitch_max': None,
        'pitch_std': None,
        'area_mean': None,
        'area_min': None,
        'area_max': None,
        'area_std': None,
        'valid_samples': valid_samples,
        'total_samples': len(samples),
    }

    if pitch_values:
        pitch_array = np.array(pitch_values, dtype=float)
        summary.update({
            'pitch_mean': float(np.mean(pitch_array)),
            'pitch_min': float(np.min(pitch_array)),
            'pitch_max': float(np.max(pitch_array)),
            'pitch_std': float(np.std(pitch_array)),
        })

    if area_values:
        area_array = np.array(area_values, dtype=float)
        summary.update({
            'area_mean': float(np.mean(area_array)),
            'area_min': float(np.min(area_array)),
            'area_max': float(np.max(area_array)),
            'area_std': float(np.std(area_array)),
        })

    return summary


def filter_robust_values(values, max_mad_scale=3.5):
    if not values:
        return []

    array = np.array(values, dtype=float)
    median = float(np.median(array))
    deviations = np.abs(array - median)
    mad = float(np.median(deviations))
    if mad <= 0:
        return array.tolist()

    threshold = mad * max_mad_scale
    filtered = array[deviations <= threshold]
    return filtered.tolist() if filtered.size else array.tolist()


def build_proposed_calibration(samples, exposure_result, motor_calibration=None):
    valid_pitch_values = [
        float(sample['sprocket_pitch_px'])
        for sample in samples
        if sample.get('pitch_valid') and sample.get('sprocket_pitch_px') is not None
    ]
    valid_pitch_values = filter_robust_values(valid_pitch_values)

    all_full_areas = []
    for sample in samples:
        all_full_areas.extend(float(area) for area in sample.get('full_sprocket_areas', []))
    trusted_areas = filter_robust_values(all_full_areas)

    if len(valid_pitch_values) < 5:
        return None, False, 'need_at_least_5_valid_pitch_samples'
    if not trusted_areas:
        return None, False, 'need_trusted_full_sprocket_area_samples'

    trusted_pitch = float(np.median(np.array(valid_pitch_values, dtype=float)))
    trusted_area = float(np.median(np.array(trusted_areas, dtype=float)))
    area_mad = float(np.median(np.abs(np.array(trusted_areas, dtype=float) - trusted_area))) if trusted_areas else 0.0
    if trusted_area <= 0:
        return None, False, 'invalid_trusted_area'

    area_spread_frac = 0.05
    if area_mad > 0:
        area_spread_frac = max(0.05, min(0.12, (2.5 * area_mad) / trusted_area))

    if motor_calibration and motor_calibration.get('motor_updated'):
        steps_per_pitch_value = motor_calibration.get('motor_steps_per_pitch')
    else:
        steps_per_pitch_value = settings.get('steps_per_pitch', STEPS_PER_PITCH)

    if steps_per_pitch_value is None:
        return None, False, 'missing_steps_per_pitch'

    steps_per_pitch_value = float(steps_per_pitch_value)
    proposed_calibration = {
        'calibration_version': 2,
        'calibration_resolution': list(CALIBRATION_RES),
        'exposure_time': int(exposure_result['exposure_time']),
        'gain': 1.0,
        'sprocket_pitch_px': trusted_pitch,
        'steps_per_pitch': int(round(steps_per_pitch_value)),
        'steps_per_px': steps_per_pitch_value / trusted_pitch,
        'sprocket_area_min': int(round(trusted_area * (1.0 - area_spread_frac))),
        'sprocket_area_max': int(round(trusted_area * (1.0 + area_spread_frac))),
    }
    return proposed_calibration, True, None

# --- Load calibration + config ---
def load_settings():
    calib = {}
    config = {}
    if os.path.exists("calibration.json"):
        with open("calibration.json", "r") as f:
            calib = json.load(f)
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    # config overrides calib
    return merge_calibration_settings(calib, config)


def refresh_runtime_settings():
    global settings, pitch_px, steps_per_pitch, calib_res, exposure_time, gain
    global steps_per_px, SPROCKET_PITCH_PX, STEPS_PER_PITCH
    global CALIBRATION_RES, EXPOSURE_TIME, GAIN

    previous_resolution = CALIBRATION_RES if 'CALIBRATION_RES' in globals() else None

    settings = load_settings()
    pitch_px = settings.get("sprocket_pitch_px", 835)
    steps_per_pitch = settings.get("steps_per_pitch", 280)
    calib_res = settings.get("calibration_resolution", [2028,1520])
    exposure_time = settings.get("exposure_time", 612)
    gain = settings.get("gain", 1.0)

    if not pitch_px or not steps_per_pitch or not calib_res or not exposure_time or gain is None:
        raise RuntimeError("Calibration data missing. Please run calibrate_16mm.py first.")

    steps_per_px = settings.get("steps_per_px", steps_per_pitch / pitch_px)

    SPROCKET_PITCH_PX = pitch_px
    STEPS_PER_PITCH = steps_per_pitch
    CALIBRATION_RES = tuple(calib_res)
    EXPOSURE_TIME = exposure_time
    GAIN = gain

    detector.min_area = settings.get("sprocket_area_min", detector.min_area)
    detector.max_area = settings.get("sprocket_area_max", detector.max_area)
    detector.expected_pitch = int(SPROCKET_PITCH_PX)
    calibrator.settings = settings

    # RAW registration works on a smaller preview stream, but its geometry is
    # derived from this same full-resolution calibration. Keep both RAW
    # detectors synchronized when calibration is saved without a restart.
    if 'raw_fast_detector' in globals():
        raw_fast_detector.reference_size = tuple(CALIBRATION_RES)
        raw_fast_detector.expected_pitch = float(SPROCKET_PITCH_PX)
        raw_fast_detector.reset()
    if 'raw_fallback_detector' in globals():
        preview_scale = RAW_PREVIEW_SIZE[1] / float(CALIBRATION_RES[1])
        raw_fallback_detector.expected_pitch = int(round(SPROCKET_PITCH_PX * preview_scale))
        raw_fallback_detector.dynamic_roi_bounds = None
        raw_fallback_detector.dynamic_roi_misses = 0
        raw_fallback_detector.dynamic_roi_candidates = []

    print("[APP] Runtime calibration settings refreshed")
    if previous_resolution is not None and tuple(previous_resolution) != CALIBRATION_RES:
        print("[APP] Calibration resolution changed; service restart recommended to reconfigure camera")
    if 'registration_tracker' in globals():
        reset_registration_tracking("runtime_settings_refresh")

settings = load_settings()
print("Loaded settings:")
print(json.dumps(settings, indent=2))

# --- Calibration constants (loaded from calibration.json) ---
pitch_px = settings.get("sprocket_pitch_px", 835)
steps_per_pitch = settings.get("steps_per_pitch", 280)
calib_res = settings.get("calibration_resolution", [2028,1520])
exposure_time = settings.get("exposure_time", 612)
gain = settings.get("gain", 1.0)

if not pitch_px or not steps_per_pitch or not calib_res or not exposure_time or gain is None:
    raise RuntimeError("Calibration data missing. Please run calibrate_16mm.py first.")

steps_per_px = steps_per_pitch / pitch_px

SPROCKET_PITCH_PX = pitch_px
STEPS_PER_PITCH = steps_per_pitch
CALIBRATION_RES = tuple(calib_res)
EXPOSURE_TIME = exposure_time
GAIN = gain

CAMERA_EXPOSURE_MIN = 100
CAMERA_EXPOSURE_MAX = 50000
CAMERA_GAIN_MIN = 1.0
CAMERA_GAIN_MAX = 16.0

current_camera_settings = {
    'ExposureTime': int(EXPOSURE_TIME),
    'AnalogueGain': float(GAIN),
    'ColourGains': None,
    'AeEnable': False,
    'AwbEnable': False,
    'source': 'default',
    'saved': False,
    'project_path': None,
    'timestamp': None,
}

active_project_name = None
active_project_safe_name = None
active_project_path = None

# --- Initialize transport ---
print("Starting WebSocket server")
tc = tcControl()
print("tcControl initialized")

# --- Initialize camera ---
camera = picamera2.Picamera2()
print("Initializing camera")

# Match calibration resolution
config_main = camera.create_still_configuration(main={"size": CALIBRATION_RES})
camera.configure(config_main)
camera.options['quality'] = 90


def configure_legacy_camera():
    camera.configure(camera.create_still_configuration(main={"size": CALIBRATION_RES}))
    camera.options['quality'] = 90


def configure_raw_camera():
    camera.configure(camera.create_still_configuration(
        main={"size": RAW_PREVIEW_SIZE, "format": "BGR888"},
        raw={"size": RAW_SENSOR_SIZE, "format": "SRGGB12"},
        buffer_count=2,
    ))


def get_camera_settings_path(project_path=None):
    target_project_path = project_path or active_project_path
    if not target_project_path:
        return None
    return os.path.join(target_project_path, 'camera_settings.json')


def clamp_camera_settings(exposure_time=None, analogue_gain=None):
    requested_exposure = EXPOSURE_TIME if exposure_time is None else int(round(float(exposure_time)))
    requested_gain = GAIN if analogue_gain is None else float(analogue_gain)
    clamped_exposure = max(CAMERA_EXPOSURE_MIN, min(CAMERA_EXPOSURE_MAX, requested_exposure))
    clamped_gain = max(CAMERA_GAIN_MIN, min(CAMERA_GAIN_MAX, requested_gain))
    return {
        'ExposureTime': int(clamped_exposure),
        'AnalogueGain': float(clamped_gain),
        'AeEnable': False,
        'AwbEnable': False,
        'exposure_clamped': clamped_exposure != requested_exposure,
        'gain_clamped': clamped_gain != requested_gain,
    }


def camera_settings_response_payload(settings_state=None):
    state = settings_state or current_camera_settings
    payload = {
        'event': 'camera_settings',
        'type': 'camera_settings',
        'exposure_time': int(state.get('ExposureTime', EXPOSURE_TIME)),
        'analogue_gain': float(state.get('AnalogueGain', GAIN)),
        'ae_enable': bool(state.get('AeEnable', False)),
        'awb_enable': bool(state.get('AwbEnable', False)),
        'saved': bool(state.get('saved', False)),
        'source': state.get('source', 'default'),
    }
    colour_gains = state.get('ColourGains')
    if colour_gains is not None:
        payload['colour_gains'] = [float(colour_gains[0]), float(colour_gains[1])]
    return payload


def camera_awb_response_payload(settings_state=None):
    state = settings_state or current_camera_settings
    colour_gains = state.get('ColourGains')
    return {
        'event': 'camera_awb',
        'type': 'camera_awb',
        'colour_gains': [float(colour_gains[0]), float(colour_gains[1])] if colour_gains is not None else None,
        'awb_enable': bool(state.get('AwbEnable', False)),
        'ae_enable': bool(state.get('AeEnable', False)),
        'source': state.get('source', 'default'),
    }


def normalize_colour_gains(colour_gains):
    if colour_gains is None:
        return None
    if not isinstance(colour_gains, (list, tuple)) or len(colour_gains) != 2:
        raise ValueError('ColourGains must be a two-element sequence')
    return (float(colour_gains[0]), float(colour_gains[1]))


def set_current_camera_settings_state(exposure_time, analogue_gain, source='manual', saved=False, project_path=None, timestamp=None, colour_gains=None):
    global current_camera_settings
    current_camera_settings = {
        'ExposureTime': int(exposure_time),
        'AnalogueGain': float(analogue_gain),
        'ColourGains': normalize_colour_gains(colour_gains),
        'AeEnable': False,
        'AwbEnable': False,
        'source': source,
        'saved': bool(saved),
        'project_path': project_path,
        'timestamp': timestamp,
    }
    return current_camera_settings.copy()


def apply_manual_camera_settings(exposure_time=None, analogue_gain=None, source='manual', saved=False, project_path=None, timestamp=None, colour_gains=None):
    clamped = clamp_camera_settings(exposure_time=exposure_time, analogue_gain=analogue_gain)
    if clamped['exposure_clamped'] or clamped['gain_clamped']:
        print(
            f"[APP] Camera settings clamped: exposure={clamped['ExposureTime']} gain={clamped['AnalogueGain']:.3f}"
        )

    controls = {
        'ExposureTime': int(clamped['ExposureTime']),
        'AnalogueGain': float(clamped['AnalogueGain']),
        'AeEnable': False,
        'AwbEnable': False,
    }
    normalized_colour_gains = normalize_colour_gains(colour_gains)
    if normalized_colour_gains is not None:
        controls['ColourGains'] = normalized_colour_gains

    camera.set_controls(controls)
    applied = set_current_camera_settings_state(
        clamped['ExposureTime'],
        clamped['AnalogueGain'],
        source=source,
        saved=saved,
        project_path=project_path,
        timestamp=timestamp,
        colour_gains=normalized_colour_gains,
    )
    print(
        f"[APP] Camera settings applied: exposure={applied['ExposureTime']} "
        f"gain={applied['AnalogueGain']:.3f} source={source} saved={saved}"
    )
    return applied


def load_project_camera_settings(project_path=None, apply=False):
    settings_path = get_camera_settings_path(project_path)
    if not settings_path or not os.path.exists(settings_path):
        return None

    with open(settings_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    clamped = clamp_camera_settings(
        exposure_time=payload.get('ExposureTime', EXPOSURE_TIME),
        analogue_gain=payload.get('AnalogueGain', GAIN),
    )
    colour_gains = payload.get('ColourGains')
    print(f"[APP] Loaded camera settings from {settings_path}")
    if apply:
        return apply_manual_camera_settings(
            clamped['ExposureTime'],
            clamped['AnalogueGain'],
            source=payload.get('source', 'manual'),
            saved=True,
            project_path=project_path or active_project_path,
            timestamp=payload.get('timestamp'),
            colour_gains=colour_gains,
        )

    return set_current_camera_settings_state(
        clamped['ExposureTime'],
        clamped['AnalogueGain'],
        source=payload.get('source', 'manual'),
        saved=True,
        project_path=project_path or active_project_path,
        timestamp=payload.get('timestamp'),
        colour_gains=colour_gains,
    )


def has_project_manual_camera_settings(project_path=None):
    target_project_path = project_path or active_project_path
    if not target_project_path:
        return False
    if get_camera_settings_path(target_project_path) and os.path.exists(get_camera_settings_path(target_project_path)):
        return True
    return (
        current_camera_settings.get('project_path') == target_project_path
        and current_camera_settings.get('source') == 'manual'
    )


def initialize_project_camera_settings(project_path=None, apply=False):
    target_project_path = project_path or active_project_path
    loaded = load_project_camera_settings(target_project_path, apply=apply)
    if loaded is not None:
        return loaded

    return set_current_camera_settings_state(
        EXPOSURE_TIME,
        GAIN,
        source='default',
        saved=False,
        project_path=target_project_path,
        colour_gains=current_camera_settings.get('ColourGains'),
    )


def apply_project_capture_camera_settings(project_path=None, prefer_saved=True):
    target_project_path = project_path or active_project_path
    if prefer_saved:
        loaded = load_project_camera_settings(target_project_path, apply=True)
        if loaded is not None:
            return loaded

    if current_camera_settings.get('project_path') == target_project_path:
        return apply_manual_camera_settings(
            current_camera_settings.get('ExposureTime', EXPOSURE_TIME),
            current_camera_settings.get('AnalogueGain', GAIN),
            source=current_camera_settings.get('source', 'default'),
            saved=current_camera_settings.get('saved', False),
            project_path=target_project_path,
            timestamp=current_camera_settings.get('timestamp'),
            colour_gains=current_camera_settings.get('ColourGains'),
        )

    return apply_manual_camera_settings(
        EXPOSURE_TIME,
        GAIN,
        source='default',
        saved=False,
        project_path=target_project_path,
        colour_gains=current_camera_settings.get('ColourGains'),
    )

def apply_capture_camera_controls():
    print("[APP] Switching camera to capture controls")
    return apply_project_capture_camera_settings(active_project_path, prefer_saved=True)


def apply_raw_capture_camera_controls():
    """Use saved project controls, with a conservative RAW-only fallback."""
    if has_project_manual_camera_settings(active_project_path):
        return apply_project_capture_camera_settings(active_project_path, prefer_saved=True)

    safe_exposure = int(settings.get(
        'raw_exposure_time',
        min(int(EXPOSURE_TIME), RAW_SAFE_DEFAULT_EXPOSURE_TIME),
    ))
    return apply_manual_camera_settings(
        safe_exposure,
        1.0,
        source='raw_safe_default',
        saved=False,
        project_path=active_project_path,
        colour_gains=current_camera_settings.get('ColourGains'),
    )


def apply_focus_camera_controls():
    print("[APP] Switching camera to focus controls")
    camera.set_controls({
        "AeEnable": True,
        "AwbEnable": False,
    })


def _metadata_value(metadata, *keys):
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


async def converge_camera_metadata(settle_frames=12, settle_delay=0.08):
    metadata = {}
    for _ in range(max(1, int(settle_frames))):
        await asyncio.sleep(float(settle_delay))
        metadata = camera.capture_metadata() or {}
    return metadata


async def run_one_shot_auto_exposure(preview_active=False, settle_frames=12, settle_delay=0.08):
    started_here = False
    started_at = time.time()
    try:
        if not preview_active:
            tc.light_on()
            camera.start()
            started_here = True
            await asyncio.sleep(0.3)

        print("[APP] Auto exposure start")
        camera.set_controls({
            'AeEnable': True,
            'AwbEnable': False,
        })
        metadata = await converge_camera_metadata(settle_frames=settle_frames, settle_delay=settle_delay)

        measured_exposure = _metadata_value(metadata, 'ExposureTime', 'SensorExposureTime')
        measured_gain = _metadata_value(metadata, 'AnalogueGain')
        if measured_exposure is None:
            measured_exposure = current_camera_settings.get('ExposureTime', EXPOSURE_TIME)
        if measured_gain is None:
            measured_gain = current_camera_settings.get('AnalogueGain', GAIN)

        applied = apply_manual_camera_settings(
            measured_exposure,
            measured_gain,
            source='auto_exposure',
            saved=False,
            project_path=active_project_path,
            timestamp=datetime.now().isoformat(),
            colour_gains=current_camera_settings.get('ColourGains'),
        )
        elapsed = time.time() - started_at
        print(
            f"[APP] Auto exposure locked: exposure={applied['ExposureTime']} "
            f"gain={applied['AnalogueGain']:.3f} elapsed={elapsed:.2f}s"
        )
        return applied
    finally:
        if started_here:
            tc.light_off()
            camera.stop()


async def run_one_shot_auto_awb(preview_active=False, settle_frames=12, settle_delay=0.08):
    started_here = False
    started_at = time.time()
    try:
        if not preview_active:
            tc.light_on()
            camera.start()
            started_here = True
            await asyncio.sleep(0.3)

        print("[APP] Auto white balance start")
        camera.set_controls({
            'AeEnable': False,
            'AwbEnable': True,
        })
        metadata = await converge_camera_metadata(settle_frames=settle_frames, settle_delay=settle_delay)

        measured_colour_gains = _metadata_value(metadata, 'ColourGains', 'ColorGains')
        if measured_colour_gains is None:
            measured_colour_gains = current_camera_settings.get('ColourGains')
        if measured_colour_gains is None:
            raise RuntimeError('Unable to read converged ColourGains from camera metadata')

        applied = apply_manual_camera_settings(
            current_camera_settings.get('ExposureTime', EXPOSURE_TIME),
            current_camera_settings.get('AnalogueGain', GAIN),
            source='auto_awb',
            saved=False,
            project_path=active_project_path,
            timestamp=datetime.now().isoformat(),
            colour_gains=measured_colour_gains,
        )
        elapsed = time.time() - started_at
        print(
            f"[APP] Auto white balance locked: colour_gains={applied.get('ColourGains')} elapsed={elapsed:.2f}s"
        )
        return applied
    finally:
        if started_here:
            tc.light_off()
            camera.stop()


async def tune_calibration_exposure(camera, detector, target=225, percentile=99.5, max_iters=8):
    exposure = int(EXPOSURE_TIME)
    last_percentile = 0.0
    last_max = 0.0
    last_clip_pct = 0.0
    iterations = 0

    for iteration in range(1, max_iters + 1):
        camera.set_controls({
            "ExposureTime": int(exposure),
            "AnalogueGain": 1.0,
            "AeEnable": False,
            "AwbEnable": False,
        })
        await asyncio.sleep(0.15)

        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError('Failed to decode calibration exposure frame')

        frame_h, frame_w = frame_bgr.shape[:2]
        strip_w = max(1, int(frame_w * detector.auto_roi))
        if detector.side == 'left':
            roi = frame_bgr[:, :strip_w]
        else:
            roi = frame_bgr[:, -strip_w:]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        last_percentile = float(np.percentile(gray_roi, percentile))
        last_max = float(np.max(gray_roi))
        last_clip_pct = float((np.count_nonzero(gray_roi >= 250) / gray_roi.size) * 100.0)
        iterations = iteration

        print(
            f"[APP] Calibration exposure iter {iteration}: exposure={int(exposure)} "
            f"p{percentile}={last_percentile:.1f} max={last_max:.1f} clip_pct={last_clip_pct:.3f}"
        )

        if abs(last_percentile - target) <= 3 and last_clip_pct < 0.1:
            break

        new_exposure = exposure * target / max(last_percentile, 1.0)
        exposure = max(100, min(20000, int(new_exposure)))

    camera.set_controls({
        "ExposureTime": int(exposure),
        "AnalogueGain": 1.0,
        "AeEnable": False,
        "AwbEnable": False,
    })
    await asyncio.sleep(0.15)

    print(
        f"[APP] Calibration exposure locked: exposure={int(exposure)} gain=1.0 "
        f"p{percentile}={last_percentile:.1f} max={last_max:.1f} clip_pct={last_clip_pct:.3f}"
    )

    return {
        "exposure_time": int(exposure),
        "gain": 1.0,
        "roi_percentile": last_percentile,
        "roi_max": last_max,
        "clip_pct": last_clip_pct,
        "iterations": iterations,
    }


async def prepare_calibration_camera_settings(detector):
    if has_project_manual_camera_settings(active_project_path):
        applied = apply_project_capture_camera_settings(active_project_path, prefer_saved=True)
        return {
            'exposure_time': int(applied['ExposureTime']),
            'gain': float(applied['AnalogueGain']),
            'roi_percentile': None,
            'roi_max': None,
            'clip_pct': None,
            'iterations': 0,
            'source': 'manual_project',
        }

    exposure_result = await tune_calibration_exposure(camera, detector)
    exposure_result['source'] = 'auto_calibration'
    return exposure_result


async def seek_two_full_sprockets(camera, tc, detector, step_size=10, max_steps=500, settle_delay=0.05):
    total_steps = 0

    while total_steps <= max_steps:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return {
                'valid': False,
                'reason': 'failed_to_decode_seek_frame',
                'steps': total_steps,
            }

        sprockets = detector.detect(frame_bgr, mode='profile') or []
        classified = detector.classify_sprockets(sprockets, frame_bgr.shape)
        full_sprocket_count = sum(1 for item in classified if item.get('status') == 'full')

        if full_sprocket_count == 2:
            return {
                'valid': True,
                'steps': total_steps,
                'full_sprocket_count': 2,
                'sprocket_count': len(sprockets),
            }

        if total_steps >= max_steps:
            break

        tc.steps_forward(step_size)
        total_steps += step_size
        await asyncio.sleep(settle_delay)

    return {
        'valid': False,
        'reason': 'did_not_find_two_full_sprockets',
        'steps': total_steps,
    }


async def measure_steps_per_pitch_live(camera, tc, detector, sprocket_pitch_px, step_chunk=20, max_steps=500):
    def choose_reference_y(frame_bgr, sprockets):
        # Motor calibration must follow one physical sprocket, not a pair midpoint.
        classified = detector.classify_sprockets(sprockets, frame_bgr.shape)
        full_sprockets = [item['sprocket'] for item in classified if item.get('status') == 'full']
        if full_sprockets:
            center_y = frame_bgr.shape[0] / 2.0
            anchor = min(full_sprockets, key=lambda sprocket: abs(sprocket[1] - center_y))
            return float(anchor[1])

        return None

    buffer = io.BytesIO()
    camera.capture_file(buffer, format='jpeg')
    frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return {'valid': False, 'reason': 'failed_to_decode_start_frame'}

    sprockets = detector.detect(frame_bgr, mode='profile') or []
    if not sprockets:
        return {'valid': False, 'reason': 'no_sprockets_in_start_frame'}

    start_y = choose_reference_y(frame_bgr, sprockets)
    if start_y is None:
        return {'valid': False, 'reason': 'no_stable_registration_reference'}

    total_steps = 0
    threshold = float(sprocket_pitch_px) * 0.85

    while total_steps < max_steps:
        tc.steps_forward(step_chunk)
        total_steps += step_chunk
        await asyncio.sleep(0.05)

        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        sprockets = detector.detect(frame_bgr, mode='profile') or []
        if not sprockets:
            continue

        current_y = choose_reference_y(frame_bgr, sprockets)
        if current_y is None:
            continue

        delta_y = abs(float(current_y) - float(start_y))
        if delta_y >= threshold:
            steps_per_px_value = total_steps / delta_y
            steps_per_pitch_value = steps_per_px_value * float(sprocket_pitch_px)
            return {
                'steps_per_pitch': int(round(steps_per_pitch_value)),
                'steps_per_px': float(steps_per_px_value),
                'total_steps': int(total_steps),
                'delta_y': float(delta_y),
                'valid': True,
            }

    return {
        'valid': False,
        'reason': 'did_not_reach_pitch_threshold',
        'total_steps': int(total_steps),
    }


async def advance_and_register_frame(camera, tc, detector,
                                     target_y=None,
                                     steps_per_pitch=None,
                                     steps_per_px=None,
                                     max_corrections=3,
                                     tolerance_px=4,
                                     max_correction_frac=0.15):
    if steps_per_pitch is None or steps_per_px is None:
        raise ValueError("Calibration values (steps_per_pitch, steps_per_px) required.")

    if target_y is None:
        frame_h = camera.capture_array("main").shape[0]
        target_y = frame_h / 2.0

    steps_per_pitch = int(round(float(steps_per_pitch)))
    steps_per_px = float(steps_per_px)
    search_step = 5
    correction_limit = max(1, int(round(steps_per_pitch * float(max_correction_frac))))

    print(
        f"[APP] Registration advance start: target_y={target_y:.1f}, "
        f"steps_per_pitch={steps_per_pitch}, tolerance_px={tolerance_px}, "
        f"max_corrections={max_corrections}"
    )

    tc.steps_forward(steps_per_pitch)
    await asyncio.sleep(0.05)

    for correction_index in range(max(0, int(max_corrections))):
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"[APP] Registration correction {correction_index + 1}: failed to decode frame")
            continue

        sprockets = detector.detect(frame_bgr, mode='profile') or []
        registration_y = get_registration_y(frame_bgr, sprockets)

        if registration_y is None:
            print(
                f"[APP] Registration correction {correction_index + 1}: "
                f"registration_y unavailable, searching forward {search_step} steps"
            )
            tc.steps_forward(search_step)
            await asyncio.sleep(0.05)
            continue

        error_px = float(target_y) - float(registration_y)
        print(
            f"[APP] Registration correction {correction_index + 1}: "
            f"target_y={target_y:.1f}, registration_y={registration_y:.1f}, error_px={error_px:+.1f}"
        )

        if abs(error_px) <= float(tolerance_px):
            return {
                'valid': True,
                'registration_y': float(registration_y),
                'target_y': float(target_y),
                'error_px': float(error_px),
                'steps_per_pitch': int(steps_per_pitch),
            }

        correction_steps = int(round(error_px * steps_per_px))
        correction_steps = max(-correction_limit, min(correction_steps, correction_limit))
        print(
            f"[APP] Registration correction {correction_index + 1}: correction_steps={correction_steps:+d}"
        )

        if correction_steps > 0:
            tc.steps_forward(correction_steps)
        elif correction_steps < 0:
            tc.steps_back(abs(correction_steps))
        else:
            print(f"[APP] Registration correction {correction_index + 1}: zero-step correction")

        await asyncio.sleep(0.05)

    buffer = io.BytesIO()
    camera.capture_file(buffer, format='jpeg')
    frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return {
            'valid': False,
            'reason': 'failed_to_decode_final_registration_frame',
        }

    sprockets = detector.detect(frame_bgr, mode='profile') or []
    registration_y = get_registration_y(frame_bgr, sprockets)
    if registration_y is None:
        return {
            'valid': False,
            'reason': 'registration_not_found_after_corrections',
        }

    error_px = float(target_y) - float(registration_y)
    print(
        f"[APP] Registration final: target_y={target_y:.1f}, "
        f"registration_y={registration_y:.1f}, error_px={error_px:+.1f}"
    )
    return {
        'valid': True,
        'registration_y': float(registration_y),
        'target_y': float(target_y),
        'error_px': float(error_px),
        'steps_per_pitch': int(steps_per_pitch),
    }


apply_capture_camera_controls()

detector = SprocketDetector(
    side="left", auto_roi=0.40,
    min_area=settings.get("sprocket_area_min",1500), 
    max_area=settings.get("sprocket_area_max",25000),
    ar_min=1.1, ar_max=2.0,
    solidity_min=0.65,
    blur=5, open_k=5, close_k=3,
    adaptive_block=41, adaptive_C=7,
    method="profile"
)

raw_preview_scale = RAW_PREVIEW_SIZE[1] / float(CALIBRATION_RES[1])
raw_area_scale = (
    RAW_PREVIEW_SIZE[0] / float(CALIBRATION_RES[0])
    * RAW_PREVIEW_SIZE[1] / float(CALIBRATION_RES[1])
)
raw_fast_detector = FastSprocketDetector(
    reference_size=CALIBRATION_RES,
    expected_pitch=SPROCKET_PITCH_PX,
)
raw_fallback_detector = SprocketDetector(
    side="left", auto_roi=0.40,
    min_area=max(500, int(round(12000 * raw_area_scale))),
    max_area=max(5000, int(round(150000 * raw_area_scale))),
    ar_min=1.1, ar_max=2.0,
    solidity_min=0.65,
    blur=5, open_k=5, close_k=3,
    adaptive_block=41, adaptive_C=7,
    method="profile",
)
raw_fallback_detector.expected_pitch = int(round(SPROCKET_PITCH_PX * raw_preview_scale))

registration_tracker = RegistrationTracker(
    expected_sprocket_pitch_px=SPROCKET_PITCH_PX,
    max_jump_px=40.0,
    smoothing_alpha=0.8,
)

latest_crop_preview_pair_midpoint = None
calibrated_baseline_registration_y = None


def reset_registration_tracking(reason=None, baseline_registration_y=None):
    seed_baseline = baseline_registration_y
    if seed_baseline is None:
        seed_baseline = calibrated_baseline_registration_y
    registration_tracker.reset(
        expected_sprocket_pitch_px=SPROCKET_PITCH_PX,
        baseline_registration_y=seed_baseline,
    )
    if reason:
        print(f"[APP] Registration tracker reset: {reason}")

calibrator = CalibrationService(camera, tc, detector, settings)

last_error = 0 # difference between actual and target for sprocket detection

async def advance_to_next_perforation(camera,
                                      websocket, 
                                      target_y=None,
                                      steps_per_pitch=None,
                                      steps_per_px=None,
                                      k_gain=0.4,
                                      min_step=5,
                                      max_step=None,
                                      smooth_alpha=0.6,
                                      new_sprocket_min_delta_frac=0.35,
                                      fine_history=5,
                                      initial_step=20,
                                      old_track_tol_frac=0.25,
                                      old_missing_required=2,
                                      min_new_samples=3,
                                      acceptance_tol_frac=0.1):
    """
    Advance film until a new sprocket appears at the top of the image,
    using feedback from detected sprocket position to self-correct
    overshoot or undershoot dynamically.
    Stabilized version with smoothing, min delta, and correction limits.
    """
    if steps_per_pitch is None or steps_per_px is None:
        raise ValueError("Calibration values (steps_per_pitch, steps_per_px) required.")

    if max_step is None:
        max_step = int(steps_per_pitch * 1.5)
    if target_y is None:
        frame_H = camera.capture_array("main").shape[0]
        target_y = int(frame_H * 0.35)

    pitch_px_est = steps_per_pitch / steps_per_px if steps_per_px else SPROCKET_PITCH_PX
    new_sprocket_min_delta = new_sprocket_min_delta_frac * pitch_px_est

    smoothed_old_cy = None
    old_reference = None
    new_sprocket_samples = deque(maxlen=fine_history)
    old_missing_frames = 0
    total_steps = 0
    step_small = max(min_step, int(initial_step))
    old_track_tol_px = max(5, int(SPROCKET_PITCH_PX * old_track_tol_frac))
    last_seen_cx = 0
    last_seen_cy = target_y
    acceptance_tol_px = max(10, int(SPROCKET_PITCH_PX * acceptance_tol_frac))

    print(f"[APP] Adaptive advance: target_y={target_y}, step_small={step_small}, max_step={max_step}")

    while True:
        # --- capture & detect ---
        buffer = io.BytesIO()
        camera.capture_file(buffer, format="jpeg")
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8),
                                 cv2.IMREAD_COLOR)
        sprockets = detector.detect(lores_bgr, mode="profile") or []

        if sprockets:
            sprockets.sort(key=lambda s: s[1])  # sort top-to-bottom
            top_cx, top_cy, *_ = sprockets[0]
            last_seen_cx, last_seen_cy = top_cx, top_cy

            if smoothed_old_cy is None:
                smoothed_old_cy = top_cy
                old_reference = top_cy
                print(f"[APP] Tracking initial sprocket at cy={smoothed_old_cy:.1f}")
            else:
                # Track previous sprocket if still visible
                old_candidate = min(sprockets, key=lambda s: abs(s[1] - smoothed_old_cy))
                if abs(old_candidate[1] - smoothed_old_cy) <= old_track_tol_px:
                    smoothed_old_cy = smooth_alpha * old_candidate[1] + (1 - smooth_alpha) * smoothed_old_cy
                    old_reference = smoothed_old_cy
                    old_missing_frames = 0
                else:
                    old_missing_frames += 1

                # Gather new sprocket samples when they appear above the last old reference
                if old_reference is not None:
                    new_candidates = [s for s in sprockets if s[1] < old_reference - new_sprocket_min_delta]
                else:
                    new_candidates = []

                if old_missing_frames == 0 and new_candidates:
                    chosen = min(new_candidates, key=lambda s: s[1])
                    new_sprocket_samples.append(chosen)
                    print(f"[APP] New sprocket sample cy={chosen[1]:.1f} (samples={len(new_sprocket_samples)})")
                elif old_missing_frames > 0:
                    if new_candidates:
                        chosen = min(new_candidates, key=lambda s: s[1])
                    else:
                        chosen = sprockets[0]
                    new_sprocket_samples.append(chosen)
                    print(f"[APP] New sprocket post-old sample cy={chosen[1]:.1f} (samples={len(new_sprocket_samples)})")

                if len(new_sprocket_samples) >= min_new_samples:
                    avg_cx = sum(s[0] for s in new_sprocket_samples) / len(new_sprocket_samples)
                    avg_cy = sum(s[1] for s in new_sprocket_samples) / len(new_sprocket_samples)

                    acceptable_position = abs(avg_cy - target_y) <= acceptance_tol_px

                    ready = old_missing_frames >= old_missing_required
                    if not ready and total_steps >= int(steps_per_pitch * 0.8):
                        ready = True
                    if not ready and acceptable_position:
                        ready = True

                    if ready:
                        print(f"[APP] Sprocket handoff after {total_steps} steps; new sprocket cy={avg_cy:.1f} (acceptable={acceptable_position})")

                        error_px = target_y - avg_cy
                        correction = int(error_px * steps_per_px * k_gain)
                        max_corr = int(0.15 * steps_per_pitch)
                        correction = max(min(correction, max_corr), -max_corr)

                        new_nominal = total_steps + correction
                        new_nominal = max(min(new_nominal, max_step), min_step)

                        print(f"[APP] Correction: error={error_px:+.1f}px → adjust {correction:+d} steps")
                        print(f"[APP] Updated nominal pitch for next advance: {new_nominal} steps (total_steps={total_steps})")

                        return (avg_cx, avg_cy, new_nominal)

        else:
            print(f"[APP] No sprockets detected in frame; continuing with small steps.")

        if total_steps >= max_step:
            print(f"[APP] Reached max_step {max_step} without confident sprocket handoff; returning fallback.")
            fallback_cx = last_seen_cx
            fallback_cy = last_seen_cy
            return (fallback_cx, fallback_cy, steps_per_pitch)

        tc.steps_forward(step_small)
        total_steps += step_small
        await asyncio.sleep(0.01)


PROJECTS_MOUNT_POINT = "/mnt/SG1TB"
if not os.path.ismount(PROJECTS_MOUNT_POINT):
    raise RuntimeError(f"Project storage drive is not mounted: {PROJECTS_MOUNT_POINT}")

PROJECTS_BASE_DIR = os.path.join(PROJECTS_MOUNT_POINT, "GrokCam", "projects")
os.makedirs(PROJECTS_BASE_DIR, exist_ok=True)

def sanitize_project_name(name):
    cleaned = str(name or "").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", cleaned)
    cleaned = cleaned.strip("_-")
    if not cleaned:
        raise ValueError("Project name must contain letters, numbers, spaces, dash, or underscore")
    return cleaned


def get_project_paths(project_safe_name):
    project_path = os.path.join(PROJECTS_BASE_DIR, project_safe_name)
    frames_path = os.path.join(project_path, "frames")
    debug_path = os.path.join(project_path, "debug")
    metadata_path = os.path.join(project_path, "metadata.json")
    return project_path, frames_path, debug_path, metadata_path


def set_active_project(project_name, project_safe_name, project_path):
    global active_project_name, active_project_safe_name, active_project_path
    active_project_name = project_name
    active_project_safe_name = project_safe_name
    active_project_path = project_path


def create_or_select_project(project_name):
    global latest_crop_preview_pair_midpoint, calibrated_baseline_registration_y
    safe_name = sanitize_project_name(project_name)
    project_path, frames_path, debug_path, metadata_path = get_project_paths(safe_name)
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    created_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    existing_metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                existing_metadata = json.load(handle)
            created_timestamp = existing_metadata.get("created_timestamp", created_timestamp)
        except Exception:
            pass

    metadata = {
        **existing_metadata,
        "project_name": str(project_name).strip(),
        "safe_folder_name": safe_name,
        "created_timestamp": created_timestamp,
        "base_path": PROJECTS_BASE_DIR,
        "project_path": project_path,
    }

    if not isinstance(metadata.get('crop'), dict):
        default_crop = settings.get('crop')
        if isinstance(default_crop, dict):
            metadata['crop'] = dict(default_crop)
            metadata['crop_source'] = 'global_default'
            metadata['crop_updated_timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

    save_project_metadata(metadata, project_path)

    set_active_project(metadata["project_name"], safe_name, project_path)
    latest_crop_preview_pair_midpoint = None
    project_crop = metadata.get('crop')
    saved_registration_y = (
        project_crop.get('registration_y')
        if isinstance(project_crop, dict) else None
    )
    calibrated_baseline_registration_y = (
        float(saved_registration_y) if saved_registration_y is not None else None
    )
    initialize_project_camera_settings(project_path, apply=False)
    reset_registration_tracking("project_selected")
    return {
        "name": metadata["project_name"],
        "safe_name": safe_name,
        "path": project_path,
        "frames_path": frames_path,
        "debug_path": debug_path,
        "metadata_path": metadata_path,
    }


def list_projects():
    projects = []
    for entry in sorted(os.listdir(PROJECTS_BASE_DIR)):
        project_path = os.path.join(PROJECTS_BASE_DIR, entry)
        if not os.path.isdir(project_path):
            continue

        metadata_path = os.path.join(project_path, "metadata.json")
        project_name = entry
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                project_name = metadata.get("project_name", project_name)
            except Exception:
                pass

        projects.append({
            "name": project_name,
            "safe_name": entry,
            "path": project_path,
        })

    return projects


def get_next_project_frame_number(frames_path):
    frame_pattern = re.compile(r"^frame_(\d+)\.png$")
    max_frame_number = 0
    if not os.path.exists(frames_path):
        return 1

    for entry in os.listdir(frames_path):
        match = frame_pattern.match(entry)
        if not match:
            continue
        max_frame_number = max(max_frame_number, int(match.group(1)))

    return max_frame_number + 1


def get_next_raw_frame_number(raw_path):
    frame_pattern = re.compile(r"^frame_(\d+)\.dng$")
    max_frame_number = 0
    if not os.path.exists(raw_path):
        return 1
    for entry in os.listdir(raw_path):
        match = frame_pattern.match(entry)
        if match:
            max_frame_number = max(max_frame_number, int(match.group(1)))
    return max_frame_number + 1


def append_registration_metadata(metadata_path, payload):
    try:
        with open(metadata_path, 'a', encoding='utf-8') as handle:
            json.dump(payload, handle)
            handle.write('\n')
    except Exception as exc:
        print(f"[APP] Registration metadata write failed: {exc}")


def save_dng_with_timing(request, output_path):
    """Save a DNG in a worker thread and return actual writer elapsed time."""
    started = time.perf_counter()
    request.save_dng(output_path)
    return (time.perf_counter() - started) * 1000.0


def get_active_camera_settings_state():
    if active_project_path and current_camera_settings.get('project_path') != active_project_path:
        initialize_project_camera_settings(active_project_path, apply=False)
    return current_camera_settings.copy()


def save_project_camera_settings(project_path=None):
    target_project_path = project_path or active_project_path
    if not target_project_path:
        raise RuntimeError('No active project selected')

    applied = get_active_camera_settings_state()
    settings_path = get_camera_settings_path(target_project_path)
    payload = {
        'ExposureTime': int(applied.get('ExposureTime', EXPOSURE_TIME)),
        'AnalogueGain': float(applied.get('AnalogueGain', GAIN)),
        'AeEnable': False,
        'AwbEnable': False,
        'source': 'manual',
        'timestamp': datetime.now().isoformat(),
    }
    if applied.get('ColourGains') is not None:
        payload['ColourGains'] = [
            float(applied['ColourGains'][0]),
            float(applied['ColourGains'][1]),
        ]
    with open(settings_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')

    set_current_camera_settings_state(
        payload['ExposureTime'],
        payload['AnalogueGain'],
        source='manual',
        saved=True,
        project_path=target_project_path,
        timestamp=payload['timestamp'],
        colour_gains=payload.get('ColourGains'),
    )
    print(f"[APP] Saved camera settings to {settings_path}")
    return settings_path


async def run_raw_capture(websocket, num_frames, stop_event):
    """Capture archival DNG frames with live sprocket registration."""
    if not active_project_path:
        raise RuntimeError("No active project selected")

    raw_path = os.path.join(active_project_path, "raw")
    debug_path = os.path.join(active_project_path, "debug")
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    next_frame_number = get_next_raw_frame_number(raw_path)
    capture_stamp = time.strftime('%Y%m%d-%H%M%S')
    metadata_path = os.path.join(
        debug_path, f"raw_capture_metadata_{capture_stamp}.jsonl"
    )
    fast_diagnostic_path = os.path.join(
        debug_path, f"raw_fast_failure_preview_{capture_stamp}.jpg"
    )
    fast_diagnostic_saved = False
    anomaly_path = os.path.join(debug_path, f"raw_anomalies_{capture_stamp}")
    os.makedirs(anomaly_path, exist_ok=True)

    configure_raw_camera()
    applied_camera_settings = apply_raw_capture_camera_controls()
    raw_fast_detector.reset()
    preview_pitch = SPROCKET_PITCH_PX * raw_preview_scale
    raw_tracker = RegistrationTracker(
        expected_sprocket_pitch_px=preview_pitch,
        max_jump_px=40.0 * raw_preview_scale,
        smoothing_alpha=0.8,
    )

    tc.light_on()
    camera.start()
    print("[APP] RAW capture: LED on + camera, stabilizing...")
    try:
        await asyncio.sleep(2)
        nominal_steps_per_pitch = resolve_nominal_steps(settings, STEPS_PER_PITCH)
        calibrated_steps_per_px = float(settings.get("steps_per_px", steps_per_px))
        full_pixels_per_step = 1.0 / calibrated_steps_per_px if calibrated_steps_per_px > 0 else 9.0
        pixels_per_step = full_pixels_per_step * raw_preview_scale
        dead_band_px = 10.0 * raw_preview_scale
        # These pre-existing 88%..112% bounds are the known safe transport
        # envelope.  Correction and base learning are independently bounded
        # inside it.
        min_steps = int(nominal_steps_per_pitch * 0.88)
        max_steps = int(nominal_steps_per_pitch * 1.12)
        project_manifest = load_project_metadata(active_project_path)
        saved_transport_state = None
        if next_frame_number > 1:
            saved_transport_state = project_manifest.get('transport_calibration_state')
        transport = AdaptiveTransportController(
            base_steps=nominal_steps_per_pitch,
            pixels_per_step=pixels_per_step,
            correction_gain=float(settings.get('transport_correction_gain', 0.25)),
            integral_gain=float(settings.get('transport_integral_gain', 0.02)),
            min_correction=int(settings.get('transport_min_correction', -8)),
            max_correction=int(settings.get('transport_max_correction', 8)),
            min_command=int(settings.get('transport_min_steps', min_steps)),
            max_command=int(settings.get('transport_max_steps', max_steps)),
            adaptation_frames=int(settings.get('transport_adaptation_frames', 30)),
            warning_frames=int(settings.get('transport_saturation_warning_frames', 15)),
            warning_interval=int(settings.get('transport_saturation_warning_interval', 100)),
            bias_window_size=int(settings.get('transport_bias_window_size', 300)),
            bias_cooldown_samples=int(settings.get('transport_bias_cooldown_samples', 100)),
            bias_median_threshold=float(settings.get('transport_bias_median_threshold', 4)),
            bias_share_threshold=float(settings.get('transport_bias_share_threshold', 0.80)),
            bias_stop_band=float(settings.get('transport_bias_stop_band', 1)),
            bias_warning_interval=int(settings.get('transport_bias_warning_interval', 300)),
            state=saved_transport_state,
        )
        current_steps = transport.adaptive_base_steps
        initial_transport_diagnostics = transport.diagnostics()
        project_manifest['transport_calibration'] = initial_transport_diagnostics
        save_project_metadata(project_manifest, active_project_path)
        # Crop calibration records the relationship between the picture crop
        # and whichever sprocket pair happened to be visible at that moment.
        # It is not a safe transport target: the saved value can put the upper
        # hole at the preview boundary, where a few pixels of normal variation
        # turn a pair into a partial/single detection. Keep transport centered;
        # the crop remains registration-relative and follows the detected pair.
        target_y = RAW_PREVIEW_SIZE[1] / 2.0
        print(
            f"[APP] RAW registration controller: target_y={target_y:.2f}, "
            f"nominal_steps={nominal_steps_per_pitch}, exposure={applied_camera_settings['ExposureTime']}"
        )
        if saved_transport_state and transport.restore_status != 'same_nominal_restored':
            saved_nominal = saved_transport_state.get('configured_nominal_steps', 'missing')
            print(
                f"[APP] Transport state reset: status={transport.restore_status}, "
                f"saved_nominal={saved_nominal}, configured_nominal={nominal_steps_per_pitch}, "
                "adaptive base reset to configured nominal and integral reset to zero"
            )
        print(
            "[APP] Transport calibration:\n"
            f"  adaptive base steps: {transport.adaptive_base_steps}\n"
            f"  correction range: {transport.min_correction:+d} .. {transport.max_correction:+d}\n"
            f"  absolute motor command range: {transport.min_command} .. {transport.max_command}"
        )
        missing_pair_count = 0
        trusted_step_history = deque(maxlen=5)
        previous_trusted_pair_y = None
        anomaly_count = 0

        for frame_index in range(1, int(num_frames) + 1):
            if stop_event.is_set():
                break
            frame_started = time.perf_counter()
            tc.steps_forward(current_steps)
            await asyncio.sleep(0.05)
            fresh_after_ns = time.monotonic_ns()

            request = None
            dng_future = None
            try:
                # Picamera2 0.3 on this Pi has no capture_request(flush=...).
                # Discard completed requests whose sensor timestamp predates the
                # end of the post-transport settling interval.
                discarded_requests = 0
                for attempt in range(7):
                    candidate = camera.capture_request()
                    candidate_metadata = candidate.get_metadata() or {}
                    sensor_timestamp = candidate_metadata.get('SensorTimestamp')
                    timestamp_is_fresh = (
                        sensor_timestamp is not None
                        and int(sensor_timestamp) >= fresh_after_ns
                    )
                    timestamp_unavailable_but_drained = (
                        sensor_timestamp is None and attempt >= 2
                    )
                    if timestamp_is_fresh or timestamp_unavailable_but_drained or attempt == 6:
                        request = candidate
                        camera_metadata = candidate_metadata
                        break
                    candidate.release()
                    discarded_requests += 1

                if request is None:
                    raise RuntimeError("Unable to acquire a post-transport camera request")
                preview_bgr = request.make_array("main")
                preview_clip_pct = float(
                    np.count_nonzero(np.max(preview_bgr, axis=2) >= 250)
                    / (preview_bgr.shape[0] * preview_bgr.shape[1]) * 100.0
                )
                frame_number = next_frame_number + frame_index - 1
                dng_path = os.path.join(raw_path, f"frame_{frame_number:06d}.dng")
                pending_dng_path = os.path.join(raw_path, f"frame_{frame_number:06d}.partial.dng")
                loop = asyncio.get_running_loop()
                dng_future = loop.run_in_executor(
                    None, save_dng_with_timing, request, pending_dng_path
                )

                detection_started = time.perf_counter()
                sprockets = raw_fast_detector.detect(preview_bgr)
                fast_failure_reason = raw_fast_detector.last_failure
                detection_method = "fast"
                crosscheck_sprockets = None
                crosscheck_registration_y = None
                detector_disagreement_px = None
                crosscheck_mode = None
                crosscheck_full_count = None
                crosscheck_partial_count = None
                if not sprockets:
                    sprockets = raw_fallback_detector.detect(preview_bgr, mode="profile") or []
                    detection_method = "fallback" if sprockets else "failed"
                elif frame_index % 25 == 0 or preview_clip_pct >= 15.0:
                    crosscheck_sprockets = raw_fallback_detector.detect(
                        preview_bgr, mode="profile"
                    ) or []
                detection_ms = (time.perf_counter() - detection_started) * 1000.0

                classified = raw_fallback_detector.classify_sprockets(sprockets, preview_bgr.shape)
                full_sprockets = [
                    item['sprocket'] for item in classified
                    if item.get('status') == 'full'
                ]
                full_count = len(full_sprockets)
                partial_count = sum(1 for item in classified if item.get('status') == 'partial')
                if detection_method == 'fallback' and full_sprockets:
                    raw_fast_detector.seed(full_sprockets, preview_bgr.shape)
                raw_registration = raw_fallback_detector.choose_registration(
                    full_sprockets, preview_bgr.shape, expected_pitch=preview_pitch
                )
                raw_mode = raw_registration.get('mode', 'none') if raw_registration else 'none'
                raw_y = raw_registration.get('actual_y') if raw_registration else None

                if crosscheck_sprockets is not None:
                    crosscheck_classified = raw_fallback_detector.classify_sprockets(
                        crosscheck_sprockets, preview_bgr.shape
                    )
                    crosscheck_full_sprockets = [
                        item['sprocket'] for item in crosscheck_classified
                        if item.get('status') == 'full'
                    ]
                    crosscheck_full_count = len(crosscheck_full_sprockets)
                    crosscheck_partial_count = sum(
                        1 for item in crosscheck_classified
                        if item.get('status') == 'partial'
                    )
                    crosscheck_choice = raw_fallback_detector.choose_registration(
                        crosscheck_full_sprockets,
                        preview_bgr.shape,
                        expected_pitch=preview_pitch,
                    ) or {}
                    crosscheck_mode = crosscheck_choice.get('mode', 'none')
                    if crosscheck_mode == 'pair':
                        crosscheck_registration_y = crosscheck_choice.get('actual_y')
                if raw_y is not None and crosscheck_registration_y is not None:
                    detector_disagreement_px = abs(
                        float(raw_y) - float(crosscheck_registration_y)
                    )
                    if detector_disagreement_px > 20.0 and crosscheck_mode == 'pair':
                        sprockets = crosscheck_full_sprockets
                        full_sprockets = crosscheck_full_sprockets
                        full_count = len(full_sprockets)
                        partial_count = int(crosscheck_partial_count or 0)
                        raw_mode = 'pair'
                        raw_y = float(crosscheck_registration_y)
                        detection_method = 'fallback_validation'
                        raw_fast_detector.seed(full_sprockets, preview_bgr.shape)
                tracked = raw_tracker.update(
                    raw_registration_y=raw_y,
                    raw_registration_mode=raw_mode,
                    frame_index=frame_index,
                    expected_sprocket_pitch_px=preview_pitch,
                )
                registration_y = tracked.get('stable_registration_y')
                crop_rect, crop_meta = get_scaled_relative_crop_rect(preview_bgr, registration_y)

                anomaly_reasons = []
                if detection_method == 'fallback':
                    anomaly_reasons.append('fast_fallback')
                elif detection_method == 'failed':
                    anomaly_reasons.append('both_detectors_failed')
                if len(sprockets) == 1:
                    anomaly_reasons.append('single_sprocket')
                elif len(sprockets) > 2:
                    anomaly_reasons.append('unexpected_sprocket_count')
                if partial_count > 0:
                    anomaly_reasons.append('partial_sprocket')
                if (
                    detector_disagreement_px is not None
                    and detector_disagreement_px > 20.0
                ):
                    anomaly_reasons.append('detector_disagreement')
                if (
                    crosscheck_sprockets is not None
                    and crosscheck_mode != 'pair'
                ):
                    anomaly_reasons.append('fallback_crosscheck_untrusted')
                if (
                    raw_mode == 'pair'
                    and raw_y is not None
                    and previous_trusted_pair_y is not None
                    and abs(float(raw_y) - float(previous_trusted_pair_y)) > 0.22 * preview_pitch
                ):
                    anomaly_reasons.append('registration_phase_jump')

                if raw_mode == 'pair':
                    missing_pair_count = 0
                else:
                    missing_pair_count += 1

                steps_before_update = current_steps
                correction = 0
                control_result = None
                reacquire_attempted = False
                reacquire_valid = False
                reacquire_steps = 0
                reacquire_reason = None
                reacquire_ms = 0.0
                next_steps = transport.adaptive_base_steps
                if raw_y is not None:
                    error_px = float(target_y) - float(raw_y)
                    update_allowed = (
                        raw_mode == 'pair'
                        and full_count >= 2
                        and partial_count == 0
                        and 'registration_phase_jump' not in anomaly_reasons
                        and (
                            detector_disagreement_px is None
                            or detector_disagreement_px <= 20.0
                            or detection_method == 'fallback_validation'
                        )
                    )
                    if update_allowed:
                        controlled_error = error_px if abs(error_px) > dead_band_px else 0.0
                        control_result = transport.update(controlled_error)
                        correction = control_result.correction
                        next_steps = control_result.commanded_steps
                        current_steps = next_steps
                        trusted_step_history.append(int(current_steps))
                        previous_trusted_pair_y = float(raw_y)
                        if control_result.warning:
                            saturated_run = max(
                                transport.negative_saturation_run,
                                transport.positive_saturation_run,
                            )
                        if control_result.bias_warning:
                            rolling = transport.rolling_statistics()
                            print(
                                f"[APP] WARNING: persistent transport bias: frame={frame_number} "
                                f"window={rolling['sample_count']} median={rolling['median_correction']:+.1f} "
                                f"negative_share={rolling['negative_share']:.1%} positive_share={rolling['positive_share']:.1%} "
                                f"range={rolling['min_correction']:+d}..{rolling['max_correction']:+d} "
                                f"adaptive_base={transport.adaptive_base_steps} nominal={nominal_steps_per_pitch} "
                                f"integral={transport.integral_steps:+.3f} "
                                f"correction_limits={transport.min_correction:+d}..{transport.max_correction:+d} "
                                f"motor_limits={transport.min_command}..{transport.max_command} "
                                f"status={control_result.bias_warning_status} "
                                f"adaptation={control_result.adaptation_reason or 'none'}"
                            )
                            print(
                                f"[APP] WARNING: sustained transport saturation: frame={frame_number} "
                                f"correction={correction:+d} limits={transport.min_correction:+d}..{transport.max_correction:+d} "
                                f"adaptive_base_steps={transport.adaptive_base_steps} "
                                f"commanded_steps={current_steps} consecutive_frames={saturated_run}"
                            )
                    else:
                        # Do not repeat a corrective command when registration
                        # is partial or otherwise untrusted.
                        transport.update(None, trusted=False)
                        current_steps = next_steps
                else:
                    error_px = None
                    transport.update(None, trusted=False)
                    current_steps = next_steps

                if missing_pair_count >= 3:
                    reacquire_attempted = True
                    anomaly_reasons.append('reacquisition')
                    reacquire_started = time.perf_counter()
                    reacquire = await reacquire_pair_registration(
                        camera, tc, raw_fallback_detector, target_y,
                        step_size=10, max_steps=300,
                    )
                    reacquire_ms = (time.perf_counter() - reacquire_started) * 1000.0
                    reacquire_valid = bool(reacquire.get('valid'))
                    reacquire_steps = int(reacquire.get('steps', 0))
                    reacquire_reason = reacquire.get('reason')
                    current_steps = transport.adaptive_base_steps
                    if reacquire.get('valid'):
                        missing_pair_count = 0
                        reacquired_y = reacquire.get('registration_y')
                        if reacquired_y is not None:
                            previous_trusted_pair_y = float(reacquired_y)

                encode_started = time.perf_counter()
                ok, encoded = cv2.imencode(
                    '.jpg', preview_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                )
                if not ok:
                    raise RuntimeError("Failed to encode RAW capture preview")
                preview_bytes = encoded.tobytes()
                encode_ms = (time.perf_counter() - encode_started) * 1000.0

                if detection_method != 'fast' and not fast_diagnostic_saved:
                    with open(fast_diagnostic_path, 'wb') as handle:
                        handle.write(preview_bytes)
                    fast_diagnostic_saved = True
                    print(f"[APP] Saved fast-detector diagnostic preview: {fast_diagnostic_path}")

                collection_reasons = list(dict.fromkeys(anomaly_reasons))
                if frame_index % 50 == 0:
                    collection_reasons.append('control_sample')
                anomaly_preview_path = None
                if collection_reasons:
                    reason_slug = '-'.join(collection_reasons)
                    anomaly_preview_path = os.path.join(
                        anomaly_path,
                        f"frame_{frame_number:06d}_{reason_slug}.jpg",
                    )
                    with open(anomaly_preview_path, 'wb') as handle:
                        handle.write(preview_bytes)
                    if anomaly_reasons:
                        anomaly_count += 1

                dng_wait_started = time.perf_counter()
                dng_write_ms = float(await dng_future)
                dng_wait_ms = (time.perf_counter() - dng_wait_started) * 1000.0
                os.replace(pending_dng_path, dng_path)
                total_ms = (time.perf_counter() - frame_started) * 1000.0

                frame_metadata = {
                    'protocol': RAW_CAPTURE_MODE,
                    'frame_index': frame_index,
                    'frame_number': frame_number,
                    'saved_frame_path': dng_path,
                    'preview_width': int(preview_bgr.shape[1]),
                    'preview_height': int(preview_bgr.shape[0]),
                    'raw_width': RAW_SENSOR_SIZE[0],
                    'raw_height': RAW_SENSOR_SIZE[1],
                    'crop_rect': [int(value) for value in crop_rect],
                    'orientation_degrees': 180,
                    'detection_method': detection_method,
                    'fast_failure_reason': fast_failure_reason,
                    'crosscheck_sprockets': (
                        [[float(value) for value in item] for item in crosscheck_sprockets]
                        if crosscheck_sprockets is not None else None
                    ),
                    'crosscheck_registration_y': (
                        float(crosscheck_registration_y)
                        if crosscheck_registration_y is not None else None
                    ),
                    'crosscheck_mode': crosscheck_mode,
                    'crosscheck_full_count': crosscheck_full_count,
                    'crosscheck_partial_count': crosscheck_partial_count,
                    'detector_disagreement_px': (
                        float(detector_disagreement_px)
                        if detector_disagreement_px is not None else None
                    ),
                    'sprockets': [[float(value) for value in item] for item in sprockets],
                    'raw_registration_mode': raw_mode,
                    'raw_registration_y': float(raw_y) if raw_y is not None else None,
                    'selected_registration_y': registration_y,
                    'selected_source': tracked.get('selected_source'),
                    'crop_clamped': bool(crop_meta.get('crop_clamped')),
                    'steps': int(steps_before_update),
                    'target_y': float(target_y),
                    'registration_error_px': float(error_px) if error_px is not None else None,
                    'adaptive_base_steps': int(transport.adaptive_base_steps),
                    'transport_integral_steps': round(float(transport.integral_steps), 4),
                    'configured_nominal_steps': int(nominal_steps_per_pitch),
                    'transport_rolling_sample_count': len(transport.correction_window),
                    'transport_rolling_median_correction': transport.rolling_statistics()['median_correction'],
                    'transport_rolling_negative_share': transport.rolling_statistics()['negative_share'],
                    'transport_rolling_positive_share': transport.rolling_statistics()['positive_share'],
                    'transport_cooldown_remaining': transport.cooldown_remaining,
                    'transport_last_adaptation_reason': transport.last_adaptation_reason,
                    'transport_active_constraint': control_result.active_constraint if control_result else 'neither',
                    'correction': int(correction),
                    'next_steps': int(current_steps),
                    'reacquire_attempted': reacquire_attempted,
                    'reacquire_valid': reacquire_valid,
                    'reacquire_steps': reacquire_steps,
                    'reacquire_reason': reacquire_reason,
                    'reacquire_ms': round(reacquire_ms, 2),
                    'anomaly_reasons': anomaly_reasons,
                    'anomaly_preview_path': anomaly_preview_path,
                    'camera_exposure_time': int(camera_metadata.get('ExposureTime', applied_camera_settings.get('ExposureTime', EXPOSURE_TIME))),
                    'camera_analogue_gain': float(camera_metadata.get('AnalogueGain', applied_camera_settings.get('AnalogueGain', GAIN))),
                    'preview_clip_pct': round(preview_clip_pct, 3),
                    'timing_detection_ms': round(detection_ms, 2),
                    'timing_preview_encode_ms': round(encode_ms, 2),
                    'timing_dng_write_ms': round(dng_write_ms, 2),
                    'timing_dng_ms': round(dng_write_ms, 2),
                    'timing_dng_wait_ms': round(dng_wait_ms, 2),
                    'timing_total_ms': round(total_ms, 2),
                    'discarded_stale_requests': int(discarded_requests),
                }
                append_registration_metadata(metadata_path, frame_metadata)
                if frame_index % 10 == 0:
                    project_manifest = load_project_metadata(active_project_path)
                    project_manifest['transport_calibration_state'] = transport.state()
                    project_manifest['transport_calibration'] = transport.diagnostics()
                    save_project_metadata(project_manifest, active_project_path)

                await websocket.send(json.dumps({
                    'event': 'capture_preview_v1',
                    'frame': frame_number,
                    'frame_index': frame_index,
                    'frame_total': int(num_frames),
                    'binary_size': len(preview_bytes),
                    'source_width': int(preview_bgr.shape[1]),
                    'source_height': int(preview_bgr.shape[0]),
                    'crop_rect': [int(value) for value in crop_rect],
                    'orientation_degrees': 180,
                    'detection_method': detection_method,
                    'fast_failure_reason': fast_failure_reason,
                    'registration_source': tracked.get('selected_source'),
                }))
                await websocket.send(preview_bytes)
                await websocket.send(json.dumps({
                    'event': 'info',
                    'message': f'RAW image {frame_index} of {num_frames}',
                    'frame_index': frame_index,
                    'frame_total': int(num_frames),
                }))
                registration_log = (
                    f"reg={raw_y:.1f} target={target_y:.1f} err={error_px:+.1f}"
                    if raw_y is not None and error_px is not None else
                    f"reg=n/a target={target_y:.1f} err=n/a"
                )
                print(
                    f"[APP] RAW frame {frame_number}: {detection_method}, {registration_log} "
                    f"steps={steps_before_update} "
                    f"base={transport.adaptive_base_steps} correction={correction:+d} next={current_steps}; "
                    f"discarded={discarded_requests} detect={detection_ms:.1f}ms "
                    f"dng_write={dng_write_ms:.1f}ms reacquire={reacquire_ms:.1f}ms "
                    f"total={total_ms:.1f}ms anomalies={','.join(anomaly_reasons) or 'none'}"
                )
            finally:
                if dng_future is not None and not dng_future.done():
                    try:
                        await dng_future
                    except Exception as exc:
                        print(f"[APP] RAW DNG writer failed during cleanup: {exc}")
                if request is not None:
                    request.release()

        project_manifest = load_project_metadata(active_project_path)
        project_manifest['transport_calibration_state'] = transport.state()
        project_manifest['transport_calibration'] = transport.diagnostics()
        save_project_metadata(project_manifest, active_project_path)
        append_registration_metadata(metadata_path, {
            'event': 'transport_calibration_summary',
            **transport.diagnostics(),
        })
        await websocket.send(json.dumps({
            'event': 'capture_complete',
            'capture_mode': RAW_CAPTURE_MODE,
            'anomaly_count': anomaly_count,
            'anomaly_path': anomaly_path,
            'metadata_path': metadata_path,
        }))
        print(
            f"[APP] RAW capture complete: anomalies={anomaly_count}, "
            f"anomaly_path={anomaly_path}"
        )
    finally:
        tc.clean_up()
        camera.stop()
        configure_legacy_camera()
        print("[APP] RAW capture cleaned up; legacy camera configuration restored")

async def run_capture(websocket, num_frames, stop_event, preview_width=800, debug_scale=1.0):
    print("[APP] Capture task starting")
    if not active_project_path:
        raise RuntimeError("No active project selected")

    project_name = active_project_name
    project_path = active_project_path
    frames_path = os.path.join(project_path, "frames")
    debug_path = os.path.join(project_path, "debug")
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    next_frame_number = get_next_project_frame_number(frames_path)
    metadata_path = os.path.join(
        debug_path,
        f"registration_metadata_{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
    )

    tc.light_on()
    applied_camera_settings = apply_capture_camera_controls()
    camera.start()
    reset_registration_tracking("capture_start")
    print("[APP] LED on + camera, stabilizing...")
    try:
        await asyncio.sleep(2)

        nominal_steps_per_pitch = int(settings.get("steps_per_pitch", STEPS_PER_PITCH))
        calibrated_steps_per_px = float(settings.get("steps_per_px", steps_per_px))
        base_steps = nominal_steps_per_pitch
        pixels_per_step = 1.0 / calibrated_steps_per_px if calibrated_steps_per_px > 0 else 9.0
        current_steps = base_steps
        gain = 0.4
        max_correction = 6
        DEAD_BAND_PX = 10
        min_steps = int(nominal_steps_per_pitch * 0.88)
        max_steps = int(nominal_steps_per_pitch * 1.12)
        target_y = None
        missing_pair_count = 0
        trusted_step_history = deque(maxlen=5)
        previous_valid_pair_error = None
        freeze_after_reacquire = False

        for frame in range(num_frames):
            if stop_event.is_set():
                print("[APP] Stop requested, leaving capture loop")
                break

            tc.steps_forward(current_steps)
            await asyncio.sleep(0.05)

            buffer = io.BytesIO()
            camera.capture_file(buffer, format='jpeg')
            buffer.seek(0)
            frame_bgr = cv2.imdecode(
                np.frombuffer(buffer.getvalue(), np.uint8),
                cv2.IMREAD_COLOR
            )

            sprockets = detector.detect(frame_bgr, mode="profile")
            classified_sprockets = detector.classify_sprockets(sprockets if sprockets else [], frame_bgr.shape)
            full_count = sum(1 for item in classified_sprockets if item.get('status') == 'full')
            partial_count = sum(1 for item in classified_sprockets if item.get('status') == 'partial')
            raw_registration = get_capture_registration(frame_bgr, sprockets if sprockets else [])
            raw_registration_y = raw_registration.get('registration_y')
            raw_registration_mode = raw_registration.get('mode', 'none')
            tracked_registration = registration_tracker.update(
                raw_registration_y=raw_registration_y,
                raw_registration_mode=raw_registration_mode,
                frame_index=frame + 1,
                expected_sprocket_pitch_px=SPROCKET_PITCH_PX,
            )
            registration_y = tracked_registration.get('stable_registration_y')
            crop_rect, crop_meta = get_relative_crop_rect(frame_bgr, registration_y, return_metadata=True)
            debug_frame = draw_sprockets_debug(
                frame_bgr,
                sprockets if sprockets else [],
                registration_y=registration_y,
                crop_rect=crop_rect,
                crop_clamped=bool(crop_meta.get('crop_clamped')),
            )

            selected_source = tracked_registration.get('selected_source')

            if selected_source == 'rejected_single':
                print(
                    f"[APP] Frame {frame}: rejected single estimate raw={tracked_registration.get('single_sprocket_y')} "
                    f"last_good={tracked_registration.get('last_good_registration_y')}"
                )
            elif selected_source == 'held_last_good':
                print(
                    f"[APP] Frame {frame}: no registration found; holding last good "
                    f"{tracked_registration.get('last_good_registration_y')}"
                )

            if crop_meta.get('crop_clamped'):
                print(
                    f"[APP] Frame {frame}: crop clamped to y1={crop_meta.get('crop_y1')} "
                    f"y2={crop_meta.get('crop_y2')}"
                )

            if raw_registration_mode == 'pair':
                missing_pair_count = 0
            else:
                missing_pair_count += 1

            sprocket_count = len(sprockets) if sprockets else 0

            if target_y is None:
                target_y = frame_bgr.shape[0] / 2.0

            steps_before_update = current_steps

            if raw_registration_y is None:
                next_steps = max(min_steps, min(max_steps, base_steps))
                current_steps = next_steps
                if raw_registration_mode != 'pair':
                    print(
                        f"[APP] Frame {frame}: pair unavailable: count={sprocket_count}, "
                        f"full={full_count}, partial={partial_count}, mode={raw_registration_mode}"
                    )
                print(
                    f"[APP] Frame {frame}: reg=n/a, mode={raw_registration_mode}, count={sprocket_count}, "
                    f"full={full_count}, partial={partial_count}, err=n/a, steps={steps_before_update}, correction=+0, next={next_steps}"
                )
            else:
                error_px = float(target_y) - float(raw_registration_y)
                update_allowed = raw_registration_mode == 'pair' and full_count == 2 and partial_count == 0
                if raw_registration_mode == 'pair' and partial_count > 0:
                    print(f"[APP] Frame {frame}: ignoring partial pair for step update")

                correction = 0
                next_steps = max(min_steps, min(max_steps, base_steps))

                if update_allowed:
                    abs_error = abs(error_px)
                    if abs_error <= DEAD_BAND_PX:
                        correction = 0
                    else:
                        correction = int(round((error_px / pixels_per_step) * gain))
                        correction = max(-max_correction, min(max_correction, correction))

                    next_steps = base_steps + correction
                    next_steps = max(min_steps, min(max_steps, next_steps))
                    current_steps = next_steps
                    trusted_step_history.append(int(current_steps))
                else:
                    current_steps = next_steps
                    print(
                        f"[APP] Frame {frame}: pair unavailable: count={sprocket_count}, "
                        f"full={full_count}, partial={partial_count}, mode={raw_registration_mode}"
                    )
                    print(f"[APP] Frame {frame}: ignoring low-confidence registration for step update")

                print(
                    f"[APP] Frame {frame}: reg={raw_registration_y:.1f}, mode={raw_registration_mode}, "
                    f"count={sprocket_count}, full={full_count}, partial={partial_count}, "
                    f"err={error_px:+.1f}px, steps={steps_before_update}, "
                    f"correction={correction:+d}, next={next_steps}"
                )

            if missing_pair_count >= 3:
                print("[APP] Pair registration lost; attempting reacquire")
                reacquire_result = await reacquire_pair_registration(
                    camera,
                    tc,
                    detector,
                    target_y,
                    step_size=10,
                    max_steps=300,
                )
                if reacquire_result.get('valid'):
                    missing_pair_count = 0
                    if trusted_step_history:
                        current_steps = int(round(float(np.median(np.array(trusted_step_history, dtype=float)))))
                    else:
                        current_steps = nominal_steps_per_pitch
                    previous_valid_pair_error = None
                    freeze_after_reacquire = True
                    print(
                        f"[APP] Reacquire succeeded: steps={reacquire_result.get('steps')}, "
                        f"registration_y={reacquire_result.get('registration_y'):.1f}; "
                        f"resetting current_steps={current_steps}"
                    )
                else:
                    if trusted_step_history:
                        current_steps = int(round(float(np.median(np.array(trusted_step_history, dtype=float)))))
                    else:
                        current_steps = nominal_steps_per_pitch
                    print(
                        f"[APP] Reacquire failed: reason={reacquire_result.get('reason')}, "
                        f"steps={reacquire_result.get('steps')}; resetting current_steps={current_steps}"
                    )

            crop_x1, crop_y1, crop_x2, crop_y2 = crop_rect
            frame_cropped = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            # Flip vertically to correct camera orientation (matches legacy behavior)
            frame_cropped = cv2.flip(frame_cropped, 0)

            frame_number = next_frame_number + frame
            filename = os.path.join(frames_path, f"frame_{frame_number:06d}.png")
            save_ok = cv2.imwrite(filename, frame_cropped)  # PNG = lossless
            if not save_ok:
                print(f"[APP] WARNING: Failed to save cropped frame to {filename}")
                await websocket.send(json.dumps({
                    'event': 'warning',
                    'message': f'Failed to save frame {frame} to disk'
                }))

            frame_metadata = {
                'frame_index': int(frame + 1),
                'frame_number': int(frame_number),
                'raw_registration_mode': tracked_registration.get('raw_registration_mode'),
                'raw_registration_y': tracked_registration.get('raw_registration_y'),
                'raw_pair_midpoint_y': tracked_registration.get('raw_pair_midpoint_y'),
                'single_sprocket_y': tracked_registration.get('single_sprocket_y'),
                'last_good_registration_y': tracked_registration.get('last_good_registration_y'),
                'baseline_registration_y': tracked_registration.get('baseline_registration_y'),
                'selected_registration_y': tracked_registration.get('selected_registration_y'),
                'selected_source': tracked_registration.get('selected_source'),
                'registration_error_px': tracked_registration.get('registration_error_px'),
                'crop_y1': int(crop_meta.get('crop_y1', crop_y1)),
                'crop_y2': int(crop_meta.get('crop_y2', crop_y2)),
                'crop_clamped': bool(crop_meta.get('crop_clamped')),
                'camera_exposure_time': int(applied_camera_settings.get('ExposureTime', EXPOSURE_TIME)),
                'camera_analogue_gain': float(applied_camera_settings.get('AnalogueGain', GAIN)),
                'camera_settings_saved': bool(applied_camera_settings.get('saved', False)),
                'camera_settings_source': applied_camera_settings.get('source', 'default'),
                'failure_count': int(tracked_registration.get('failure_count', 0)),
                'saved_frame_path': filename,
            }
            append_registration_metadata(metadata_path, frame_metadata)

            scale_w = max(1, int(preview_width))
            scale_h = int(frame_cropped.shape[0] * (scale_w / frame_cropped.shape[1]))
            frame_lowres = cv2.resize(frame_cropped, (scale_w, scale_h), interpolation=cv2.INTER_AREA)

            if debug_scale != 1.0:
                dbg_scale = max(0.1, float(debug_scale))
                dbg_w = int(debug_frame.shape[1] * dbg_scale)
                dbg_h = int(debug_frame.shape[0] * dbg_scale)
                debug_frame_resized = cv2.resize(debug_frame, (dbg_w, dbg_h), interpolation=cv2.INTER_NEAREST)
            else:
                debug_frame_resized = debug_frame

            _, cropped_bytes = await encode_frame_async(frame_lowres, frame)
            _, debug_bytes = await encode_frame_async(debug_frame_resized, frame)

            header = len(cropped_bytes).to_bytes(4, 'big')
            payload = header + cropped_bytes + debug_bytes
            await websocket.send(payload)
            await websocket.send(json.dumps({
                'event': 'info',
                'message': f'Image {frame + 1} of {num_frames}',
                'frame_index': frame + 1,
                'frame_total': num_frames,
            }))
            print(f"[APP] Frame {frame}: sent {len(payload)} bytes to client")


            if save_ok:
                print(f"[APP] Sent frame {frame} → saved cropped {filename}")
            else:
                print(f"[APP] Sent frame {frame} → no disk save")
            await asyncio.sleep(0.1)

        await websocket.send(json.dumps({'event': 'capture_complete'}))
    finally:
        tc.clean_up()
        camera.stop()
        print("[APP] Capture task cleaned up")

async def run_preview_stream(websocket, stop_event, preview_width=800, fps=5, mode='focus'):
    print(f"[APP] {mode} preview task starting")
    tc.light_on()
    camera.start()
    if mode == 'focus':
        apply_focus_camera_controls()
        started_event = 'focus_started'
        frame_event = 'focus_frame'
        stopped_event = 'focus_stopped'
        print("[APP] LED on + camera for focus")
    else:
        apply_project_capture_camera_settings(active_project_path, prefer_saved=False)
        started_event = 'camera_calibration_preview_started'
        frame_event = 'camera_calibration_preview_frame'
        stopped_event = 'camera_calibration_preview_stopped'
        print("[APP] LED on + camera for camera calibration preview")

    frame_num = 0
    preview_width = max(1, int(preview_width))
    frame_delay = 1.0 / max(1.0, float(fps))

    try:
        await asyncio.sleep(1.0)
        await websocket.send(json.dumps({
            'event': started_event,
            'type': started_event,
        }))

        while not stop_event.is_set():
            buffer = io.BytesIO()
            camera.capture_file(buffer, format='jpeg')
            frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError('Failed to decode focus preview frame')

            frame_num += 1
            scale_h = int(frame_bgr.shape[0] * (preview_width / frame_bgr.shape[1]))
            frame_resized = cv2.resize(
                frame_bgr,
                (preview_width, scale_h),
                interpolation=cv2.INTER_AREA
            )
            frame_flipped = cv2.flip(frame_resized, 0)

            ok, encoded = cv2.imencode(
                '.jpg',
                frame_flipped,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not ok:
                raise RuntimeError('Failed to encode focus preview frame')

            jpg_bytes = encoded.tobytes()
            await websocket.send(json.dumps({
                'event': frame_event,
                'type': frame_event,
                'frame': frame_num,
                'size': len(jpg_bytes)
            }))
            await websocket.send(jpg_bytes)
            await asyncio.sleep(frame_delay)
    finally:
        tc.light_off()
        camera.stop()
        try:
            await websocket.send(json.dumps({
                'event': stopped_event,
                'type': stopped_event,
            }))
        except (ConnectionClosedError, ConnectionClosedOK):
            print(f"[APP] {mode} stop notification skipped: client disconnected")
        except Exception as exc:
            print(f"[APP] {mode} stop notification error: {exc}")
        print(f"[APP] {mode} preview task cleaned up")


async def run_focus(websocket, stop_event, preview_width=800, fps=5):
    return await run_preview_stream(websocket, stop_event, preview_width=preview_width, fps=fps, mode='focus')


async def run_camera_calibration_preview(websocket, stop_event, preview_width=800, fps=5):
    return await run_preview_stream(websocket, stop_event, preview_width=preview_width, fps=fps, mode='camera_calibration')


async def run_capture_with_error_reporting(websocket, capture_coroutine):
    """Ensure background capture failures immediately release the GUI state."""
    try:
        await capture_coroutine
    except Exception as exc:
        print(f"[APP] Capture task error: {exc}")
        try:
            await websocket.send(json.dumps({
                'event': 'error',
                'message': f'Capture failed: {exc}',
            }))
        except (ConnectionClosedError, ConnectionClosedOK):
            pass


async def handle_client(websocket):
    print("Client connected")
    capture_task = None
    capture_stop_event = None
    focus_task = None
    focus_stop_event = None
    camera_calibration_task = None
    camera_calibration_stop_event = None
    latest_proposed_calibration = None
    latest_can_save = False
    latest_save_block_reason = 'no_calibration_sweep_run'
    try:
        async for message in websocket:
            print(f"[APP] Got message: {message}")
            data = json.loads(message)

            if capture_task and capture_task.done():
                try:
                    capture_task.result()
                except Exception as exc:
                    print(f"[APP] Capture task error: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Capture task failed'
                    }))
                capture_task = None
                capture_stop_event = None

            if focus_task and focus_task.done():
                try:
                    focus_task.result()
                except Exception as exc:
                    print(f"[APP] Focus task error: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Focus task failed'
                    }))
                focus_task = None
                focus_stop_event = None

            if camera_calibration_task and camera_calibration_task.done():
                try:
                    camera_calibration_task.result()
                except Exception as exc:
                    print(f"[APP] Camera calibration preview task error: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Camera calibration preview failed'
                    }))
                camera_calibration_task = None
                camera_calibration_stop_event = None

            event = data.get('event') or data.get('type')

            if event == 'create_project':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot create or switch project while capture is running'
                    }))
                    continue

                try:
                    project_info = create_or_select_project(data.get('name', ''))
                    if not (focus_task and not focus_task.done()) and not (camera_calibration_task and not camera_calibration_task.done()):
                        apply_project_capture_camera_settings(project_info['path'], prefer_saved=True)
                    print(f"[APP] Active project set: {project_info['name']} ({project_info['path']})")
                    await websocket.send(json.dumps({
                        'event': 'project_created',
                        'active_project_name': project_info['name'],
                        'active_project_safe_name': project_info['safe_name'],
                        'active_project_path': project_info['path'],
                        'frames_path': project_info['frames_path'],
                        'debug_path': project_info['debug_path'],
                        'metadata_path': project_info['metadata_path'],
                    }))
                except Exception as exc:
                    print(f"[APP] Create project failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Create project failed: {exc}'
                    }))
                continue

            elif event == 'list_projects':
                try:
                    await websocket.send(json.dumps({
                        'event': 'projects_list',
                        'projects': list_projects(),
                        'active_project_name': active_project_name,
                        'active_project_safe_name': active_project_safe_name,
                        'active_project_path': active_project_path,
                    }))
                except Exception as exc:
                    print(f"[APP] List projects failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'List projects failed: {exc}'
                    }))
                continue

            elif event == 'get_camera_settings':
                state = get_active_camera_settings_state()
                await websocket.send(json.dumps(camera_settings_response_payload(state)))
                continue

            elif event == 'set_camera_settings':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot change camera settings during capture'
                    }))
                    continue
                if focus_task and not focus_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot change manual camera settings while focus preview is active'
                    }))
                    continue

                requested_exposure = data.get('exposure_time')
                requested_gain = data.get('analogue_gain')
                if requested_exposure is None or requested_gain is None:
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Missing exposure_time or analogue_gain'
                    }))
                    continue

                try:
                    applied = apply_manual_camera_settings(
                        requested_exposure,
                        requested_gain,
                        source='manual',
                        saved=False,
                        project_path=active_project_path,
                    )
                except (TypeError, ValueError) as exc:
                    print(f"[APP] Invalid camera settings rejected: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Invalid camera settings payload'
                    }))
                    continue
                await websocket.send(json.dumps(camera_settings_response_payload(applied)))
                continue

            elif event == 'save_camera_settings':
                try:
                    settings_path = save_project_camera_settings(active_project_path)
                    await websocket.send(json.dumps({
                        'event': 'camera_settings_saved',
                        'type': 'camera_settings_saved',
                        'path': settings_path,
                    }))
                except Exception as exc:
                    print(f"[APP] Save camera settings failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Save camera settings failed: {exc}'
                    }))
                continue

            elif event == 'auto_set_exposure':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot run auto exposure during capture'
                    }))
                    continue

                preview_active = (
                    (focus_task and not focus_task.done())
                    or (camera_calibration_task and not camera_calibration_task.done())
                )
                try:
                    applied = await run_one_shot_auto_exposure(preview_active=bool(preview_active))
                    await websocket.send(json.dumps(camera_settings_response_payload(applied)))
                except Exception as exc:
                    print(f"[APP] Auto exposure failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Auto exposure failed: {exc}'
                    }))
                continue

            elif event == 'auto_set_awb':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot run auto white balance during capture'
                    }))
                    continue

                preview_active = (
                    (focus_task and not focus_task.done())
                    or (camera_calibration_task and not camera_calibration_task.done())
                )
                try:
                    applied = await run_one_shot_auto_awb(preview_active=bool(preview_active))
                    await websocket.send(json.dumps(camera_awb_response_payload(applied)))
                except Exception as exc:
                    print(f"[APP] Auto white balance failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Auto white balance failed: {exc}'
                    }))
                continue

            elif event == 'start_camera_calibration_preview':
                if camera_calibration_task and not camera_calibration_task.done():
                    await websocket.send(json.dumps({
                        'event': 'info',
                        'message': 'Camera calibration preview already active'
                    }))
                    continue
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot start camera calibration preview during capture'
                    }))
                    continue
                if focus_task and not focus_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot start camera calibration preview while focus preview is active'
                    }))
                    continue

                initialize_project_camera_settings(active_project_path, apply=False)
                camera_calibration_stop_event = asyncio.Event()
                preview_width = data.get('preview_width', 800)
                fps = data.get('fps', 5)
                camera_calibration_task = asyncio.create_task(
                    run_camera_calibration_preview(
                        websocket,
                        camera_calibration_stop_event,
                        preview_width=preview_width,
                        fps=fps,
                    )
                )
                continue

            elif event == 'stop_camera_calibration_preview':
                if camera_calibration_task and not camera_calibration_task.done():
                    camera_calibration_stop_event.set()
                    try:
                        await camera_calibration_task
                    finally:
                        camera_calibration_task = None
                        camera_calibration_stop_event = None
                else:
                    await websocket.send(json.dumps({
                        'event': 'info',
                        'message': 'Camera calibration preview not active'
                    }))
                continue

            if event == 'start_capture':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Capture already running'
                    }))
                    continue
                if focus_task and not focus_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot start capture while focus is active'
                    }))
                    continue
                if camera_calibration_task and not camera_calibration_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot start capture while camera calibration preview is active'
                    }))
                    continue
                if not active_project_path:
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot start capture without an active project'
                    }))
                    continue
                num_frames = data.get('num_frames', 100)
                # RAW is the archival default. Older clients can still request
                # legacy_png explicitly for compatibility.
                capture_mode = data.get('capture_mode', RAW_CAPTURE_MODE)
                preview_width = data.get('preview_width', 800)
                debug_scale = data.get('debug_scale', 1.0)
                capture_stop_event = asyncio.Event()
                if capture_mode == RAW_CAPTURE_MODE:
                    capture_task = asyncio.create_task(
                        run_capture_with_error_reporting(
                            websocket,
                            run_raw_capture(websocket, num_frames, capture_stop_event),
                        )
                    )
                else:
                    capture_task = asyncio.create_task(
                        run_capture_with_error_reporting(
                            websocket,
                            run_capture(
                                websocket,
                                num_frames,
                                capture_stop_event,
                                preview_width=preview_width,
                                debug_scale=debug_scale
                            ),
                        )
                    )
                continue

            elif event == 'stop_capture':
                if capture_task and not capture_task.done():
                    print("[APP] Stop requested")
                    capture_stop_event.set()
                    await websocket.send(json.dumps({
                        'event': 'info',
                        'message': 'Stop requested'
                    }))
                    try:
                        await capture_task
                    finally:
                        capture_task = None
                        capture_stop_event = None
                else:
                    await websocket.send(json.dumps({
                        'event': 'info',
                        'message': 'No active capture task'
                    }))
                continue

            elif event == 'calibration_preview':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot preview calibration while capture is running'
                    }))
                    continue
                if focus_task and not focus_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot preview calibration while focus is active'
                    }))
                    continue

                debug_scale = float(data.get('debug_scale', 1.0))

                try:
                    tc.light_on()
                    camera.start()
                    print("[APP] LED on + camera, stabilizing for calibration preview...")
                    await asyncio.sleep(0.5)
                    exposure_result = await prepare_calibration_camera_settings(detector)

                    measurement, jpg_bytes = calibrator.capture_sprocket_preview(debug_scale)

                    await websocket.send(json.dumps({
                        'event': 'calibration_measurement',
                        **measurement,
                        'exposure': exposure_result,
                        'size': len(jpg_bytes)
                    }))
                    await websocket.send(jpg_bytes)

                    print(f"[APP] Sent calibration preview ({len(jpg_bytes)} bytes)")

                except Exception as exc:
                    print(f"[APP] Calibration preview failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Calibration preview failed: {exc}'
                    }))

                finally:
                    tc.clean_up()
                    camera.stop()

                continue

            elif event == 'crop_calibration_preview':
                global latest_crop_preview_pair_midpoint
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot preview crop calibration while capture is running'
                    }))
                    continue
                if focus_task and not focus_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot preview crop calibration while focus is active'
                    }))
                    continue

                preview_width = max(1, int(data.get('preview_width', 800)))

                try:
                    tc.light_on()
                    apply_capture_camera_controls()
                    camera.start()
                    print('[APP] LED on + camera for crop calibration preview')
                    await asyncio.sleep(0.5)

                    crop_fast_detector = FastSprocketDetector(
                        reference_size=CALIBRATION_RES,
                        expected_pitch=SPROCKET_PITCH_PX,
                    )
                    frame_bgr = None
                    sprockets = []
                    registration_y = None
                    registration_mode = 'none'
                    detection_method = 'failed'
                    full_count = 0
                    partial_count = 0

                    for attempt in range(1, 6):
                        buffer = io.BytesIO()
                        camera.capture_file(buffer, format='jpeg')
                        candidate_frame = cv2.imdecode(
                            np.frombuffer(buffer.getvalue(), np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if candidate_frame is None:
                            await asyncio.sleep(0.05)
                            continue

                        frame_bgr = candidate_frame
                        sprockets = crop_fast_detector.detect(frame_bgr) or []
                        detection_method = 'fast'
                        if not sprockets:
                            sprockets = detector.detect(frame_bgr, mode='profile') or []
                            detection_method = 'fallback' if sprockets else 'failed'

                        classified = detector.classify_sprockets(sprockets, frame_bgr.shape)
                        full_sprockets = [
                            item['sprocket'] for item in classified
                            if item.get('status') == 'full'
                        ]
                        full_count = len(full_sprockets)
                        partial_count = sum(
                            1 for item in classified
                            if item.get('status') == 'partial'
                        )
                        registration_choice = detector.choose_registration(
                            full_sprockets,
                            frame_bgr.shape,
                            expected_pitch=SPROCKET_PITCH_PX,
                        ) or {}
                        if registration_choice.get('mode') == 'pair':
                            registration_y = registration_choice.get('actual_y')
                            registration_mode = 'pair'
                            break
                        await asyncio.sleep(0.05)

                    if frame_bgr is None:
                        raise RuntimeError('Failed to decode crop calibration preview frame')
                    print(
                        f"[APP] Crop preview registration: mode={registration_mode}, "
                        f"method={detection_method}, full={full_count}, "
                        f"partial={partial_count}, y={registration_y}"
                    )
                    latest_crop_preview_pair_midpoint = registration_y if registration_mode == 'pair' else None

                    preview_frame = frame_bgr.copy()
                    frame_height = preview_frame.shape[0]
                    if registration_y is not None:
                        # Draw in raw coordinates. The whole preview is flipped
                        # below, which moves this annotation into display space.
                        cv2.line(
                            preview_frame,
                            (0, int(round(registration_y))),
                            (preview_frame.shape[1] - 1, int(round(registration_y))),
                            (0, 0, 255),
                            2,
                        )

                    existing_crop = get_effective_crop_settings()
                    existing_crop_preview_rect = None
                    if existing_crop is not None and registration_y is not None:
                        x1, y1, x2, y2 = get_relative_crop_rect(frame_bgr, registration_y)
                        flipped_y1 = int(frame_height - y2)
                        flipped_y2 = int(frame_height - y1)
                        # As with the registration line, draw once in raw space;
                        # cv2.flip below performs the only vertical transform.
                        cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    # Preview is vertically flipped to match capture output; coordinates remain in original frame space
                    preview_frame = cv2.flip(preview_frame, 0)

                    preview_height = max(1, int(preview_frame.shape[0] * (preview_width / preview_frame.shape[1])))
                    if existing_crop is not None and registration_y is not None:
                        scale_x = preview_width / float(frame_bgr.shape[1])
                        scale_y = preview_height / float(frame_bgr.shape[0])
                        existing_crop_preview_rect = [
                            float(x1) * scale_x,
                            float(flipped_y1) * scale_y,
                            float(x2) * scale_x,
                            float(flipped_y2) * scale_y,
                        ]
                    preview_frame = cv2.resize(
                        preview_frame,
                        (preview_width, preview_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    ok, encoded = cv2.imencode('.jpg', preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if not ok:
                        raise RuntimeError('Failed to encode crop calibration preview')

                    jpg_bytes = encoded.tobytes()
                    await websocket.send(json.dumps({
                        'event': 'crop_calibration_preview',
                        'full_width': int(frame_bgr.shape[1]),
                        'full_height': int(frame_bgr.shape[0]),
                        'preview_width': int(preview_width),
                        'preview_height': int(preview_height),
                        'display_flipped_vertical': True,
                        'registration_y': registration_y,
                        'registration_mode': registration_mode,
                        'detection_method': detection_method,
                        'full_sprocket_count': full_count,
                        'partial_sprocket_count': partial_count,
                        'existing_crop': existing_crop_preview_rect,
                        'existing_crop_definition': existing_crop,
                        'size': len(jpg_bytes)
                    }))
                    await websocket.send(jpg_bytes)

                except Exception as exc:
                    print(f'[APP] Crop calibration preview failed: {exc}')
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Crop calibration preview failed: {exc}'
                    }))

                finally:
                    tc.clean_up()
                    camera.stop()

                continue

            elif event == 'crop_calibration_save':
                global calibrated_baseline_registration_y
                rect = data.get('rect')
                preview_rect = data.get('preview_rect')
                registration_y = data.get('registration_y')
                if registration_y is None:
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Invalid crop calibration payload'
                    }))
                    continue

                try:
                    frame_w, frame_h = CALIBRATION_RES

                    if isinstance(preview_rect, list) and len(preview_rect) == 4:
                        preview_width = int(data.get('preview_width', 0))
                        preview_height = int(data.get('preview_height', 0))
                        if preview_width <= 0 or preview_height <= 0:
                            raise ValueError('Missing preview dimensions for crop calibration save')

                        display_flipped_vertical = bool(data.get('display_flipped_vertical', False))
                        scale_x = float(frame_w) / float(preview_width)
                        scale_y = float(frame_h) / float(preview_height)

                        preview_x1 = float(preview_rect[0])
                        preview_y1 = float(preview_rect[1])
                        preview_x2 = float(preview_rect[2])
                        preview_y2 = float(preview_rect[3])

                        raw_x1 = preview_x1 * scale_x
                        raw_x2 = preview_x2 * scale_x
                        if display_flipped_vertical:
                            raw_y1 = float(frame_h) - (preview_y2 * scale_y)
                            raw_y2 = float(frame_h) - (preview_y1 * scale_y)
                        else:
                            raw_y1 = preview_y1 * scale_y
                            raw_y2 = preview_y2 * scale_y

                        x1_raw = min(raw_x1, raw_x2)
                        x2_raw = max(raw_x1, raw_x2)
                        y1_raw = min(raw_y1, raw_y2)
                        y2_raw = max(raw_y1, raw_y2)

                        x1 = max(0, min(frame_w - 1, int(round(x1_raw))))
                        y1 = max(0, min(frame_h - 1, int(round(y1_raw))))
                        x2 = max(x1 + 1, min(frame_w, int(round(x2_raw))))
                        y2 = max(y1 + 1, min(frame_h, int(round(y2_raw))))
                    elif isinstance(rect, list) and len(rect) == 4:
                        x1 = max(0, min(frame_w - 1, int(rect[0])))
                        y1 = max(0, min(frame_h - 1, int(rect[1])))
                        x2 = max(x1 + 1, min(frame_w, int(rect[2])))
                        y2 = max(y1 + 1, min(frame_h, int(rect[3])))
                    else:
                        raise ValueError('Missing crop rectangle for crop calibration save')

                    registration_y = float(registration_y)

                    crop_data = save_crop_settings((x1, y1, x2, y2), registration_y, (frame_w, frame_h))
                    if latest_crop_preview_pair_midpoint is not None:
                        calibrated_baseline_registration_y = float(latest_crop_preview_pair_midpoint)
                        reset_registration_tracking(
                            "crop_calibration_saved",
                            baseline_registration_y=calibrated_baseline_registration_y,
                        )
                    refresh_runtime_settings()

                    await websocket.send(json.dumps({
                        'event': 'crop_calibration_saved',
                        'crop': crop_data,
                        'rect': (
                            [float(value) for value in preview_rect]
                            if isinstance(preview_rect, list) and len(preview_rect) == 4
                            else [int(x1), int(y1), int(x2), int(y2)]
                        ),
                    }))
                except Exception as exc:
                    print(f'[APP] Crop calibration save failed: {exc}')
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Crop calibration save failed: {exc}'
                    }))
                continue

            elif event == 'calibration_sweep':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot run calibration sweep while capture is running'
                    }))
                    continue
                if focus_task and not focus_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot run calibration sweep while focus is active'
                    }))
                    continue

                total_samples = max(1, int(data.get('samples', 10)))
                step_size = int(data.get('step_size', 25))
                seek_step_size = int(data.get('seek_step_size', 10))
                debug_scale = float(data.get('debug_scale', 1.0))
                sweep_samples = []
                motor_runs = []
                latest_proposed_calibration = None
                latest_can_save = False
                latest_save_block_reason = 'calibration_sweep_in_progress'

                try:
                    print(f"[APP] Starting calibration sweep: samples={total_samples}, step_size={step_size}, debug_scale={debug_scale}")
                    progress_total = total_samples + 7
                    await websocket.send(json.dumps({
                        'event': 'calibration_sweep_progress', 'phase': 'starting',
                        'completed': 0, 'total': progress_total,
                        'message': 'Starting camera and stabilizing light…'
                    }))
                    tc.light_on()
                    camera.start()
                    print("[APP] LED on + camera, stabilizing for calibration sweep...")
                    await asyncio.sleep(0.5)
                    exposure_result = await prepare_calibration_camera_settings(detector)
                    await websocket.send(json.dumps({
                        'event': 'calibration_sweep_progress', 'phase': 'seeking',
                        'completed': 1, 'total': progress_total,
                        'message': 'Exposure set; finding two complete sprockets…'
                    }))
                    seek_result = await seek_two_full_sprockets(
                        camera,
                        tc,
                        detector,
                        step_size=seek_step_size,
                        max_steps=500,
                        settle_delay=0.05,
                    )
                    if not seek_result.get('valid'):
                        await websocket.send(json.dumps({
                            'event': 'calibration_error',
                            'message': 'Could not find two full sprockets before calibration',
                            'seek': seek_result
                        }))
                        continue

                    await websocket.send(json.dumps({
                        'event': 'info',
                        'message': 'Found two full sprockets; starting calibration sweep'
                    }))

                    for sample_index in range(total_samples):
                        measurement, jpg_bytes = calibrator.capture_sprocket_preview(debug_scale)
                        sample_measurement = dict(measurement)
                        sample_measurement['sample'] = sample_index
                        sample_measurement['exposure_time'] = exposure_result['exposure_time']
                        sample_measurement['gain'] = exposure_result['gain']
                        sweep_samples.append(sample_measurement)

                        await websocket.send(json.dumps({
                            'event': 'calibration_sweep_sample',
                            **sample_measurement,
                            'size': len(jpg_bytes)
                        }))
                        await websocket.send(jpg_bytes)
                        await websocket.send(json.dumps({
                            'event': 'calibration_sweep_progress', 'phase': 'samples',
                            'completed': sample_index + 2, 'total': progress_total,
                            'message': f'Measured sprocket sample {sample_index + 1} of {total_samples}'
                        }))
                        print(
                            f"[APP] Calibration sweep sample {sample_index + 1}/{total_samples}: "
                            f"pitch={sample_measurement.get('sprocket_pitch_px')} "
                            f"area={sample_measurement.get('sprocket_area_nominal')}"
                        )

                        if sample_index < total_samples - 1:
                            tc.steps_forward(step_size)
                            print(f"[APP] Calibration sweep moved forward {step_size} steps")
                            await asyncio.sleep(0.15)

                    summary = compute_calibration_summary(sweep_samples)
                    trusted_pitch_for_motor = summary.get('pitch_mean')
                    if trusted_pitch_for_motor is None:
                        trusted_pitch_for_motor = settings.get('sprocket_pitch_px', 814)

                    motor_total_runs = 5
                    for motor_run_index in range(motor_total_runs):
                        await websocket.send(json.dumps({
                            'event': 'calibration_sweep_progress', 'phase': 'motor',
                            'completed': total_samples + 2 + motor_run_index,
                            'total': progress_total,
                            'message': f'Measuring transport run {motor_run_index + 1} of {motor_total_runs}…'
                        }))
                        motor_result = await measure_steps_per_pitch_live(
                            camera,
                            tc,
                            detector,
                            trusted_pitch_for_motor,
                            step_chunk=20,
                            max_steps=500,
                        )
                        motor_runs.append(motor_result)
                        print(f"[APP] Motor calibration run {motor_run_index + 1}/{motor_total_runs}: {motor_result}")
                        if motor_run_index < motor_total_runs - 1:
                            tc.steps_forward(30)
                            print("[APP] Motor calibration offset move: 30 steps")
                            await asyncio.sleep(0.05)

                    valid_motor_steps = [
                        float(result['steps_per_pitch'])
                        for result in motor_runs
                        if result.get('valid') and result.get('steps_per_pitch') is not None
                    ]
                    filtered_motor_steps = filter_robust_values(valid_motor_steps)
                    motor_valid_runs = len(valid_motor_steps)
                    motor_updated = len(filtered_motor_steps) >= 3
                    motor_steps_per_pitch = None
                    if motor_updated:
                        motor_steps_per_pitch = int(round(np.median(np.array(filtered_motor_steps, dtype=float))))

                    motor_calibration = {
                        'motor_steps_per_pitch': motor_steps_per_pitch,
                        'motor_valid_runs': motor_valid_runs,
                        'motor_total_runs': motor_total_runs,
                        'motor_updated': motor_updated,
                    }

                    proposed_calibration, can_save, save_block_reason = build_proposed_calibration(
                        sweep_samples,
                        exposure_result,
                        motor_calibration,
                    )
                    latest_proposed_calibration = proposed_calibration
                    latest_can_save = can_save
                    latest_save_block_reason = save_block_reason
                    summary['seek'] = seek_result
                    summary['exposure'] = exposure_result
                    summary['motor_steps_per_pitch'] = motor_steps_per_pitch
                    summary['motor_valid_runs'] = motor_valid_runs
                    summary['motor_total_runs'] = motor_total_runs
                    summary['motor_updated'] = motor_updated
                    print(f"[APP] Calibration sweep complete: {summary}")
                    await websocket.send(json.dumps({
                        'event': 'calibration_sweep_complete',
                        'summary': summary,
                        'proposed_calibration': proposed_calibration,
                        'can_save': can_save,
                        'save_block_reason': save_block_reason
                    }))

                except Exception as exc:
                    print(f"[APP] Calibration sweep failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Calibration sweep failed: {exc}'
                    }))

                finally:
                    tc.clean_up()
                    camera.stop()

                continue

            elif event == 'calibration_save':
                if not latest_can_save or not latest_proposed_calibration:
                    reason = latest_save_block_reason or 'no_proposed_calibration_available'
                    print(f"[APP] Calibration save blocked: {reason}")
                    await websocket.send(json.dumps({
                        'event': 'calibration_save_blocked',
                        'reason': reason
                    }))
                    continue

                try:
                    backup_path = calibrator.save_calibration(latest_proposed_calibration)
                    refresh_runtime_settings()
                    apply_capture_camera_controls()
                    print(f"[APP] Calibration saved to calibration.json (backup={backup_path})")
                    await websocket.send(json.dumps({
                        'event': 'calibration_saved',
                        'message': 'Calibration saved and runtime settings refreshed',
                        'settings': {
                            'exposure_time': EXPOSURE_TIME,
                            'gain': GAIN,
                            'sprocket_pitch_px': SPROCKET_PITCH_PX,
                            'steps_per_pitch': STEPS_PER_PITCH,
                            'steps_per_px': steps_per_px,
                            'sprocket_area_min': detector.min_area,
                            'sprocket_area_max': detector.max_area
                        },
                        'backup_path': backup_path
                    }))
                except Exception as exc:
                    print(f"[APP] Calibration save failed: {exc}")
                    await websocket.send(json.dumps({
                        'event': 'calibration_error',
                        'message': f'Calibration save failed: {exc}'
                    }))
                continue

            elif event == 'jog_forward' or event == 'jog_back':
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot jog while capture is running'
                    }))
                    continue
                frames = int(data.get("frames", 1))
                direction = 1 if event == "jog_forward" else -1
                tc.light_on()
                apply_capture_camera_controls()
                camera.start()
                print("[APP] LED on + camera, stabilizing...")

                steps_per_pitch = STEPS_PER_PITCH
                total_steps = max(0, frames) * steps_per_pitch
                if direction > 0:
                    print(f"[APP] Jogging forward {frames} frames ({total_steps} steps)")
                    tc.steps_forward(total_steps)
                else:
                    print(f"[APP] Jogging backward {frames} frames ({total_steps} steps)")
                    tc.steps_back(total_steps)
                    #tc.rewind()

                # Capture image immediately after jogging; manual alignment happens via next_step/prev_step.
                buffer = io.BytesIO()
                camera.capture_file(buffer, format='jpeg')
                frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

                # Detect sprockets + crop
                sprockets = detector.detect(frame_bgr, mode="profile")
                registration_y = get_registration_y(frame_bgr, sprockets if sprockets else [])
                debug_frame = draw_sprockets_debug(frame_bgr, sprockets if sprockets else [], registration_y=registration_y)
                anchor = sprockets[0] if sprockets else None
                frame_cropped = crop_film_frame(frame_bgr, anchor, SPROCKET_PITCH_PX)

                # Encode cropped image
                _, cropped_bytes = await encode_frame_async(frame_cropped, 1)
                _, debug_bytes = await encode_frame_async(debug_frame, 1)
                header = len(cropped_bytes).to_bytes(4,'big') # 4-byte big-endian int
                payload = header + cropped_bytes + debug_bytes
                await websocket.send(payload)

                await websocket.send(json.dumps({
                    "event": "info",
                    "message": f"Jogged {'forward' if direction>0 else 'back'} {frames} frames"
                }))
                tc.clean_up()
                camera.stop()

            elif event == 'focus_start':
                if focus_task and not focus_task.done():
                    await websocket.send(json.dumps({
                        'event': 'info',
                        'message': 'Focus already active'
                    }))
                    continue
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot start focus during capture'
                    }))
                    continue
                if camera_calibration_task and not camera_calibration_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot start focus while camera calibration preview is active'
                    }))
                    continue
                focus_stop_event = asyncio.Event()
                preview_width = data.get('preview_width', 800)
                fps = data.get('fps', 5)
                focus_task = asyncio.create_task(
                    run_focus(websocket, focus_stop_event, preview_width=preview_width, fps=fps)
                )
                continue

            elif event == "troubleshoot_start":
                if capture_task and not capture_task.done():
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': 'Cannot troubleshoot while capture is running'
                    }))
                    continue
                await troubleshoot_sprocket_detection(camera, websocket, tc, detector)
                continue

            elif event == "focus_stop":
                if focus_task and not focus_task.done():
                    focus_stop_event.set()
                    try:
                        await focus_task
                    finally:
                        focus_task = None
                        focus_stop_event = None
                else:
                    await websocket.send(json.dumps({
                        'event': 'info',
                        'message': 'Focus not active'
                    }))

            else:
                print(f"[APP] Unrecognized event: {event}")
    except ConnectionClosedOK:
        print("[APP] Client connection closed cleanly")
    except ConnectionClosedError as exc:
        print(f"[APP] Client disconnected unexpectedly: {exc}")
    except Exception as exc:
        print(f"[APP] Unexpected client handler error: {exc}")
        raise
    finally:
        if capture_task and not capture_task.done():
            print("[APP] Cleaning up capture task after disconnect")
            capture_stop_event.set()
            try:
                await capture_task
            except Exception as exc:
                print(f"[APP] Capture task cleanup error: {exc}")
            finally:
                capture_task = None
                capture_stop_event = None
        if focus_task and not focus_task.done():
            print("[APP] Cleaning up focus task after disconnect")
            focus_stop_event.set()
            try:
                await focus_task
            except Exception as exc:
                print(f"[APP] Focus task cleanup error: {exc}")
            finally:
                focus_task = None
                focus_stop_event = None
        if camera_calibration_task and not camera_calibration_task.done():
            print("[APP] Cleaning up camera calibration preview task after disconnect")
            camera_calibration_stop_event.set()
            try:
                await camera_calibration_task
            except Exception as exc:
                print(f"[APP] Camera calibration preview cleanup error: {exc}")
            finally:
                camera_calibration_task = None
                camera_calibration_stop_event = None

async def main():
    print("Starting WebSocket server on ws://0.0.0.0:5000")
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
         5000,
         ping_interval=None
    )
    print("Server started")
    await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        print("Cleaning up")
        tc.clean_up()
        camera.stop()
