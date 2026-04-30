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
from sprocket import SprocketDetector
from calibration_service import CalibrationService
import socket
import os
from collections import deque

async def troubleshoot_sprocket_detection(camera, websocket, tc, detector,
                                          step_size=10, delay=0.05):
    """
    Bi-directional sprocket troubleshooting loop.
    The client can send:
      - {"event": "next_step"} → move + capture + send debug frame
      - {"event": "stop_troubleshoot"} → exit loop
    """
    print("[TROUBLE] Entering interactive sprocket troubleshooting mode")

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

        if evt == "next_step":
            frame_counter += 1
            print(f"[TROUBLE] Step {frame_counter}: moving {step_size} steps")
            tc.steps_forward(step_size)
            await asyncio.sleep(delay)

            # --- capture & detect ---
            buffer = io.BytesIO()
            camera.capture_file(buffer, format="jpeg")
            frame = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            sprockets = detector.detect(frame, mode="profile")

            print(f"[TROUBLE] Detected {len(sprockets)} sprockets")

            # --- draw debug overlay ---
            dbg = frame.copy()
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
                    "sprocket_count": len(sprockets)
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

def draw_sprockets_debug(frame, sprockets):
    """
    Draw bounding boxes, centers, and labels for detected sprockets.
    Always returns a valid flipped debug frame.
    """
    debug_frame = frame.copy()

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


def get_registration_y(frame_bgr, sprockets):
    registration = detector.choose_registration(
        sprockets,
        frame_bgr.shape,
        expected_pitch=SPROCKET_PITCH_PX,
    )
    if registration.get('mode') in ('pair', 'single') and registration.get('actual_y') is not None:
        return float(registration['actual_y'])
    return None


def get_capture_registration(frame_bgr, sprockets):
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
    }


def get_relative_crop_rect(frame_bgr, registration_y):
    frame_h, frame_w = frame_bgr.shape[:2]
    crop_settings = settings.get('crop')
    if not isinstance(crop_settings, dict) or registration_y is None:
        return 0, 0, frame_w, frame_h

    try:
        x1 = int(crop_settings['x1'])
        x2 = int(crop_settings['x2'])
        y_offset = float(crop_settings['y_offset'])
        height = int(crop_settings['height'])
    except (KeyError, TypeError, ValueError):
        return 0, 0, frame_w, frame_h

    if height <= 0:
        return 0, 0, frame_w, frame_h

    x1 = max(0, min(frame_w - 1, x1))
    x2 = max(x1 + 1, min(frame_w, x2))

    y1 = int(round(float(registration_y) + y_offset))
    y2 = y1 + height

    if y1 < 0:
        y2 = min(frame_h, y2 - y1)
        y1 = 0
    if y2 > frame_h:
        y1 = max(0, y1 - (y2 - frame_h))
        y2 = frame_h

    if y2 <= y1:
        return 0, 0, frame_w, frame_h

    return int(x1), int(y1), int(x2), int(y2)


def crop_frame_relative_to_registration(frame_bgr, registration_y):
    x1, y1, x2, y2 = get_relative_crop_rect(frame_bgr, registration_y)
    return frame_bgr[y1:y2, x1:x2]


def save_crop_settings(rect, registration_y, frame_size):
    frame_w, frame_h = frame_size
    x1, y1, x2, y2 = rect
    crop_data = {
        'reference': 'registration_y',
        'x1': int(x1),
        'x2': int(x2),
        'y_offset': float(y1 - registration_y),
        'height': int(y2 - y1),
        'source_resolution': [int(frame_w), int(frame_h)],
    }

    config_path = 'config.json'
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as handle:
            config_data = json.load(handle)

    config_data['crop'] = crop_data

    with open(config_path, 'w', encoding='utf-8') as handle:
        json.dump(config_data, handle, indent=2)
        handle.write('\n')

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
    merged = {**calib, **config}
    return merged


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

    print("[APP] Runtime calibration settings refreshed")
    if previous_resolution is not None and tuple(previous_resolution) != CALIBRATION_RES:
        print("[APP] Calibration resolution changed; service restart recommended to reconfigure camera")

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

def apply_capture_camera_controls():
    print("[APP] Switching camera to capture controls")
    camera.set_controls({
        "ExposureTime": EXPOSURE_TIME,
        "AnalogueGain": GAIN,
        "AeEnable": False,
        "AwbEnable": False,
    })


def apply_focus_camera_controls():
    print("[APP] Switching camera to focus controls")
    camera.set_controls({
        "AeEnable": True,
        "AwbEnable": False,
    })


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
    ar_min=1.2, ar_max=1.8,
    solidity_min=0.75,
    blur=5, open_k=5, close_k=3,
    adaptive_block=41, adaptive_C=7,
    method="profile"
)

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


SAVE_DIR = "/media/williatf/SG1TB/GrokCam/testframes"
os.makedirs(SAVE_DIR, exist_ok=True)

async def run_capture(websocket, num_frames, stop_event, preview_width=800, debug_scale=1.0):
    print("[APP] Capture task starting")
    tc.light_on()
    apply_capture_camera_controls()
    camera.start()
    print("[APP] LED on + camera, stabilizing...")
    try:
        await asyncio.sleep(2)

        nominal_steps_per_pitch = int(settings.get("steps_per_pitch", STEPS_PER_PITCH))
        current_steps = int(settings.get("steps_per_pitch", 280))
        calibrated_steps_per_px = float(settings.get("steps_per_px", steps_per_px))
        min_steps = int(nominal_steps_per_pitch * 0.88)
        max_steps = int(nominal_steps_per_pitch * 1.12)
        correction_gain = 0.18
        max_correction_steps = 6
        target_y = None

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
            debug_frame = draw_sprockets_debug(frame_bgr, sprockets if sprockets else [])
            capture_registration = get_capture_registration(frame_bgr, sprockets if sprockets else [])
            registration_y = capture_registration.get('registration_y')
            registration_mode = capture_registration.get('mode', 'none')

            if target_y is None:
                target_y = frame_bgr.shape[0] / 2.0

            if registration_y is None:
                print(
                    f"[APP] Frame {frame}: registration_y unavailable, keeping steps={current_steps}"
                )
            else:
                steps_before_update = current_steps
                error_px = float(target_y) - float(registration_y)
                update_allowed = registration_mode == 'pair'
                if registration_mode == 'single' and abs(error_px) <= 80.0:
                    update_allowed = True

                correction = int(round(error_px * calibrated_steps_per_px * correction_gain))
                correction = max(-max_correction_steps, min(correction, max_correction_steps))
                next_steps = current_steps

                if update_allowed:
                    next_steps = steps_before_update + correction
                    next_steps = max(min_steps, min(max_steps, next_steps))
                    current_steps = next_steps
                else:
                    print(f"[APP] Frame {frame}: ignoring low-confidence registration for step update")

                print(
                    f"[APP] Frame {frame}: reg={registration_y:.1f}, mode={registration_mode}, "
                    f"err={error_px:+.1f}px, steps={steps_before_update}, "
                    f"correction={correction:+d}, next={next_steps}"
                )

            frame_cropped = crop_frame_relative_to_registration(frame_bgr, registration_y)
            # Flip vertically to correct camera orientation (matches legacy behavior)
            frame_cropped = cv2.flip(frame_cropped, 0)

            timestamp = int(time.time() * 1000)
            filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.png")
            save_ok = cv2.imwrite(filename, frame_cropped)  # PNG = lossless
            if not save_ok:
                print(f"[APP] WARNING: Failed to save cropped frame to {filename}")
                await websocket.send(json.dumps({
                    'event': 'warning',
                    'message': f'Failed to save frame {frame} to disk'
                }))

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

async def run_focus(websocket, stop_event, preview_width=800, fps=5):
    print("[APP] Focus task starting")
    tc.light_on()
    camera.start()
    apply_focus_camera_controls()
    print("[APP] LED on + camera for focus")

    frame_num = 0
    preview_width = max(1, int(preview_width))
    frame_delay = 1.0 / max(1.0, float(fps))

    try:
        await asyncio.sleep(1.0)
        await websocket.send(json.dumps({
            'event': 'focus_started'
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
                'event': 'focus_frame',
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
                'event': 'focus_stopped'
            }))
        except (ConnectionClosedError, ConnectionClosedOK):
            print("[APP] Focus stop notification skipped: client disconnected")
        except Exception as exc:
            print(f"[APP] Focus stop notification error: {exc}")
        print("[APP] Focus task cleaned up")

async def handle_client(websocket):
    print("Client connected")
    capture_task = None
    capture_stop_event = None
    focus_task = None
    focus_stop_event = None
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

            event = data.get('event')

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
                num_frames = data.get('num_frames', 100)
                preview_width = data.get('preview_width', 800)
                debug_scale = data.get('debug_scale', 1.0)
                capture_stop_event = asyncio.Event()
                capture_task = asyncio.create_task(
                    run_capture(
                        websocket,
                        num_frames,
                        capture_stop_event,
                        preview_width=preview_width,
                        debug_scale=debug_scale
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
                    exposure_result = await tune_calibration_exposure(camera, detector)

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

                    buffer = io.BytesIO()
                    camera.capture_file(buffer, format='jpeg')
                    frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        raise RuntimeError('Failed to decode crop calibration preview frame')

                    sprockets = detector.detect(frame_bgr, mode='profile') or []
                    registration_y = get_registration_y(frame_bgr, sprockets)

                    preview_frame = frame_bgr.copy()
                    frame_height = preview_frame.shape[0]
                    if registration_y is not None:
                        flipped_y = int(round(frame_height - registration_y))
                        cv2.line(preview_frame, (0, flipped_y), (preview_frame.shape[1] - 1, flipped_y), (0, 0, 255), 2)

                    existing_crop = settings.get('crop') if isinstance(settings.get('crop'), dict) else None
                    if existing_crop is not None and registration_y is not None:
                        x1, y1, x2, y2 = get_relative_crop_rect(frame_bgr, registration_y)
                        flipped_y1 = int(frame_height - y2)
                        flipped_y2 = int(frame_height - y1)
                        cv2.rectangle(preview_frame, (x1, flipped_y1), (x2, flipped_y2), (255, 255, 0), 2)

                    # Preview is vertically flipped to match capture output; coordinates remain in original frame space
                    preview_frame = cv2.flip(preview_frame, 0)

                    preview_height = max(1, int(preview_frame.shape[0] * (preview_width / preview_frame.shape[1])))
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
                        'existing_crop': existing_crop,
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
                    refresh_runtime_settings()

                    await websocket.send(json.dumps({
                        'event': 'crop_calibration_saved',
                        'crop': crop_data
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
                    tc.light_on()
                    camera.start()
                    print("[APP] LED on + camera, stabilizing for calibration sweep...")
                    await asyncio.sleep(0.5)
                    exposure_result = await tune_calibration_exposure(camera, detector)
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
                    print(f"[APP] Jogging forward {frames} frames ({total_steps} steps) before alignment")
                    tc.steps_forward(total_steps)
                else:
                    print(f"[APP] Jogging backward {frames} frames ({total_steps} steps) before alignment")
                    tc.steps_back(total_steps)
                    #tc.rewind()

                # Capture image after jogging
                anchor = await advance_to_next_perforation(camera, websocket,
                    steps_per_pitch = settings.get("steps_per_pitch", 280), 
                    steps_per_px = settings.get("steps_per_px", 0.5))
                buffer = io.BytesIO()
                camera.capture_file(buffer, format='jpeg')
                frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

                # Detect sprockets + crop
                sprockets = detector.detect(frame_bgr, mode="profile")
                debug_frame = draw_sprockets_debug(frame_bgr, sprockets if sprockets else [])
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
