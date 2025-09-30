import asyncio
import websockets
import picamera2
import cv2
import numpy as np
import io
import time
import base64
import json
from control import tcControl
from sprocket import SprocketDetector
import socket
import os

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

    crop_h = int(pitch_px * 1.2)
    offset = int(0.1 * crop_h)

    y1 = max(0, cy - offset)
    y2 = min(H, y1 + crop_h)

    # If bottom is clipped, shift up
    if y2 - y1 < crop_h and y1 > 0:
        y1 = max(0, y2 - crop_h)

    return frame[y1:y2, 0:W]

def encode_frame(buffer, frame_num):
    """Heavy CPU work (base64 + json.dumps) offloaded to threadpool"""
    img_data = buffer.read()
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    return json.dumps({
        "event": "new_image",
        "frame": frame_num,
        "image": img_base64
    })

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

# Use calibrated exposure/gain
camera.set_controls({
    "ExposureTime": EXPOSURE_TIME,
    "AnalogueGain": GAIN,
    "AeEnable": False,
    "AwbEnable": False,
})

detector = SprocketDetector(
    side="left", auto_roi=0.40,
    min_area=1500, max_area=25000,
    ar_min=1.2, ar_max=1.8,
    solidity_min=0.75,
    blur=5, open_k=5, close_k=3,
    adaptive_block=41, adaptive_C=7,
    method="profile"
)

async def advance_to_next_perforation(camera, websocket, steps_per_pitch=STEPS_PER_PITCH, first_frame=False):
    target_y = 246
    tolerance = 30
    max_steps = 2000
    steps_taken = 0

    # --- First frame logic ---
    if first_frame:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

        sprockets = detector.detect(lores_bgr, mode="profile")
        if sprockets:
            sprockets.sort(key=lambda s: abs(s[1] - target_y))
            _, cy, _, _, _ = sprockets[0]
            while cy < target_y:
                tc.steps_forward(10)
                steps_taken += 10
                await asyncio.sleep(0.01)
                buffer = io.BytesIO()
                camera.capture_file(buffer, format='jpeg')
                lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                sprockets = detector.detect(lores_bgr, mode="profile")
                if not sprockets:
                    break
                sprockets.sort(key=lambda s: abs(s[1] - target_y))
                _, cy, _, _, _ = sprockets[0]
        else:
            tc.steps_forward(steps_per_pitch)
            steps_taken += steps_per_pitch
            await asyncio.sleep(0.05)

    else:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        sprockets = detector.detect(lores_bgr, mode="profile")

        if sprockets:
            sprockets.sort(key=lambda s: abs(s[1] - target_y))
            _, cy, _, _, _ = sprockets[0]
            error = target_y - cy
            correction = int(error * steps_per_px)
            coarse_steps = steps_per_pitch + correction
            coarse_steps = max(steps_per_pitch // 2, min(steps_per_pitch * 2, coarse_steps))
        else:
            coarse_steps = steps_per_pitch

        tc.steps_forward(coarse_steps)
        steps_taken += coarse_steps
        await asyncio.sleep(0.05)

    # --- Fine alignment ---
    while steps_taken < max_steps:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

        sprockets = detector.detect(lores_bgr, mode="profile")
        if not sprockets:
            tc.steps_forward(20)
            steps_taken += 20
            await websocket.send(json.dumps({
                'event': 'warning',
                'message': f'No sprocket detected at step {steps_taken}'
            }))
            continue

        sprockets.sort(key=lambda s: abs(s[1] - target_y))
        cx, cy, w, h, area = sprockets[0]
        print(f"[APP] Anchor sprocket at (cx={cx}, cy={cy})")

        if abs(cy - target_y) < tolerance:
            print(f"[APP] Alignment success: cy={cy}")
            return (cx, cy)

        error = target_y - cy
        step_size = int(error * steps_per_px)

        # Clamp step size to reasonable range
        if abs(step_size) > 20:
            step_size = 20 if step_size > 0 else -20
        elif abs(step_size) < 2:
            step_size = 2 if step_size > 0 else -2

        if step_size > 0:
            tc.steps_forward(step_size)
        else:
            tc.steps_back(abs(step_size))
        await asyncio.sleep(0.01)

    print("[APP] Alignment failed")
    return None

async def handle_client(websocket):
    print("Client connected")
    stop_requested = False
    async for message in websocket:
        print(f"[APP] Got message: {message}")
        data = json.loads(message)

        if data.get('event') == 'start_capture':
            num_frames = data.get('num_frames', 100)
            tc.light_on()
            camera.start()
            print("[APP] LED on + camera, stabilizing...")
            await asyncio.sleep(2)

            for frame in range(num_frames):
                if stop_requested:
                    break
                anchor = await advance_to_next_perforation(
                    camera, websocket, first_frame=(frame == 0)
                )
                if not anchor:
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Failed to align frame {frame}'
                    }))
                    break

                # Capture full frame
                buffer = io.BytesIO()
                camera.capture_file(buffer, format='jpeg')
                buffer.seek(0)
                frame_bgr = cv2.imdecode(
                    np.frombuffer(buffer.getvalue(), np.uint8),
                    cv2.IMREAD_COLOR
                )

                # Crop actual film frame relative to sprocket anchor
                frame_cropped = crop_film_frame(
                    frame_bgr, anchor, settings.get("frame_crop_offsets")
                )

                # Re-encode cropped frame to JPEG
                _, encoded = cv2.imencode('.jpg', frame_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                img_base64 = base64.b64encode(encoded).decode('utf-8')

                await websocket.send(json.dumps({
                    'event': 'new_image',
                    'frame': frame,
                    'image': img_base64
                }))

                print(f"[APP] Sent frame {frame}")

                await asyncio.sleep(0.5)

            await websocket.send(json.dumps({'event': 'capture_complete'}))
            tc.clean_up()
            camera.stop()

        if data.get('event') == 'stop_capture':
            print("[APP] Stop requested")
            stop_requested = True
            await websocket.send(json.dumps({
                'event': 'info',
                'message': 'Stop requested'
            }))
            continue

async def main():
    print("Starting WebSocket server on ws://0.0.0.0:5000")
    server = await websockets.serve(
        handle_client,
        "::",
         5000,
         ping_interval=30,
         ping_timeout=30
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
