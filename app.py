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

def crop_film_frame(frame, anchor, film_frame_crop):
    """Crop the film frame region relative to sprocket anchor (cx, cy)."""
    if not film_frame_crop or not anchor:
        return frame  # fallback to full frame

    cx, cy = int(anchor[0]), int(anchor[1])
    dx1, dy1, dx2, dy2 = film_frame_crop
    x1, y1 = cx + dx1, cy + dy1
    x2, y2 = cx + dx2, cy + dy2

    # Clamp to frame bounds
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    return frame[y1:y2, x1:x2]

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

# --- Initialize transport ---
print("Starting WebSocket server")
tc = tcControl()
print("tcControl initialized")

# --- Initialize camera ---
camera = picamera2.Picamera2()
print("Initializing camera")

# Match calibration resolution (1920x1080)
config_main = camera.create_still_configuration(main={"size": (1920, 1080)})
camera.configure(config_main)
camera.options['quality'] = 90

SPROCKET_WIDTH = settings.get("sprocket_width_px", 72)   # default if missing
SPROCKET_PITCH = settings.get("sprocket_pitch_px", 148)  # default if missing

# Example: nominal sprocket area (px^2) for filtering
nominal_area = SPROCKET_WIDTH * SPROCKET_PITCH

detector = SprocketDetector(
    side="left",
    auto_roi=0.25,
    min_area=int(nominal_area * 0.5),      # 50% smaller
    max_area=int(nominal_area * 1.5),      # 50% larger
    ar_min=0.5,                            # relaxed a bit
    ar_max=3.0,
    solidity_min=0.3,
    edge_margin_frac=1.0,
    blur=5, open_k=7, close_k=3,
    adaptive_block=51, adaptive_C=5,
    method="profile"
)

# Calibration constants
STEPS_PER_PITCH = settings.get("steps_per_pitch_avg", 900)
SPROCKET_PITCH_PX = settings.get("sprocket_pitch_px", 148)
CROP_COORDS = settings.get("crop_coords", None)

def apply_crop(frame, crop_coords=CROP_COORDS):
    """Crop frame if crop coords available."""
    if crop_coords and len(crop_coords) == 4:
        x1, y1, x2, y2 = crop_coords
        return frame[y1:y2, x1:x2]
    return frame

async def advance_to_next_perforation(camera, websocket, steps_per_pitch=STEPS_PER_PITCH, first_frame=False):
    target_y = 246
    tolerance = 30
    max_steps = 2000
    steps_taken = 0

    # Use calibrated exposure/gain
    camera.set_controls({
        "ExposureTime": settings.get("exposure_time", 5000),
        "AnalogueGain": settings.get("gain", 1.0),
        "AeEnable": False,
        "AwbEnable": False,
        "ColourGains": (1.0, 1.0)
    })
    await asyncio.sleep(0.1)

    # --- First frame logic ---
    if first_frame:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        lores_bgr = apply_crop(lores_bgr)

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
                lores_bgr = apply_crop(lores_bgr)
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
        lores_bgr = apply_crop(lores_bgr, settings.get("crop_coords"))
        sprockets = detector.detect(lores_bgr, mode="profile")

        if sprockets:
            sprockets.sort(key=lambda s: abs(s[1] - target_y))
            _, cy, _, _, _ = sprockets[0]
            error = target_y - cy
            steps_per_pixel = steps_per_pitch / SPROCKET_PITCH_PX
            correction = int(error * steps_per_pixel)
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
        lores_bgr = apply_crop(lores_bgr)

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

        step_size = 20 if abs(cy - target_y) > tolerance * 2 else 5
        if cy > target_y:
            tc.steps_back(step_size)
        else:
            tc.steps_forward(step_size)
        steps_taken += step_size
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
                anchor = await advance_to_next_perforation(camera, websocket, first_frame=(frame==0))
                if not anchor:
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Failed to align frame {frame}'
                    }))
                    break

                # enable auto exposure just for capture
                camera.set_controls({"AeEnable": True, "AwbEnable": True})
                await asyncio.sleep(0.1)

                buffer = io.BytesIO()
                camera.capture_file(buffer, format='jpeg')
                buffer.seek(0)

                # Decode to OpenCV
                frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

                # stage 1: apply gate crop first
                frame_gate = apply_crop(frame_bgr, settings.get("crop_coords"))

                # stage 2: Apply film frame crop relative to anchor
                frame_cropped = crop_film_frame(frame_gate, anchor, settings.get("frame_crop_offsets"))

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
            await websocket.send(json.dumps({'event': 'info','message':'Stop requested'}))
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
