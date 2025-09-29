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

# --- Load calibration config ---
with open("calibration.json", "r") as f:
    config = json.load(f)

print("Loaded calibration.json:")
print(json.dumps(config, indent=2))

# --- Initialize transport ---
print("Starting WebSocket server")
tc = tcControl()
print("tcControl initialized")

# --- Initialize camera ---
camera = picamera2.Picamera2()
print("Initializing camera")

config_lores = camera.create_still_configuration(main={"size": (640, 480)})
camera.configure(config_lores)
camera.options['quality'] = 90

# --- Sprocket detector (params informed by calibration) ---
detector = SprocketDetector(
    side="left",
    auto_roi=0.25,
    min_area=3000, max_area=7000,
    ar_min=0.5, ar_max=3.0,              # widened a bit to allow "wider than tall"
    solidity_min=0.3,
    edge_margin_frac=1.0,
    blur=5, open_k=7, close_k=3,
    adaptive_block=51, adaptive_C=5,
    method="profile"                     # use profile-based detection
)

# Calibration-based constants
STEPS_PER_PITCH = config.get("steps_per_pitch_avg", 900)
SPROCKET_PITCH_PX = config.get("sprocket_pitch_px", 148)

async def advance_to_next_perforation(camera, websocket, steps_per_pitch=STEPS_PER_PITCH, first_frame=False):
    target_y = 246   # vertical target location
    tolerance = 30
    max_steps = 2000
    steps_taken = 0

    # Use calibrated exposure
    camera.set_controls({
        "ExposureTime": config.get("exposure_time", 5000),
        "AnalogueGain": config.get("gain", 1.0),
        "AeEnable": False,
        "AwbEnable": False,
        "ColourGains": (1.0, 1.0)
    })
    await asyncio.sleep(0.1)

    # --- STEP 1: First frame logic ---
    if first_frame:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

        sprockets = detector.detect(lores_bgr, debug_prefix="", mode="profile")
        if sprockets:
            sprockets.sort(key=lambda s: abs(s[1] - target_y))
            _, cy, _, _, _ = sprockets[0]

            # Back up if above target
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
            # fallback coarse move
            tc.steps_forward(steps_per_pitch)
            steps_taken += steps_per_pitch
            await asyncio.sleep(0.05)

    else:
        # --- STEP 1: Adaptive coarse step ---
        # Capture last sprocket position
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        sprockets = detector.detect(lores_bgr, mode="profile")

        if sprockets:
            sprockets.sort(key=lambda s: abs(s[1] - target_y))
            _, cy, _, _, _ = sprockets[0]
            error = target_y - cy   # positive if sprocket below target

            # adjust step size by error (pixels * steps_per_pixel)
            steps_per_pixel = steps_per_pitch / SPROCKET_PITCH_PX
            correction = int(error * steps_per_pixel)

            coarse_steps = steps_per_pitch + correction
            coarse_steps = max(steps_per_pitch // 2, min(steps_per_pitch * 2, coarse_steps))  # clamp
        else:
            coarse_steps = steps_per_pitch

        tc.steps_forward(coarse_steps)
        steps_taken += coarse_steps
        await asyncio.sleep(0.05)

    # --- STEP 2: Fine alignment loop ---
    while steps_taken < max_steps:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format='jpeg')
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

        sprockets = detector.detect(lores_bgr, debug_prefix="", mode="profile")
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

    print("[APP] Alignment failed, returning None")
    return None

async def handle_client(websocket):
    print("Client connected")
    stop_requested = False
    async for message in websocket:
        print(f"[APP] Got message: {message}")
        data = json.loads(message)

        # === START CAPTURE ===
        if data.get('event') == 'start_capture':
            num_frames = data.get('num_frames', 100)

            # --- Turn LED on only when capture starts ---
            tc.light_on()
            camera.start()
            print("[APP] LED on and camera, waiting for stabilization...")
            await asyncio.sleep(2)  # allow light + camera to stabilize

            for frame in range(num_frames):
                if stop_requested:
                    print("[APP] Stop flag detected, breaking capture loop")
                    break

                anchor = await advance_to_next_perforation(
                    camera, websocket,
                    steps_per_pitch=STEPS_PER_PITCH,
                    first_frame=(frame == 0)
                )
                if not anchor:
                    await websocket.send(json.dumps({
                        'event': 'error',
                        'message': f'Failed to align frame {frame}'
                    }))
                    break

                # Enable auto exposure/white balance for capture
                camera.set_controls({"AeEnable": True, "AwbEnable": True})
                time.sleep(0.1)

                buffer = io.BytesIO()
                camera.capture_file(buffer, format='jpeg')
                buffer.seek(0)
                img_data = buffer.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')

                await websocket.send(json.dumps({
                    'event': 'new_image',
                    'frame': frame,
                    'image': img_base64
                }))
                print(f"[APP] Sent frame {frame}")
                time.sleep(0.5)

            await websocket.send(json.dumps({'event': 'capture_complete'}))
            tc.clean_up()
            camera.stop()

        # === STOP CAPTURE ===
        if data.get('event') == 'stop_capture':
           print("[APP] Stop command received from client")
           stop_requested = True
           await websocket.send(json.dumps({
               'event': 'info',
               'message': 'Stop requested by client'
           }))
           continue

async def main():
    print("Starting WebSocket server on ws://0.0.0.0:5000")
    start_server = await websockets.serve(handle_client, "::", 5000)
    print("Server started")
    await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        print("Cleaning up")
        tc.clean_up()
        camera.stop()
