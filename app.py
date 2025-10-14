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
            ok, jpg = cv2.imencode(".jpg", dbg)
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
    min_area=settings.get("sprocket_area_min",1500), 
    max_area=settings.get("sprocket_area_max",25000),
    ar_min=1.2, ar_max=1.8,
    solidity_min=0.75,
    blur=5, open_k=5, close_k=3,
    adaptive_block=41, adaptive_C=7,
    method="profile"
)

last_error = 0 # difference between actual and target for sprocket detection

async def advance_to_next_perforation(camera, websocket, 
                                      target_y=None,
                                      steps_per_pitch=None,
                                      steps_per_px=None,
                                      k_gain=0.4,
                                      min_step=5,
                                      max_step=None,
                                      smooth_alpha=0.6,
                                      new_sprocket_min_delta_frac=0.4):
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

    step_chunk = steps_per_pitch // 2  # start with nominal half pitch
    tracked_cy = None
    smoothed_cy = None
    new_sprocket_min_delta = new_sprocket_min_delta_frac * (steps_per_pitch / steps_per_px)

    print(f"[APP] Adaptive advance: target_y={target_y}, start step_chunk={step_chunk}")

    while True:
        # --- capture & detect ---
        buffer = io.BytesIO()
        camera.capture_file(buffer, format="jpeg")
        lores_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8),
                                 cv2.IMREAD_COLOR)
        sprockets = detector.detect(lores_bgr, mode="profile")

        if not sprockets:
            tc.steps_forward(step_chunk)
            await asyncio.sleep(0.01)
            continue

        sprockets.sort(key=lambda s: s[1])  # sort top-to-bottom
        cx, cy, *_ = sprockets[0]

        if tracked_cy is None:
            tracked_cy = cy
            smoothed_cy = cy
            print(f"[APP] Tracking initial sprocket at cy={cy:.1f}")
        else:
            # smooth the y position to reduce bounce
            smoothed_cy = smooth_alpha * cy + (1 - smooth_alpha) * smoothed_cy

            if smoothed_cy < tracked_cy - new_sprocket_min_delta:
                # confirmed new sprocket rolled in
                print(f"[APP] Confirmed new sprocket at cy={smoothed_cy:.1f}, replacing old (was {tracked_cy:.1f})")

                # feedback correction
                error_px = target_y - smoothed_cy
                correction = int(error_px * steps_per_px * k_gain)

                # clamp correction magnitude
                max_corr = int(0.15 * steps_per_pitch)
                correction = max(min(correction, max_corr), -max_corr)

                new_nominal = steps_per_pitch + correction
                new_nominal = max(min(new_nominal, max_step), min_step)

                print(f"[APP] Correction: error={error_px:+.1f}px → adjust {correction:+d} steps")
                print(f"[APP] Updated nominal pitch for next advance: {new_nominal} steps")

                return (cx, smoothed_cy, new_nominal)

            tracked_cy = smoothed_cy

        # --- move forward and retry ---
        tc.steps_forward(step_chunk)
        await asyncio.sleep(0.01)


SAVE_DIR = "/media/williatf/SG1TB/GrokCam/testframes"
os.makedirs(SAVE_DIR, exist_ok=True)

async def run_capture(websocket, num_frames, stop_event, preview_width=800, debug_scale=1.0):
    print("[APP] Capture task starting")
    tc.light_on()
    camera.start()
    print("[APP] LED on + camera, stabilizing...")
    try:
        await asyncio.sleep(2)

        for frame in range(num_frames):
            if stop_event.is_set():
                print("[APP] Stop requested, leaving capture loop")
                break

            advance_result = await advance_to_next_perforation(
                camera,
                websocket,
                steps_per_pitch=settings.get("steps_per_pitch", 280),
                steps_per_px=settings.get("steps_per_px", 0.5)
            )
            if not advance_result:
                await websocket.send(json.dumps({
                    'event': 'error',
                    'message': f'Failed to align frame {frame}'
                }))
                break

            buffer = io.BytesIO()
            camera.capture_file(buffer, format='jpeg')
            buffer.seek(0)
            frame_bgr = cv2.imdecode(
                np.frombuffer(buffer.getvalue(), np.uint8),
                cv2.IMREAD_COLOR
            )

            sprockets = detector.detect(frame_bgr, mode="profile")
            debug_frame = draw_sprockets_debug(frame_bgr, sprockets if sprockets else [])
            frame_cropped = crop_film_frame(
                frame_bgr, sprockets[0] if sprockets else None, SPROCKET_PITCH_PX
            )

            timestamp = int(time.time() * 1000)
            filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.png")
            cv2.imwrite(filename, frame_cropped)  # PNG = lossless

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

            print(f"[APP] Sent frame {frame} → saved cropped {filename}")
            await asyncio.sleep(0.1)

        await websocket.send(json.dumps({'event': 'capture_complete'}))
    finally:
        tc.clean_up()
        camera.stop()
        print("[APP] Capture task cleaned up")

async def run_focus(websocket, stop_event):
    print("[APP] Focus task starting")
    tc.light_on()
    camera.start()
    print("[APP] LED on + camera for focus")
    try:
        await websocket.send(json.dumps({
            'event': 'info',
            'message': 'Focus mode started'
        }))
        while not stop_event.is_set():
            buffer = io.BytesIO()
            camera.capture_file(buffer, format='jpeg')
            frame_bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

            cv2.imshow("Focus Preview", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[APP] Focus task received local quit")
                stop_event.set()
                break
            await asyncio.sleep(0.05)
    finally:
        try:
            cv2.destroyWindow("Focus Preview")
            cv2.waitKey(1)  # ensure UI thread processes destroy
        except cv2.error as err:
            print(f"[APP] Focus destroy warning: {err}")
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        tc.light_off()
        camera.stop()
        await websocket.send(json.dumps({
            'event': 'info',
            'message': 'Focus mode stopped'
        }))
        print("[APP] Focus task cleaned up")

async def handle_client(websocket):
    print("Client connected")
    capture_task = None
    capture_stop_event = None
    focus_task = None
    focus_stop_event = None
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
            camera.start()
            print("[APP] LED on + camera, stabilizing...")

            steps_per_pitch = STEPS_PER_PITCH
            for f in range(frames):
                if direction > 0:
                    tc.steps_forward(steps_per_pitch)
                else:
                    tc.steps_back(steps_per_pitch)
                    #tc.rewind()
                #await asyncio.sleep(0.05)

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
            focus_task = asyncio.create_task(run_focus(websocket, focus_stop_event))
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

async def main():
    print("Starting WebSocket server on ws://0.0.0.0:5000")
    server = await websockets.serve(
        handle_client,
        "::",
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
