import cv2
import numpy as np
import time
import json
from picamera2 import Picamera2
from control import tcControl
from sprocket import SprocketDetector
import random
import os

CONFIG_FILE = "config.json"

def show_debug(frame, sprockets, title="calib"):
    """Draw boxes using (cx, cy, w, h, area) tuples from SprocketDetector."""
    dbg = frame.copy()
    for (cx, cy, w, h, _) in sprockets:
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(dbg, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)
    cv2.imshow(title, dbg)
    cv2.waitKey(1)

def load_crop():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
            return cfg.get("crop_coords")
    return None

def auto_calibrate_exposure(camera, target_p99=240, max_iter=10):
    exposure = 2000
    gain = 1.0
    camera.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
    time.sleep(0.2)

    for i in range(max_iter):
        request = camera.capture_request()
        frame = request.make_array("main")
        request.release()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p99 = np.percentile(gray, 99)
        print(f"[CALIB] Iter {i}: p99={int(p99)}, Exposure={exposure}, Gain={gain}")
        if abs(p99 - target_p99) < 5:
            print("[CALIB] Target reached.")
            break
        exposure = int(exposure * target_p99 / max(p99, 1))
        exposure = max(100, min(20000, exposure))
        camera.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
        time.sleep(0.2)
    return frame, exposure, gain

def apply_crop(frame, crop_coords):
    if crop_coords and len(crop_coords) == 4:
        x1, y1, x2, y2 = crop_coords
        return frame[y1:y2, x1:x2]
    return frame

def measure_steps_per_pitch(camera, tc, detector, sprocket_pitch_px, crop_coords, step_chunk=20, max_steps=500):
    request = camera.capture_request()
    frame_before = apply_crop(request.make_array("main"), crop_coords)
    request.release()
    sprockets_before = detector.detect(frame_before, mode="profile")
    if not sprockets_before:
        print("[CALIB] No sprockets detected at start.")
        return None

    cy_anchor = min(sprockets_before, key=lambda s: s[1])[1]
    cy_first = cy_anchor
    total_steps = 0

    while total_steps < max_steps:
        tc.steps_forward(step_chunk)
        total_steps += step_chunk
        time.sleep(0.05)
        request = camera.capture_request()
        frame_after = apply_crop(request.make_array("main"), crop_coords)
        request.release()
        sprockets_after = detector.detect(frame_after, mode="profile")
        if sprockets_after:
            show_debug(frame_after, sprockets_after, title="Tracking")
        if not sprockets_after:
            continue
        candidates = [s for s in sprockets_after if s[1] > cy_anchor]
        if not candidates:
            print(f"[CALIB] No valid candidate at step {total_steps}, skipping…")
            continue
        cy_new = min(candidates, key=lambda s: s[1] - cy_anchor)[1]
        print(f"[CALIB] New anchor cy={cy_new:.1f} (prev={cy_anchor:.1f})")
        cy_anchor = cy_new
        delta_y = cy_anchor - cy_first
        if delta_y >= sprocket_pitch_px:
            steps_per_pixel = total_steps / delta_y
            steps_per_pitch = steps_per_pixel * sprocket_pitch_px
            print(f"[CALIB] total Δy={delta_y:.1f}px, steps={total_steps}, "
                  f"steps/px={steps_per_pixel:.2f}, steps/pitch={steps_per_pitch:.1f}")
            return int(steps_per_pitch)
    print("[CALIB] Failed to measure pitch within max_steps.")
    return None

def reject_outliers(data, m=2.5):
    if len(data) < 3:
        return data
    median = np.median(data)
    abs_dev = np.abs(data - median)
    mad = np.median(abs_dev)
    if mad == 0:
        return data
    mask = abs_dev / mad < m
    return data[mask]

def main():
    print("LED on")
    tc = tcControl()
    tc.light_on()

    camera = Picamera2()
    config = camera.create_still_configuration(main={"size": (1920, 1080)})
    camera.configure(config)
    camera.start()
    time.sleep(1)

    crop_coords = load_crop()
    print(f"[CALIB] Using crop: {crop_coords}")

    full_frame, exp, gain = auto_calibrate_exposure(camera)
    cropped_frame = apply_crop(full_frame, crop_coords)

    detector = SprocketDetector(
        side="left", auto_roi=0.40,
        min_area=2000, max_area=12000,
        ar_min=0.8, ar_max=2.2,
        solidity_min=0.85,
        blur=5, open_k=5, close_k=3,
        adaptive_block=41, adaptive_C=7
    )
    sprockets = detector.detect(cropped_frame, debug_prefix="calib", mode="profile")
    if not sprockets:
        print("[CALIB] No sprockets found in cropped frame, aborting.")
        tc.light_off()
        camera.stop()
        return

    show_debug(cropped_frame, sprockets, title="Initial detection")

    widths = [w for (_, _, w, _, _) in sprockets]
    avg_width = int(np.mean(widths))
    sprocket_pitch_px = int(np.mean([sprockets[i+1][1] - sprockets[i][1] for i in range(len(sprockets)-1)]))
    print(f"[CALIB] Expected sprocket width={avg_width}px, pitch={sprocket_pitch_px}px")

    runs = 10
    steps_per_pitch_runs = []
    for run in range(runs):
        print(f"\n[CALIB] === Run {run+1}/{runs} ===")
        steps = measure_steps_per_pitch(camera, tc, detector, sprocket_pitch_px, crop_coords)
        if steps:
            steps_per_pitch_runs.append(steps)
        offset = random.randint(100, 500)
        print(f"[CALIB] Jogging forward {offset} steps")
        tc.steps_forward(offset)
        time.sleep(0.5)

    steps_per_pitch_runs = np.array(steps_per_pitch_runs)
    filtered = reject_outliers(steps_per_pitch_runs)
    avg_steps = int(np.mean(filtered)) if len(filtered) else None

    print(f"[CALIB] Runs: {steps_per_pitch_runs.tolist()}")
    print(f"[CALIB] Filtered: {filtered.tolist() if len(filtered) else []}")
    print(f"[CALIB] Final averaged steps_per_pitch={avg_steps}")

    calibration_data = {
        "exposure_time": exp,
        "gain": gain,
        "crop_coords": crop_coords,
        "sprocket_width_px": avg_width,
        "sprocket_pitch_px": sprocket_pitch_px,
        "steps_per_pitch_runs": steps_per_pitch_runs.tolist(),
        "steps_per_pitch_filtered": filtered.tolist() if len(filtered) else [],
        "steps_per_pitch_avg": avg_steps
    }
    with open("calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=2)

    print("[CALIB] Saved calibration.json")
    tc.light_off()
    camera.stop()
    print("Cleanup complete, LED off")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
