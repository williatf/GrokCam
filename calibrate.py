import cv2
import numpy as np
import time
import json
from picamera2 import Picamera2
from control import tcControl
from sprocket import SprocketDetector
import random

def auto_calibrate_exposure(camera, target_p99=240, max_iter=10):
    """
    Adjust exposure until the 99th percentile (p99) of brightness is near target_p99.
    """
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

        # Adjust exposure proportionally
        exposure = int(exposure * target_p99 / max(p99, 1))
        exposure = max(100, min(20000, exposure))
        camera.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
        time.sleep(0.2)

    return frame, exposure, gain

def measure_steps_per_pitch(camera, tc, detector, sprocket_pitch_px, step_chunk=20, max_steps=500):
    """
    Estimate steps per pitch by rolling anchor: track sprockets as they move down the frame.
    """
    # --- Capture starting frame ---
    request = camera.capture_request()
    frame_before = request.make_array("main")
    request.release()

    sprockets_before = detector.detect(frame_before, mode="profile")
    if not sprockets_before:
        print("[CALIB] No sprockets detected at start.")
        return None

    H = frame_before.shape[0]
    # Pick sprocket closest to vertical center as initial anchor
    anchor = min(sprockets_before, key=lambda s: s[1])
    _, cy_anchor, _, _, _ = anchor
    cy_first = cy_anchor

    total_steps = 0

    while total_steps < max_steps:
        # --- Move transport ---
        tc.steps_forward(step_chunk)
        total_steps += step_chunk
        time.sleep(0.05)

        # --- Capture new frame ---
        request = camera.capture_request()
        frame_after = request.make_array("main")
        request.release()

        sprockets_after = detector.detect(frame_after, mode="profile")
        if not sprockets_after:
            continue

        # --- Look for sprocket just *below* the current anchor ---
        candidates = [s for s in sprockets_after if s[1] > cy_anchor]
        if not candidates:
            print(f"[CALIB] No valid candidate at step {total_steps}, skipping…")
            continue

        # Choose sprocket closest below the current anchor
        best = min(candidates, key=lambda s: s[1] - cy_anchor)
        _, cy_new, _, _, _ = best
        print(f"[CALIB] New anchor cy={cy_new:.1f} (prev={cy_anchor:.1f})")

        # Update anchor
        cy_anchor = cy_new

        # Measure total vertical displacement from very first sprocket
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
    """Remove outliers using Median Absolute Deviation (MAD)."""
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
    config = camera.create_still_configuration(main={"size": (640, 480)})
    camera.configure(config)
    camera.start()
    time.sleep(1)

    calibrated_frame, exp, gain = auto_calibrate_exposure(camera)

    detector = SprocketDetector(
        side="left", auto_roi=0.25,
        min_area=3000, max_area=10000,
        ar_min=1.0, ar_max=2.0,
        solidity_min=0.9,
        blur=5, open_k=7, close_k=3,
        adaptive_block=51, adaptive_C=5
    )
    sprockets = detector.detect(calibrated_frame, debug_prefix="calib", mode="profile")
    if not sprockets:
        print("[CALIB] No sprockets found in calibrated frame, aborting.")
        tc.light_off()
        camera.stop()
        return

    widths = [w for (_, _, w, _, _) in sprockets]
    avg_width = int(np.mean(widths))
    sprocket_pitch_px = int(np.mean([sprockets[i+1][1] - sprockets[i][1] for i in range(len(sprockets)-1)]))

    print(f"[CALIB] Expected sprocket width={avg_width}px, pitch={sprocket_pitch_px}px")

    runs = 10
    steps_per_pitch_runs = []
    for run in range(runs):
        print(f"\n[CALIB] === Run {run+1}/{runs} ===")
        steps = measure_steps_per_pitch(camera, tc, detector, sprocket_pitch_px)
        if steps:
            steps_per_pitch_runs.append(steps)

        # Random offset between runs
        offset = random.randint(100,500)
        direction = random.choice([1,1])

        if direction == 1:
            print(f"[CALIB] Jogging forward {offset} steps")
            tc.steps_forward(offset)
        else:
            print(f"[CALIB] Jogging back {offset} steps")
            tc.steps_back(offset)

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


if __name__ == "__main__":
    main()
