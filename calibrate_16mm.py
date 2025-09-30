#!/usr/bin/env python3
import cv2
import numpy as np
import time
import json
from picamera2 import Picamera2
from control import tcControl
from sprocket import SprocketDetector
import random
import os

os.environ["DISPLAY"] = ":0" # force OpenCV windows to appear on Pi's HDMI monitor

# -------------------- Config --------------------
FULL_RES = (2028, 1520)  # full FoV bin2 for HQ cam. Use (4056, 3040) for full native.
STEP_CHUNK = 10          # small movement for searching/tracking
MAX_STEPS_SEARCH = 2000  # cap during search for two sprockets
MAX_STEPS_TRACK  = 2000  # cap during single-sprocket tracking
BOTTOM_MARGIN_FRAC = 0.06  # stop tracking when anchor gets within ~6% of bottom

CONFIG_FILE = "config.json"

# -------------------- Debug draw --------------------
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

# -------------------- Exposure auto-cal --------------------
def auto_calibrate_exposure(camera, target_p99=240, max_iter=10):
    exposure = 1500
    gain = 1.0
    camera.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
    time.sleep(0.2)

    last_frame = None
    for i in range(max_iter):
        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()
        last_frame = frame

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

    return last_frame, exposure, gain

# -------------------- Ensure two sprockets visible --------------------
def ensure_two_sprockets(camera, tc, detector, step_chunk=STEP_CHUNK, max_steps=MAX_STEPS_SEARCH):
    """
    Move forward in small increments until we see >= 2 sprockets.
    Returns (frame, sprockets, steps_moved). If never found, returns (frame, sprockets_or_empty, steps).
    """
    steps = 0
    while steps <= max_steps:
        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()

        sprockets = detector.detect(frame, mode="profile")
        if sprockets:
            show_debug(frame, sprockets, title="SearchTwo")
        if len(sprockets) >= 2:
            return frame, sprockets, steps

        tc.steps_forward(step_chunk)
        steps += step_chunk
        time.sleep(0.02)

    print("[CALIB] Could not get two sprockets within max search steps.")
    return frame, sprockets if 'sprockets' in locals() else [], steps

# -------------------- Measure steps-per-pixel by tracking one sprocket --------------------
def measure_steps_per_pixel(camera, tc, detector, step_chunk=STEP_CHUNK, max_steps=MAX_STEPS_TRACK):
    """
    Track a single sprocket as it moves down until it leaves (or near bottom).
    Returns steps_per_pixel (float) or None.
    """
    # capture start
    req = camera.capture_request()
    frame = req.make_array("main")
    req.release()

    spro = detector.detect(frame, mode="profile")
    if not spro:
        print("[CALIB] No sprockets detected to start tracking.")
        return None

    # Pick the *top-most* sprocket as anchor so it has room to travel down
    anchor = min(spro, key=lambda s: s[1])
    cy_anchor = anchor[1]
    cy_start  = cy_anchor

    H = frame.shape[0]
    bottom_stop = H * (1.0 - BOTTOM_MARGIN_FRAC)

    steps_total = 0
    last_debug = 0

    print(f"[TRACK] Start anchor cy={cy_anchor:.1f}, bottom_stop={bottom_stop:.1f}")

    while steps_total <= max_steps:
        # move small chunk
        tc.steps_forward(step_chunk)
        steps_total += step_chunk
        time.sleep(0.02)

        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()

        spro_after = detector.detect(frame, mode="profile")
        if spro_after:
            if steps_total - last_debug >= 50:
                show_debug(frame, spro_after, title="TrackOne")
                last_debug = steps_total
        if not spro_after:
            print("[TRACK] Lost all sprockets—stop.")
            break

        # Find candidate below previous anchor (film moving down)
        candidates = [s for s in spro_after if s[1] > cy_anchor]
        if not candidates:
            print("[TRACK] No candidate below anchor—stop.")
            break

        # Choose closest below
        cy_new = min(candidates, key=lambda s: s[1] - cy_anchor)[1]
        # update
        print(f"[TRACK] cy: {cy_anchor:.1f} -> {cy_new:.1f} (+{cy_new - cy_anchor:.1f}px) at steps {steps_total}")
        cy_anchor = cy_new

        if cy_anchor >= bottom_stop:
            print("[TRACK] Near bottom of frame—stop.")
            break

    delta_y = cy_anchor - cy_start
    if delta_y <= 0 or steps_total <= 0:
        print(f"[TRACK] Invalid track (Δy={delta_y}, steps={steps_total}).")
        return None

    spp = steps_total / delta_y
    print(f"[TRACK] Δy={delta_y:.1f}px, steps={steps_total} → steps/px={spp:.4f}")
    return spp

# -------------------- Outlier filter --------------------
def reject_outliers(arr, m=2.5):
    arr = np.asarray(arr)
    if len(arr) < 3:
        return arr
    med = np.median(arr)
    dev = np.abs(arr - med)
    mad = np.median(dev)
    if mad == 0:
        return arr
    mask = dev / mad < m
    return arr[mask]

# -------------------- Interactive relative crop (optional) --------------------
def select_relative_crop(img, anchor_xy, win_name="Select Film Frame"):
    """
    Let user drag a rectangle on img. Returns [dx1, dy1, dx2, dy2] relative to anchor_xy (full frame).
    Keys: Enter/y accept, c clear, q/ESC skip (returns None).
    """
    anchor_x, anchor_y = int(round(anchor_xy[0])), int(round(anchor_xy[1]))
    base = img.copy()
    drawing = {"pt1": None, "pt2": None, "drag": False}

    def overlay():
        vis = base.copy()
        cv2.circle(vis, (anchor_x, anchor_y), 5, (0, 0, 255), -1)
        cv2.line(vis, (anchor_x, 0), (anchor_x, vis.shape[0]-1), (0, 0, 255), 1)
        cv2.line(vis, (0, anchor_y), (vis.shape[1]-1, anchor_y), (0, 0, 255), 1)
        if drawing["pt1"] and drawing["pt2"]:
            x1, y1 = drawing["pt1"]; x2, y2 = drawing["pt2"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return vis

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["pt1"] = (x, y)
            drawing["pt2"] = (x, y)
            drawing["drag"] = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing["drag"]:
            drawing["pt2"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["pt2"] = (x, y)
            drawing["drag"] = False

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, overlay())
    cv2.setMouseCallback(win_name, on_mouse)
    print("[CROP] Drag film frame. Enter/y=save, c=clear, q/ESC=skip.")

    while True:
        cv2.imshow(win_name, overlay())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord('y')):
            if drawing["pt1"] and drawing["pt2"]:
                (x1, y1) = drawing["pt1"]; (x2, y2) = drawing["pt2"]
                x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
                h, w = img.shape[:2]
                x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
                dx1, dy1 = x1 - anchor_x, y1 - anchor_y
                dx2, dy2 = x2 - anchor_x, y2 - anchor_y
                cv2.destroyWindow(win_name)
                print(f"[CROP] Saved offsets: ({dx1},{dy1})→({dx2},{dy2})")
                return [int(dx1), int(dy1), int(dx2), int(dy2)]
            else:
                print("[CROP] No rectangle drawn.")
        elif key == ord('c'):
            drawing["pt1"] = None; drawing["pt2"] = None
        elif key in (27, ord('q')):
            cv2.destroyWindow(win_name)
            print("[CROP] Skipped.")
            return None

def self_scan_pitch(camera, tc, detector, step_size=10, max_steps=800):
    """
    Move transport in small increments until a new sprocket enters view.
    Measure Δy between first sprocket and next sprocket to estimate pitch.
    """
    print("[CALIB] Starting self-scan pitch measurement...")
    request = camera.capture_request()
    frame = request.make_array("main")
    request.release()

    sprockets = detector.detect(frame, mode="profile")
    if not sprockets:
        print("[CALIB] No sprocket detected to start self-scan.")
        return None

    cy_first = sprockets[0][1]
    cy_anchor = cy_first
    steps_taken = 0

    while steps_taken < max_steps:
        tc.steps_forward(step_size)
        steps_taken += step_size
        time.sleep(0.05)

        request = camera.capture_request()
        frame = request.make_array("main")
        request.release()

        sprockets = detector.detect(frame, mode="profile")
        if not sprockets:
            continue

        # pick sprocket below the anchor
        candidates = [s for s in sprockets if s[1] > cy_anchor]
        if candidates:
            cy_new = min(candidates, key=lambda s: s[1] - cy_anchor)[1]
            delta_y = cy_new - cy_first
            if delta_y > 20:  # sanity threshold
                print(f"[CALIB] Found second sprocket at Δy={delta_y:.1f}px after {steps_taken} steps")
                return int(delta_y)

    print("[CALIB] Self-scan failed, no second sprocket found.")
    return None

# -------------------- Main --------------------
def main():
    print("LED on")
    tc = tcControl()
    tc.light_on()

    camera = Picamera2()
    cfg = camera.create_still_configuration(main={"size": FULL_RES})
    camera.configure(cfg)
    camera.start()
    time.sleep(1)

    # Auto exposure on *full frame*
    full_frame, exp, gain = auto_calibrate_exposure(camera)

    # Detector tuned for full frame, left strip
    detector = SprocketDetector(
        side="left", auto_roi=0.40,   # search 40% strip at left
        min_area=1500, max_area=25000,
        ar_min=1.2, ar_max=1.8,
        solidity_min=0.75,
        blur=5, open_k=5, close_k=3,
        adaptive_block=41, adaptive_C=7,
        method="profile"
    )

    # Ensure two sprockets are visible so we can measure pitch in px
    frame_two, spro_two, _ = ensure_two_sprockets(camera, tc, detector)
    if len(spro_two) < 2:
        print("[CALIB] Still <2 sprockets; calibration will proceed but pitch_px may be noisy.")
    show_debug(frame_two, spro_two, title="TwoSprockets")

# Estimate pitch_px
    pitch_px = None
    if len(spro_two) >= 2:
        spro_two_sorted = sorted(spro_two, key=lambda s: s[1])
        diffs = [spro_two_sorted[i+1][1] - spro_two_sorted[i][1]
             for i in range(len(spro_two_sorted)-1)]
        pitch_px = float(np.median(diffs))
        print(f"[CALIB] pitch_px estimate from 2 sprockets in view: {pitch_px:.1f}")
    else:
        # fallback: run self-scan to measure pitch from motion
        pitch_px = self_scan_pitch(camera, tc, detector)
        if pitch_px:
            print(f"[CALIB] pitch_px estimate from self-scan: {pitch_px:.1f}")
        else:
            pitch_px = 300.0
            print(f"[CALIB] Using fallback pitch_px={pitch_px:.1f}")
    detector.update_pitch(pitch_px)

    # Multi-run measure steps/pitch via steps/px * pitch_px
    runs = 8
    steps_per_pitch_runs = []
    for run in range(runs):
        print(f"\n[CALIB] === Run {run+1}/{runs} ===")
        spp = measure_steps_per_pixel(camera, tc, detector)
        if spp is not None and pitch_px is not None and pitch_px > 0:
            sppitch = int(round(spp * pitch_px))
            print(f"[CALIB] steps/px={spp:.4f}, pitch_px={pitch_px:.1f} → steps/pitch={sppitch}")
            steps_per_pitch_runs.append(sppitch)
        else:
            print("[CALIB] Skipping run; insufficient measurement.")

        # jog randomly between runs (forward only to avoid rewinding issues)
        offset = random.randint(80, 300)
        print(f"[CALIB] Jogging forward {offset} steps")
        tc.steps_forward(offset)
        time.sleep(0.3)

    runs_arr = np.array(steps_per_pitch_runs, dtype=float)
    filtered = reject_outliers(runs_arr)
    avg_steps = int(round(np.mean(filtered))) if len(filtered) else None

    print(f"\n[CALIB] Runs: {runs_arr.tolist()}")
    print(f"[CALIB] Filtered: {filtered.tolist() if len(filtered) else []}")
    print(f"[CALIB] Final averaged steps_per_pitch={avg_steps}")

    # Grab a fresh frame to store a film-frame crop relative to sprocket center (FULL frame reference)
    req = camera.capture_request()
    frame_now = req.make_array("main")
    req.release()

    spro_now = detector.detect(frame_now, mode="profile")
    rel_offsets = None
    if spro_now:
        H = frame_now.shape[0]
        # anchor near center vertically (nicer for framing selection)
        anchor = min(spro_now, key=lambda s: abs(s[1] - H/2))
        anchor_cx, anchor_cy = float(anchor[0]), float(anchor[1])
        print(f"[CROP] Anchor sprocket at (cx={anchor_cx:.1f}, cy={anchor_cy:.1f}) in FULL frame.")
        rel_offsets = select_relative_crop(frame_now, (anchor_cx, anchor_cy), win_name="Select Film Frame")
    else:
        print("[CROP] Could not find sprocket for frame-crop selection; skipping.")

    # Save calibration.json
    calibration_data = {
        "exposure_time": exp,
        "gain": gain,
        "sprocket_pitch_px": float(pitch_px) if pitch_px is not None else None,
        "steps_per_pitch_runs": runs_arr.tolist(),
        "steps_per_pitch_filtered": filtered.tolist() if len(filtered) else [],
        "steps_per_pitch_avg": avg_steps,
        # film-frame crop offsets relative to sprocket center in FULL frame
        "frame_crop_offsets": rel_offsets,
        "frame_crop_ref": "sprocket_center_in_full_frame",
        # store which resolution was used for consistency
        "calibration_resolution": list(FULL_RES)
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
