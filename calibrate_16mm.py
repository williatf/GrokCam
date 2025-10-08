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
BOTTOM_MARGIN_FRAC = 0.12  # stop tracking when anchor gets within ~12% of bottom

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
    dbg_small = cv2.resize(dbg,(1280,960))
    cv2.imshow(title, dbg_small)
    cv2.waitKey(1)

# -------------------- Exposure auto-cal --------------------
def auto_calibrate_exposure(camera, settle_time=2.0):
    """
    Enable auto-exposure and auto-white-balance briefly, let them settle,
    then lock the chosen ExposureTime and AnalogueGain for consistent captures.
    """
    # Turn on auto exposure/white balance
    camera.set_controls({
        "AeEnable": True,
        "AwbEnable": True
    })
    print("[CALIB] Auto-exposure enabled, settling...")
    time.sleep(settle_time)

    # Grab current metadata (chosen by AE)
    metadata = camera.capture_metadata()
    exposure = metadata.get("ExposureTime", None)
    gain = metadata.get("AnalogueGain", None)

    print(f"[CALIB] AE selected ExposureTime={exposure}, Gain={gain}")

    # Lock those values
    if exposure and gain:
        camera.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exposure,
            "AnalogueGain": gain
        })
        print("[CALIB] Exposure locked.")
    else:
        print("[CALIB] Warning: AE did not return exposure/gain.")

    # Capture one frame with locked settings for confirmation
    req = camera.capture_request()
    frame = req.make_array("main")
    req.release()

    return frame, exposure, gain

def ensure_two_sprockets(camera, tc, detector, step_chunk=STEP_CHUNK, max_steps=MAX_STEPS_SEARCH):
    steps = 0
    while steps <= max_steps:
        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()
        sprockets = detector.detect(frame, mode="profile")
        print(f"[CALIB] Sprockets detected is {len(sprockets)}")
        if len(sprockets) >= 2:
            return frame, sprockets, steps
        tc.steps_forward(step_chunk)
        steps += step_chunk
        time.sleep(0.02)
    return None, [], steps

# -------------------- Measure steps-per-pixel by tracking one sprocket --------------------
def measure_steps_per_pitch(camera, tc, detector,
                            step_chunk=STEP_CHUNK, 
                            max_steps=MAX_STEPS_TRACK,
                            target_frac=0.35,  # desired sprocket center fraction (e.g., 0.35 = 35% down from top)
                            tolerance_px=25,    # acceptable deviation from target
                            k_gain=0.5,        # proportional gain for correction
                            min_step=2,
                            max_step=25):
    """
    Track sprocket motion while dynamically adjusting step size
    so the sprocket center converges to a target position in the frame.
    Returns (pitch_px, steps_per_px) or (None, None).
    """
    # Capture initial frame and detect sprockets
    req = camera.capture_request()
    frame = req.make_array("main")
    req.release()

    spro = detector.detect(frame, mode="profile")
    if len(spro) < 2:
        print("[CALIB] Need at least two sprockets in view to measure pitch.")
        return None, None

    # Sort top-to-bottom
    spro_sorted = sorted(spro, key=lambda s: s[1])
    top_sprocket = spro_sorted[0]
    bottom_sprocket = spro_sorted[1]

    cy_start = top_sprocket[1]
    cy_target = bottom_sprocket[1]
    pitch_px = cy_target - cy_start

    H = frame.shape[0]
    target_y = int(H * target_frac)
    print(f"[CALIB] Target sprocket center set to y={target_y}")

    steps_total = 0
    last_debug = 0
    cy_anchor = cy_start
    cx_anchor = top_sprocket[0]

    print(f"[TRACK] Starting tracking from cy={cy_start:.1f} toward target_y={target_y}")

    while steps_total <= max_steps:
        # Calculate pixel error from target
        error_px = target_y - cy_anchor
        correction = int(error_px * k_gain)
        correction = max(min(correction, max_step), -max_step)
        next_step = max(min_step, step_chunk + correction)

        # Step film transport
        tc.steps_forward(next_step)
        steps_total += next_step
        time.sleep(0.02)

        # Capture new frame
        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()

        spro_after = detector.detect(frame, mode="profile")
        if not spro_after:
            print("[TRACK] Lost sprocket—stop.")
            break

        # Find closest sprocket horizontally
        sprocket = min(spro_after, key=lambda s: abs(s[0] - cx_anchor))
        cx_new, cy_new, w_new, h_new, _ = sprocket

        if steps_total - last_debug >= 50:
            show_debug(frame, spro_after, title="TrackAutoStep")
            last_debug = steps_total

        print(f"[TRACK] cy: {cy_anchor:.1f} -> {cy_new:.1f} (err={error_px:+.1f}px, step={next_step})")
        cy_anchor = cy_new

        # Exit when within tolerance of target
        if abs(cy_anchor - target_y) <= tolerance_px:
            print(f"[TRACK] Target reached at cy={cy_anchor:.1f}, within ±{tolerance_px}px")
            break

    # Final calculation
    delta_y = cy_anchor - cy_start
    if delta_y <= 0 or steps_total <= 0:
        print(f"[TRACK] Invalid track (Δy={delta_y}, steps={steps_total}).")
        return None, None

    steps_per_px = steps_total / delta_y
    print(f"[CALIB] Δy={delta_y:.1f}px, steps={steps_total} → steps/px={steps_per_px:.4f}")

    return pitch_px, steps_per_px

def compute_area_bounds(areas):
    """
    Compute robust nominal/min/max sprocket area bounds.
    Returns (nominal_area, min_area, max_area).
    """
    if not areas:
        return 10000, 7000, 13000  # fallback for safety

    a = np.array(areas, dtype=float)
    a = a[a > 0]
    if a.size < 5:
        med = float(np.median(a))
        return int(med), int(med * 0.7), int(med * 1.3)

    med = float(np.median(a))
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    iqr = max(1.0, q3 - q1)

    lo = max(med * 0.6, q1 - 1.5 * iqr)
    hi = min(med * 1.6, q3 + 1.5 * iqr)

    # Ensure at least ±20% band
    lo = min(lo, med * 0.8)
    hi = max(hi, med * 1.2)

    return int(med), int(lo), int(hi)


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

    # Auto exposure
    full_frame, exp, gain = auto_calibrate_exposure(camera)

    detector = SprocketDetector(
        side="left", auto_roi=0.40,
        min_area=83000, max_area=125000,
        ar_min=1.2, ar_max=1.8,
        solidity_min=0.75,
        blur=5, open_k=5, close_k=3,
        adaptive_block=41, adaptive_C=7,
        method="profile"
    )

    runs = 8
    pitch_vals, steps_per_pitch_vals = [], []
    area_samples = []
    for run in range(runs):
        print(f"\n[CALIB] === Run {run+1}/{runs} ===")

        # Ensure we have two sprockets
        frame, spro, _ = ensure_two_sprockets(camera, tc, detector)
        if not spro or len(spro) < 2:
            print("[CALIB] Could not find 2 sprockets, skipping run.")
            continue

        # Collect area samples from detected sprockets
        for (_, _, _, _, area) in spro:
            area_samples.append(area)


        pitch_px, steps_per_px = measure_steps_per_pitch(camera, tc, detector)
        if pitch_px and steps_per_px:
            steps_per_pitch = steps_per_px * pitch_px
            pitch_vals.append(pitch_px)
            steps_per_pitch_vals.append(steps_per_pitch)
            print(f"[CALIB] pitch_px={pitch_px:.1f}, steps/px={steps_per_px:.4f} → steps/pitch={steps_per_pitch:.1f}")
        else:
            print("[CALIB] Measurement failed.")

        # Jog forward
        offset = random.randint(80, 300)
        tc.steps_forward(offset)
        time.sleep(0.3)

    # Filter and average
    pitch_vals = np.array(pitch_vals, dtype=float)
    steps_per_pitch_vals = np.array(steps_per_pitch_vals, dtype=float)
    pitch_filtered = reject_outliers(pitch_vals)
    spp_filtered = reject_outliers(steps_per_pitch_vals)

    avg_pitch = np.mean(pitch_filtered) if len(pitch_filtered) else None
    avg_spp = int(round(np.mean(spp_filtered))) if len(spp_filtered) else None

    print(f"\n[CALIB] Final pitch_px={avg_pitch:.1f}, steps/pitch={avg_spp}")

    # --- Compute sprocket area bounds ---
    nom_area, min_area, max_area = compute_area_bounds(area_samples)
    print(f"[CALIB] Sprocket area nominal={nom_area}, range=({min_area}, {max_area})")


    # Save calibration.json (simplified)
    calibration_data = {
        "exposure_time": exp,
        "gain": gain,
        "sprocket_pitch_px": float(avg_pitch) if avg_pitch else None,
        "steps_per_pitch": avg_spp,
        "calibration_resolution": list(FULL_RES),
        "sprocket_area_nominal": nom_area,
        "sprocket_area_min": min_area,
        "sprocket_area_max": max_area
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
