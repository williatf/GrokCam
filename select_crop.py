import cv2
import json
import time
from picamera2 import Picamera2

# Output config file
CONFIG_FILE = "config.json"

# Globals for drawing rectangle
refPt = []
cropping = False
image = None
clone = None
scale_factor = 1.0

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        # Draw rectangle on preview
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Crop Selector", image)

def main():
    global image, clone, scale_factor

    # --- Initialize camera ---
    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (1920, 1080)})
    cam.configure(config)
    cam.start()
    time.sleep(2)  # allow camera to warm up

    # --- Capture full resolution frame ---
    frame = cam.capture_array("main")
    cam.stop()

    H, W = frame.shape[:2]

    # --- Scale for display (fit into 1280px wide window, preserving aspect) ---
    max_display_width = 1280
    scale_factor = min(1.0, max_display_width / W)
    display_size = (int(W * scale_factor), int(H * scale_factor))
    image = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
    clone = image.copy()

    cv2.namedWindow("Crop Selector", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Crop Selector", click_and_crop)

    print("[INFO] Draw a rectangle with the mouse, press 'r' to reset, 'c' to confirm, 'q' to quit")

    while True:
        cv2.imshow("Crop Selector", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
            refPt.clear()

        elif key == ord("c"):
            if len(refPt) == 2:
                # Scale back to original full resolution coords
                (x1, y1), (x2, y2) = refPt
                x1 = int(x1 / scale_factor)
                y1 = int(y1 / scale_factor)
                x2 = int(x2 / scale_factor)
                y2 = int(y2 / scale_factor)
                crop_coords = [x1, y1, x2, y2]

                # Save to config.json
                try:
                    with open(CONFIG_FILE, "r") as f:
                        config_data = json.load(f)
                except FileNotFoundError:
                    config_data = {}

                config_data["crop_coords"] = crop_coords

                with open(CONFIG_FILE, "w") as f:
                    json.dump(config_data, f, indent=2)

                print(f"[INFO] Saved crop coords: {crop_coords} → {CONFIG_FILE}")
                break
            else:
                print("[WARN] No crop selected!")

        elif key == ord("q"):
            print("[INFO] Quit without saving")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
