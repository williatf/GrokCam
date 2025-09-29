import cv2 as cv
import numpy as np

class SprocketDetector:
    def __init__(self, side="left", auto_roi=0.25,
                 min_area=3000, max_area=6000,
                 ar_min=1.0, ar_max=2.0,
                 solidity_min=0.3, edge_margin_frac=1.0,
                 blur=5, open_k=7, close_k=3,
                 adaptive_block=51, adaptive_C=5,
                 method="adaptive", inv=True):
        self.side = side
        self.auto_roi = auto_roi
        self.min_area = min_area
        self.max_area = max_area
        self.ar_min = ar_min
        self.ar_max = ar_max
        self.solidity_min = solidity_min
        self.edge_margin_frac = edge_margin_frac
        self.blur = blur
        self.open_k = open_k
        self.close_k = close_k
        self.adaptive_block = adaptive_block
        self.adaptive_C = adaptive_C
        self.method = method   # "adaptive" or "otsu"
        self.inv = inv

        # Calibration
        self.expected_width = None
        self.expected_pitch = None
        self.width_tol = 0.25
        self.pitch_tol = 12

    def detect(self, frame_bgr, debug_prefix=None, mode="contour"):
        """Dispatch to contour-based or profile-based detection."""
        H, W = frame_bgr.shape[:2]
        strip_w = int(W * self.auto_roi)

        if self.side == "left":
            roi = frame_bgr[:, :strip_w]
            roi_offset = (0, 0)
        else:
            roi = frame_bgr[:, -strip_w:]
            roi_offset = (W - strip_w, 0)

        if mode == "profile":
            return self._detect_profile(roi, roi_offset, frame_bgr, debug_prefix)
        else:
            return self._detect_contours(roi, roi_offset, frame_bgr, debug_prefix)

    def _detect_contours(self, roi, roi_offset, frame_bgr, debug_prefix=None):
        """Contour-based detection pipeline."""
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if self.blur > 0:
            gray = cv.GaussianBlur(gray, (self.blur, self.blur), 0)

        # Threshold
        if self.method == "adaptive":
            th = cv.adaptiveThreshold(
                gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY_INV if self.inv else cv.THRESH_BINARY,
                self.adaptive_block, self.adaptive_C
            )
        else:  # Otsu
            _, th = cv.threshold(
                gray, 0, 255,
                (cv.THRESH_BINARY_INV if self.inv else cv.THRESH_BINARY) | cv.THRESH_OTSU
            )

        # Morph cleanup
        k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.open_k, self.open_k))
        k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.close_k, self.close_k))
        mask = cv.morphologyEx(th, cv.MORPH_OPEN, k_open)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_close)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        sprockets = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            cx, cy = x + w/2 + roi_offset[0], y + h/2 + roi_offset[1]
            ar = max(w / h, h / w)

            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            if not (self.min_area <= area <= self.max_area):
                continue
            if not (self.ar_min <= ar <= self.ar_max):
                continue

            # Damage-tolerant solidity check
            if solidity < self.solidity_min:
                if self.expected_width and abs(w - self.expected_width) <= self.expected_width * self.width_tol:
                    pass
                else:
                    continue

            sprockets.append((cx, cy, w, h, area))

        sprockets.sort(key=lambda s: s[1])  # top-to-bottom

        # Calibration
        if self.expected_width is None and len(sprockets) >= 1:
            widths = [s[2] for s in sprockets]
            self.expected_width = int(np.median(widths))
            print(f"[SPROCKET] Expected sprocket width set to {self.expected_width}px")

        if self.expected_pitch is None and len(sprockets) >= 2:
            pitches = [sprockets[i+1][1] - sprockets[i][1] for i in range(len(sprockets)-1)]
            self.expected_pitch = int(np.median(pitches))
            print(f"[SPROCKET] Expected sprocket pitch set to {self.expected_pitch}px")

        # Debug
        if debug_prefix is not None:
            cv.imwrite(f"{debug_prefix}_gray.jpg", gray)
            cv.imwrite(f"{debug_prefix}_thresh.jpg", th)
            cv.imwrite(f"{debug_prefix}_mask.jpg", mask)
            dbg = frame_bgr.copy()
            cv.drawContours(dbg, contours, -1, (255, 0, 0), 1)
            for (cx, cy, w, h, area) in sprockets:
                cv.rectangle(dbg, (int(cx-w/2), int(cy-h/2)),
                             (int(cx+w/2), int(cy+h/2)), (0, 255, 0), 2)
                cv.circle(dbg, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv.imwrite(f"{debug_prefix}_annotated.jpg", dbg)

        return sprockets

    def _detect_profile(self, roi, roi_offset, frame_bgr, debug_prefix=None):
        """Profile-based detection pipeline (experimental, now handles multiple sprockets)."""
        H, W = roi.shape[:2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # --- vertical profile through the center column ---
        mid_x = W // 2
        column = gray[:, mid_x]

        col_norm = column / 255.0
        thresh_val = np.max(col_norm) * 0.95
        mask_y = np.where(col_norm > thresh_val)[0]

        sprockets = []

        if len(mask_y) > 0:
            # --- group contiguous runs of bright pixels ---
            gap_thresh = 5  # pixels of vertical gap = new sprocket
            splits = np.where(np.diff(mask_y) > gap_thresh)[0] + 1
            bands = np.split(mask_y, splits)

            for band in bands:
                if len(band) < 5:
                    continue  # ignore tiny noise bands
                y_top, y_bot = band[0], band[-1]
                cy = (y_top + y_bot) / 2 + roi_offset[1]

                # --- horizontal profile at this sprocket center ---
                row = gray[int(cy - roi_offset[1]), :]

                # Threshold relative to the brightest point in row
                peak_val = np.max(row)
                thresh_val = peak_val * 0.95

                # Start from the center column and walk outward
                mid_x = W // 2

                # Walk left
                x_left = mid_x
                while x_left > 0 and row[x_left] > thresh_val:
                 x_left -= 1

                # Walk right
                x_right = mid_x
                while x_right < W - 1 and row[x_right] > thresh_val:
                    x_right += 1

                # Validate width
                if (x_right - x_left) > 5: #simple sanity check
                    cx = (x_left + x_right) / 2 + roi_offset[0]
                    w = x_right - x_left
                    h = y_bot - y_top
                    area = w * h

                    # Aspect ratio (width/height)
                    ar = w / h if h > 0 else 0

                    # Apply aspect ratio filter
                    if 1.0 <= ar <= 2.0:
                        sprockets.append((cx, cy, w, h, area))
                        #print(f"[SPROCKET] Accepted candidate: w={w}, h={h}, ar={ar:.2f}, cy={cy}")

        # --- Debugging images ---
        if debug_prefix is not None:
            dbg = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

            # vertical line at mid_x
            dbg[:, mid_x] = (0, 0, 255)

            if len(mask_y) > 0:
                for y in mask_y:
                    dbg[y, mid_x] = (0, 255, 0)

            for (cx, cy, w, h, area) in sprockets:
                cv.rectangle(dbg, (int(cx - w / 2), int(cy - h / 2)),
                             (int(cx + w / 2), int(cy + h / 2)),
                             (0, 255, 255), 2)
                cv.circle(dbg, (int(cx), int(cy)), 3, (255, 255, 0), -1)

            cv.imwrite(f"{debug_prefix}_profile_dbg.jpg", dbg)

            # save raw profiles for inspection
            col_img = np.tile((col_norm * 255).astype(np.uint8)[:, None], (1, 50))
            #cv.imwrite(f"{debug_prefix}_col_profile.jpg", col_img)

            if len(mask_y) > 0:
                for band in bands:
                    row_y = int((band[0] + band[-1]) / 2)
                    row = gray[row_y, :]
                    row_norm = row / 255.0
                    row_img = np.tile((row_norm * 255).astype(np.uint8)[None, :], (50, 1))
            #        cv.imwrite(f"{debug_prefix}_row_profile_{row_y}.jpg", row_img)

        return sprockets
