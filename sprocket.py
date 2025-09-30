import cv2 as cv
import numpy as np

class SprocketDetector:
    def __init__(self, side="left", auto_roi=0.25,
                 min_area=3000, max_area=6000,
                 ar_min=1.2, ar_max=1.6,
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
        self.method = method
        self.inv = inv

        # Calibration hints
        self.expected_width = None
        self.expected_pitch = None
        self.width_tol = 0.25
        self.pitch_tol = 12
        self.expected_ar = 1.5  # expected aspect ratio (width/height)

    def update_pitch(self, pitch_px):
        """Update expected sprocket pitch from external measurement."""
        if self.expected_pitch is None:
            self.expected_pitch = int(pitch_px)
            print(f"[SPROCKET] Initialized expected_pitch={self.expected_pitch}px")
        else:
            # smooth update (running average) for stability
            self.expected_pitch = int(0.8 * self.expected_pitch + 0.2 * pitch_px)
            print(f"[SPROCKET] Updated expected_pitch={self.expected_pitch}px")

    def detect(self, frame_bgr, debug_prefix=None, mode="profile"):
        """Always use profile-based detection now."""
        H, W = frame_bgr.shape[:2]
        strip_w = int(W * self.auto_roi)

        if self.side == "left":
            roi = frame_bgr[:, :strip_w]
            roi_offset = (0, 0)
        else:
            roi = frame_bgr[:, -strip_w:]
            roi_offset = (W - strip_w, 0)

        return self._detect_profile(roi, roi_offset, frame_bgr, debug_prefix)

    def _detect_profile(self, roi, roi_offset, frame_bgr, debug_prefix=None):
        """
        Profile-based sprocket detection (damage-tolerant).
        - Ignores half-visible sprockets at top/bottom edges.
        - If sprocket AR is out of range, reconstruct using expected aspect ratio.
        - Uses row profiles to measure width near sprocket midline.
        """
        H, W = roi.shape[:2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # vertical profile down center
        mid_x = W // 2
        column = gray[:, mid_x]
        col_norm = column / 255.0
        thresh_val = np.max(col_norm) * 0.95
        mask_y = np.where(col_norm > thresh_val)[0]

        sprockets = []
        bands = []
        if len(mask_y) > 0:
            gap_thresh = 5
            splits = np.where(np.diff(mask_y) > gap_thresh)[0] + 1
            bands = np.split(mask_y, splits)

        # Guard band: ignore sprockets too close to edges
        top_guard = int(0.005 * H)       # top 0.5% of frame
        bottom_guard = int(H * (1.0 - 0.005))  # bottom 0.5% of frame

        for band in bands:
            if len(band) < 5:
                continue
            y_top, y_bot = band[0], band[-1]

            # Reject partial sprockets at edges
            if y_top <= top_guard or y_bot >= bottom_guard:
                print(f"[DEBUG] Rejecting partial band at edges: y_range={y_top}-{y_bot}")
                continue

            # ... continue with bounding box, width/height, aspect ratio checks


            print(f"  y_range: {band[0]}–{band[-1]}, size={len(band)}")

            h = y_bot - y_top
            cy = (y_top + y_bot) / 2 + roi_offset[1]

            # horizontal row near mid-sprocket
            row_y = int((y_top + y_bot) / 2)
            row = gray[row_y, :]

            peak_val = np.max(row)
            thresh_row = peak_val * 0.95

            # Walk left/right from mid_x
            x_left = mid_x
            while x_left > 0 and row[x_left] > thresh_row:
                x_left -= 1
            x_right = mid_x
            while x_right < W - 1 and row[x_right] > thresh_row:
                x_right += 1

            w = x_right - x_left
            cx = (x_left + x_right) / 2 + roi_offset[0]
            area = w * h

            # aspect ratio check
            ar = w / h if h > 0 else 0

            if not (self.ar_min <= ar <= self.ar_max):
                # Lock top edge, recalc bottom from expected aspect ratio
                h = int(max(1, w / self.expected_ar))
                y_bot = y_top + h
                cy = (y_top + y_bot) / 2 + roi_offset[1]
                torn = True
            else:
                torn = False

            # Accept if corrected or valid
            if torn:
                sprockets.append((cx, cy, w, h, area))
            else:
                if self.ar_min <= ar <= self.ar_max:
                    sprockets.append((cx, cy, w, h, area))

        # Calibration update
        if self.expected_pitch is None and len(sprockets) >= 2:
            pitches = [sprockets[i+1][1] - sprockets[i][1] for i in range(len(sprockets)-1)]
            self.expected_pitch = int(np.median(pitches))
            print(f"[SPROCKET] Calibrated expected_pitch={self.expected_pitch}px")

        if self.expected_width is None and len(sprockets) >= 1:
            widths = [s[2] for s in sprockets]
            self.expected_width = int(np.median(widths))
            print(f"[SPROCKET] Calibrated expected_width={self.expected_width}px")

        if self.expected_ar is None and len(sprockets) >= 1:
            ars = [s[2] / s[3] for s in sprockets if s[3] > 0]
            if ars:
                self.expected_ar = float(np.median(ars))
                print(f"[SPROCKET] Calibrated expected_ar={self.expected_ar:.3f}")

        # Debug
        if debug_prefix is not None:
            dbg = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            dbg[:, mid_x] = (0, 0, 255)
            for (cx, cy, w, h, area) in sprockets:
                color = (255, 0, 0) if torn else (0, 255, 255)  # blue if corrected
                cv.rectangle(dbg, (int(cx - w/2), int(cy - h/2)),
                            (int(cx + w/2), int(cy + h/2)), color, 2)
                cv.circle(dbg, (int(cx), int(cy)), 3, (255, 0, 0), -1)
            cv.imwrite(f"{debug_prefix}_profile_dbg.jpg", dbg)

        return sprockets
