class RegistrationTracker:
    def __init__(self, expected_sprocket_pitch_px, max_jump_px=40.0, smoothing_alpha=0.8):
        self.expected_sprocket_pitch_px = float(expected_sprocket_pitch_px) if expected_sprocket_pitch_px is not None else None
        self.max_jump_px = float(max_jump_px)
        self.smoothing_alpha = max(0.0, min(1.0, float(smoothing_alpha)))
        self.reset(expected_sprocket_pitch_px=expected_sprocket_pitch_px)

    def reset(self, expected_sprocket_pitch_px=None):
        if expected_sprocket_pitch_px is not None:
            self.expected_sprocket_pitch_px = float(expected_sprocket_pitch_px)

        self.last_good_registration_y = None
        self.smoothed_registration_y = None
        self.last_good_mode = None
        self.frame_index = 0
        self.failure_count = 0

    def predicted_y(self):
        if self.smoothed_registration_y is not None:
            return float(self.smoothed_registration_y)
        if self.last_good_registration_y is not None:
            return float(self.last_good_registration_y)
        return None

    def update(self, raw_registration_y, raw_registration_mode, frame_index=None, expected_sprocket_pitch_px=None):
        if expected_sprocket_pitch_px is not None:
            self.expected_sprocket_pitch_px = float(expected_sprocket_pitch_px)

        self.frame_index = int(frame_index) if frame_index is not None else (self.frame_index + 1)
        raw_mode = raw_registration_mode if raw_registration_mode in ("pair", "single") else "none"
        raw_y = float(raw_registration_y) if raw_registration_y is not None else None
        predicted_y = self.predicted_y()

        selected_y = None
        selected_mode = raw_mode
        accepted = False
        estimated = False
        rejected = False

        if raw_mode == "pair" and raw_y is not None:
            selected_y = raw_y
            if predicted_y is None or self._is_acceptable_jump(selected_y, predicted_y):
                accepted = True
                selected_mode = "pair"
            else:
                rejected = True

        elif raw_mode == "single" and raw_y is not None:
            candidates = self._single_mode_candidates(raw_y)
            if predicted_y is None:
                selected_y = candidates[0]
            else:
                selected_y = min(candidates, key=lambda candidate: abs(candidate - predicted_y))

            if predicted_y is None or self._is_acceptable_jump(selected_y, predicted_y):
                accepted = True
                selected_mode = "single"
            else:
                rejected = True

        if accepted and selected_y is not None:
            self.failure_count = 0
            self.last_good_registration_y = float(selected_y)
            self.last_good_mode = selected_mode

            if self.smoothed_registration_y is None:
                self.smoothed_registration_y = float(selected_y)
            else:
                self.smoothed_registration_y = (
                    (self.smoothing_alpha * float(self.smoothed_registration_y))
                    + ((1.0 - self.smoothing_alpha) * float(selected_y))
                )
        else:
            self.failure_count += 1
            fallback_y = predicted_y if predicted_y is not None else self.last_good_registration_y
            if fallback_y is not None:
                selected_y = float(fallback_y)
                estimated = True
                selected_mode = self.last_good_mode or raw_mode
            else:
                selected_y = None

        crop_anchor_y = None
        if self.smoothed_registration_y is not None:
            crop_anchor_y = float(self.smoothed_registration_y)
        elif selected_y is not None:
            crop_anchor_y = float(selected_y)

        registration_error_px = None
        if selected_y is not None and predicted_y is not None:
            registration_error_px = float(selected_y) - float(predicted_y)

        return {
            "frame_index": int(self.frame_index),
            "raw_registration_y": raw_y,
            "raw_registration_mode": raw_mode,
            "selected_registration_y": float(selected_y) if selected_y is not None else None,
            "smoothed_registration_y": float(self.smoothed_registration_y) if self.smoothed_registration_y is not None else None,
            "predicted_registration_y": float(predicted_y) if predicted_y is not None else None,
            "registration_error_px": registration_error_px,
            "registration_estimated": bool(estimated),
            "registration_rejected": bool(rejected),
            "registration_accepted": bool(accepted),
            "selected_mode": selected_mode,
            "stable_registration_y": crop_anchor_y,
            "failure_count": int(self.failure_count),
            "last_good_mode": self.last_good_mode,
        }

    def _single_mode_candidates(self, raw_y):
        candidates = [float(raw_y)]
        if self.expected_sprocket_pitch_px is None or self.expected_sprocket_pitch_px <= 0:
            return candidates

        half_pitch = float(self.expected_sprocket_pitch_px) / 2.0
        return [
            float(raw_y),
            float(raw_y) + half_pitch,
            float(raw_y) - half_pitch,
        ]

    def _is_acceptable_jump(self, candidate_y, predicted_y):
        if predicted_y is None:
            return True
        return abs(float(candidate_y) - float(predicted_y)) <= float(self.max_jump_px)