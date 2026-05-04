class RegistrationTracker:
    def __init__(self, expected_sprocket_pitch_px, max_jump_px=40.0, smoothing_alpha=0.8):
        self.expected_sprocket_pitch_px = float(expected_sprocket_pitch_px) if expected_sprocket_pitch_px is not None else None
        self.max_jump_px = float(max_jump_px)
        self.smoothing_alpha = max(0.0, min(1.0, float(smoothing_alpha)))
        self.reset(expected_sprocket_pitch_px=expected_sprocket_pitch_px)

    def reset(self, expected_sprocket_pitch_px=None, baseline_registration_y=None):
        if expected_sprocket_pitch_px is not None:
            self.expected_sprocket_pitch_px = float(expected_sprocket_pitch_px)

        self.baseline_registration_y = float(baseline_registration_y) if baseline_registration_y is not None else None
        self.last_good_registration_y = float(baseline_registration_y) if baseline_registration_y is not None else None
        self.frame_index = 0
        self.failure_count = 0
        self.last_selected_source = None

    def predicted_y(self):
        if self.last_good_registration_y is not None:
            return float(self.last_good_registration_y)
        if self.baseline_registration_y is None:
            return None
        return float(self.baseline_registration_y)

    def update(self, raw_registration_y, raw_registration_mode, frame_index=None, expected_sprocket_pitch_px=None):
        if expected_sprocket_pitch_px is not None:
            self.expected_sprocket_pitch_px = float(expected_sprocket_pitch_px)

        self.frame_index = int(frame_index) if frame_index is not None else (self.frame_index + 1)
        raw_mode = raw_registration_mode if raw_registration_mode in ("pair", "single") else "none"
        raw_y = float(raw_registration_y) if raw_registration_y is not None else None
        reference_y = self.predicted_y()
        raw_pair_midpoint_y = raw_y if raw_mode == "pair" else None
        single_sprocket_y = raw_y if raw_mode == "single" else None
        selected_y = None
        selected_source = None
        registration_error_px = None
        estimated = False
        rejected = False

        if raw_pair_midpoint_y is not None:
            if self.baseline_registration_y is None:
                self.baseline_registration_y = float(raw_pair_midpoint_y)
            self.last_good_registration_y = float(raw_pair_midpoint_y)
            selected_y = float(raw_pair_midpoint_y)
            selected_source = "pair_actual"
            self.failure_count = 0
            if reference_y is not None:
                registration_error_px = float(raw_pair_midpoint_y) - float(reference_y)

        elif single_sprocket_y is not None:
            if reference_y is not None:
                candidates = self._single_mode_candidates(single_sprocket_y)
                if candidates:
                    estimated_midpoint = min(candidates, key=lambda candidate: abs(candidate - reference_y))
                    registration_error_px = float(estimated_midpoint) - float(reference_y)
                    if self._is_acceptable_jump(estimated_midpoint, reference_y):
                        selected_y = float(estimated_midpoint)
                        selected_source = "single_estimated"
                        estimated = True
                        self.last_good_registration_y = float(estimated_midpoint)
                        self.failure_count = 0
                    else:
                        selected_y = float(reference_y)
                        selected_source = "rejected_single"
                        rejected = True
                        estimated = True
                        self.failure_count += 1
                else:
                    selected_y = float(reference_y)
                    selected_source = "rejected_single"
                    rejected = True
                    estimated = True
                    self.failure_count += 1
            else:
                selected_y = None
                selected_source = None
                self.failure_count += 1

        else:
            if reference_y is not None:
                selected_y = float(reference_y)
                selected_source = "held_last_good"
                estimated = True
            else:
                selected_y = None
                selected_source = None
            self.failure_count += 1

        self.last_selected_source = selected_source

        return {
            "frame_index": int(self.frame_index),
            "raw_registration_y": raw_y,
            "raw_registration_mode": raw_mode,
            "raw_pair_midpoint_y": raw_pair_midpoint_y,
            "single_sprocket_y": single_sprocket_y,
            "baseline_registration_y": float(self.baseline_registration_y) if self.baseline_registration_y is not None else None,
            "selected_registration_y": float(selected_y) if selected_y is not None else None,
            "last_good_registration_y": float(self.last_good_registration_y) if self.last_good_registration_y is not None else None,
            "registration_error_px": registration_error_px,
            "registration_estimated": bool(estimated),
            "registration_rejected": bool(rejected),
            "selected_source": selected_source,
            "stable_registration_y": float(selected_y) if selected_y is not None else reference_y,
            "failure_count": int(self.failure_count),
        }

    def _single_mode_candidates(self, raw_y):
        if self.expected_sprocket_pitch_px is None or self.expected_sprocket_pitch_px <= 0:
            return []

        half_pitch = float(self.expected_sprocket_pitch_px) / 2.0
        return [
            float(raw_y) + half_pitch,
            float(raw_y) - half_pitch,
        ]

    def _is_acceptable_jump(self, candidate_y, predicted_y):
        if predicted_y is None:
            return True
        return abs(float(candidate_y) - float(predicted_y)) <= float(self.max_jump_px)