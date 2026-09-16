"""Display-only diagnostics for Super 8 RAW previews."""

import cv2


def annotate_super8_debug_preview(
    preview_bgr,
    candidates=None,
    selected_y=None,
    phase_trusted=False,
    predicted_y=None,
    selected_unwrapped_y=None,
    predicted_unwrapped_y=None,
    phase_pitch_offset=0,
    phase_wrapped=False,
    registration_error_px=None,
    requested_correction=None,
    limited_correction=None,
    registration_target_y=None,
    crop_center_y=None,
    phase_reason=None,
    crop_source=None,
    recovery_candidate_index=None,
    recovery_distance_px=None,
    recovery_age=0,
    recovery_confirmations=0,
    transport_eligible=False,
):
    """Return a diagnostic copy of a Super 8 preview.

    Values are observations already produced by the detector, phase tracker,
    and crop selector. This helper never mutates the detector input and has no
    effect on capture, registration, or transport control.
    """
    debug = preview_bgr.copy()
    candidates = list(candidates or [])

    for index, candidate in enumerate(candidates):
        cx = float(candidate['center_x'])
        cy = float(candidate['center_y'])
        width = float(candidate['width'])
        height = float(candidate['height'])
        x1 = int(round(cx - width / 2.0))
        y1 = int(round(cy - height / 2.0))
        x2 = int(round(cx + width / 2.0))
        y2 = int(round(cy + height / 2.0))
        is_recovery = bool(
            recovery_candidate_index is not None
            and index == int(recovery_candidate_index)
        )
        is_selected = bool(
            phase_trusted
            and selected_y is not None
            and abs(cy - float(selected_y)) < 1e-6
        )
        color = (0, 220, 0) if is_selected else (0, 165, 255) if is_recovery else (180, 180, 0)
        thickness = 3 if is_selected or is_recovery else 2
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(debug, (int(round(cx)), int(round(cy))), 5, color, -1)
        label = 'SELECTED' if is_selected else 'RECOVERY' if is_recovery else f'{cy:.1f}'
        cv2.putText(
            debug, label, (x1 + 4, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    def marker(y, color, label, thickness=2):
        if y is None:
            return
        y_px = int(round(float(y)))
        cv2.line(debug, (0, y_px), (debug.shape[1] - 1, y_px), color, thickness)
        cv2.putText(
            debug, label, (8, max(16, y_px - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
        )

    marker(predicted_y, (255, 0, 255), 'phase predicted Y', 2)
    marker(registration_target_y, (0, 255, 255), 'registration target Y', 2)
    marker(crop_center_y, (255, 150, 0), 'crop center Y', 1)

    phase_label = 'trusted' if phase_trusted else (
        phase_reason.replace('_', ' ') if phase_reason and phase_reason.startswith('recovery_')
        else 'untrusted'
    )
    lines = [
        f'Phase: {phase_label}' + (f' ({phase_reason})' if phase_reason else ''),
        f'Candidates: {len(candidates)}',
    ]
    if selected_y is not None:
        label = 'Selected Y' if phase_trusted else 'Candidate Y'
        lines.append(f'{label}: {float(selected_y):.1f}')
    if predicted_y is not None:
        lines.append(f'Predicted Y: {float(predicted_y):.1f}')
    if selected_unwrapped_y is not None:
        lines.append(f'Selected unwrapped: {float(selected_unwrapped_y):.1f}')
    if predicted_unwrapped_y is not None:
        lines.append(f'Predicted unwrapped: {float(predicted_unwrapped_y):.1f}')
    if phase_pitch_offset or phase_wrapped:
        lines.append(f'Pitch offset: {int(phase_pitch_offset):+d} (wrapped)')
    if registration_target_y is not None:
        lines.append(f'Registration target: {float(registration_target_y):.1f}')
    if phase_trusted and selected_y is not None and registration_target_y is not None:
        lines.append(f'Error: {float(registration_target_y) - float(selected_y):+.1f} px')
    if crop_center_y is not None:
        lines.append(f'Crop center: {float(crop_center_y):.1f}')
    if crop_source:
        lines.append(f'Crop source: {crop_source}')
    if registration_error_px is not None:
        lines.append(f'Registration error: {float(registration_error_px):+.1f} px')
    if requested_correction is not None or limited_correction is not None:
        lines.append(
            f'Correction: {requested_correction if requested_correction is not None else "n/a"}'
            f'/{limited_correction if limited_correction is not None else "n/a"}'
        )
    if phase_reason and phase_reason.startswith('recovery_'):
        lines.append(f'Recovery: {recovery_age}/{recovery_confirmations}')
        if recovery_distance_px is not None:
            lines.append(f'Recovery distance: {float(recovery_distance_px):.1f} px')
    lines.append(f'Transport eligible: {"yes" if transport_eligible else "no"}')

    x, y = 8, 20
    line_height = 18
    box_bottom = y + line_height * len(lines) + 5
    cv2.rectangle(
        debug, (2, 2), (min(debug.shape[1] - 3, 300), box_bottom),
        (0, 0, 0), -1,
    )
    for line in lines:
        cv2.putText(
            debug, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += line_height
    return debug
