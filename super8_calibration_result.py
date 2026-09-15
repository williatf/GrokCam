"""Validation and proposal construction for measured Super 8 calibration."""

import time


def build_super8_client_summary(result, target_transitions):
    """Add fields required by the existing Regular 8 GUI summary decoder."""
    measurements = result.get('measurements') or {}
    pitch = measurements.get('sprocket_pitch_px', {})
    area = measurements.get('area', {})
    transitions = measurements.get('steps_per_pitch', {})
    return {
        **result,
        'pitch_mean': pitch.get('mean'),
        'pitch_min': pitch.get('min'),
        'pitch_max': pitch.get('max'),
        'pitch_std': pitch.get('std'),
        'area_mean': area.get('mean'),
        'area_min': area.get('min'),
        'area_max': area.get('max'),
        'area_std': area.get('std'),
        'valid_samples': transitions.get('count', 0),
        'total_samples': result.get('target_transitions', target_transitions),
    }


def build_super8_calibration_proposal(
    result,
    exposure_result,
    calibration_resolution,
    minimum_transitions=5,
    max_cross_check_percent=12.0,
    timestamp=None,
):
    measurements = result.get('measurements') or {}
    transitions = measurements.get('steps_per_pitch', {})
    pitch = measurements.get('sprocket_pitch_px', {})
    steps_per_px = measurements.get('steps_per_px', {})
    area = measurements.get('area', {})
    accepted = int(transitions.get('count') or 0)
    if accepted < minimum_transitions:
        return None, False, 'need_at_least_5_accepted_transitions'
    if pitch.get('median') is None:
        return None, False, 'missing_sprocket_pitch_measurements'
    if steps_per_px.get('median') is None:
        return None, False, 'missing_steps_per_px_measurements'
    if area.get('median') is None:
        return None, False, 'missing_sprocket_area_measurements'

    cross_check = measurements.get('cross_check')
    if cross_check is None:
        return None, False, 'missing_independent_cross_check'
    if cross_check['difference_percent'] > max_cross_check_percent:
        return None, False, 'cross_check_disagreement_too_large'

    proposal = {
        'calibration_version': 2,
        'film_format': 'super8',
        'status': 'calibrated',
        'source': 'production_super8_physical_track_calibration',
        'timestamp': timestamp or time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'calibration_resolution': list(calibration_resolution),
        'exposure_time': int(exposure_result['exposure_time']),
        'gain': float(exposure_result['gain']),
        'steps_per_pitch': int(round(transitions['median'])),
        'steps_per_px': float(steps_per_px['median']),
        'sprocket_pitch_px': float(pitch['median']),
        'sprocket_area_min': int(round(float(area['median']) * 0.88)),
        'sprocket_area_max': int(round(float(area['median']) * 1.12)),
        'quality': {
            'successful_transitions': accepted,
            'rejected_transitions': int(result.get('rejected_transitions', 0)),
            'steps_per_pitch': transitions,
            'sprocket_pitch_px': pitch,
            'steps_per_px': steps_per_px,
            'detector_confidence': measurements.get('confidence'),
            'sprocket_area': area,
            'ambiguous_track_count': measurements.get('ambiguous_track_count', 0),
            'rejected_track_count': measurements.get('rejected_track_count', 0),
            'detector_loss_frames': measurements.get('detector_loss_frames', 0),
            'cross_check': cross_check,
            'steps_per_pitch_measurements': measurements.get(
                'steps_per_pitch_values', []
            ),
            'sprocket_pitch_px_measurements': measurements.get(
                'sprocket_pitch_px_values', []
            ),
            'steps_per_px_measurements': measurements.get(
                'steps_per_px_values', []
            ),
        },
    }
    return proposal, True, None
