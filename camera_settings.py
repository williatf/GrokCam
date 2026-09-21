"""Pure camera-setting helpers shared by the Picamera2 application and tests."""

DEFAULT_CAMERA_EXPOSURE_TIME = 3300
CAMERA_EXPOSURE_MIN = 100
CAMERA_EXPOSURE_MAX = 50000
CAMERA_GAIN_MIN = 1.0
CAMERA_GAIN_MAX = 16.0


def clamp_camera_settings(
    exposure_time=None,
    analogue_gain=None,
    *,
    default_exposure=DEFAULT_CAMERA_EXPOSURE_TIME,
    default_gain=1.0,
):
    """Normalize user/project settings without changing an explicit value."""
    requested_exposure = (
        default_exposure
        if exposure_time is None
        else int(round(float(exposure_time)))
    )
    requested_gain = default_gain if analogue_gain is None else float(analogue_gain)
    clamped_exposure = max(
        CAMERA_EXPOSURE_MIN,
        min(CAMERA_EXPOSURE_MAX, requested_exposure),
    )
    clamped_gain = max(CAMERA_GAIN_MIN, min(CAMERA_GAIN_MAX, requested_gain))
    return {
        'ExposureTime': int(clamped_exposure),
        'AnalogueGain': float(clamped_gain),
        'AeEnable': False,
        'AwbEnable': False,
        'exposure_clamped': clamped_exposure != requested_exposure,
        'gain_clamped': clamped_gain != requested_gain,
    }


def build_camera_controls(exposure_time, analogue_gain, colour_gains=None):
    """Build the exact manual controls passed to Picamera2."""
    controls = {
        'ExposureTime': int(exposure_time),
        'AnalogueGain': float(analogue_gain),
        'AeEnable': False,
        'AwbEnable': False,
    }
    if colour_gains is not None:
        controls['ColourGains'] = (float(colour_gains[0]), float(colour_gains[1]))
    return controls


def serialize_camera_settings(exposure_time, analogue_gain, colour_gains=None, timestamp=None):
    """Return the on-disk representation used by project camera_settings.json."""
    payload = {
        'ExposureTime': int(exposure_time),
        'AnalogueGain': float(analogue_gain),
        'AeEnable': False,
        'AwbEnable': False,
    }
    if colour_gains is not None:
        payload['ColourGains'] = [float(colour_gains[0]), float(colour_gains[1])]
    if timestamp is not None:
        payload['timestamp'] = timestamp
    return payload


def deserialize_camera_settings(payload, *, default_gain=1.0):
    """Load a project payload while retaining an explicit ExposureTime."""
    clamped = clamp_camera_settings(
        payload.get('ExposureTime'),
        payload.get('AnalogueGain'),
        default_gain=default_gain,
    )
    return {
        **clamped,
        'ColourGains': payload.get('ColourGains'),
        'source': payload.get('source', 'manual'),
        'timestamp': payload.get('timestamp'),
    }
