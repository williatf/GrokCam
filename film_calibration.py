"""Film-format-specific calibration loading and RAW preview geometry."""

from dataclasses import dataclass
import json
import os


FILM_FORMAT_REGULAR8 = 'regular8'
FILM_FORMAT_SUPER8 = 'super8'
SUPPORTED_FILM_FORMATS = (FILM_FORMAT_REGULAR8, FILM_FORMAT_SUPER8)

CALIBRATION_FILENAMES = {
    FILM_FORMAT_REGULAR8: 'calibration.json',
    FILM_FORMAT_SUPER8: 'calibration.super8.json',
}


def normalize_film_format(value):
    normalized = str(value or FILM_FORMAT_REGULAR8).strip().lower().replace('_', '')
    if normalized not in SUPPORTED_FILM_FORMATS:
        raise ValueError(
            f"Unsupported film_format {value!r}; expected one of {', '.join(SUPPORTED_FILM_FORMATS)}"
        )
    return normalized


@dataclass(frozen=True)
class FilmCalibration:
    film_format: str
    path: str
    values: dict
    legacy_unnamespaced: bool = False

    @property
    def source_name(self):
        return os.path.basename(self.path)

    @property
    def status(self):
        if self.values.get('status'):
            return str(self.values['status'])
        return 'legacy' if self.legacy_unnamespaced else 'calibrated'


def calibration_path_for_format(film_format, base_dir='.'):
    normalized = normalize_film_format(film_format)
    return os.path.join(base_dir, CALIBRATION_FILENAMES[normalized])


def load_film_calibration(film_format, base_dir='.'):
    normalized = normalize_film_format(film_format)
    path = calibration_path_for_format(normalized, base_dir=base_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {normalized} calibration file: {path}"
        )

    with open(path, 'r', encoding='utf-8') as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {path}")

    declared_format = values.get('film_format')
    legacy_unnamespaced = declared_format is None
    if declared_format is not None and normalize_film_format(declared_format) != normalized:
        raise ValueError(
            f"Calibration file {path} declares film_format={declared_format!r}, "
            f"but {normalized!r} was requested"
        )
    if legacy_unnamespaced and normalized != FILM_FORMAT_REGULAR8:
        raise ValueError(
            f"Unnamespaced calibration is Regular 8 calibration and cannot be used for {normalized}"
        )

    return FilmCalibration(
        film_format=normalized,
        path=path,
        values=values,
        legacy_unnamespaced=legacy_unnamespaced,
    )


def resolve_raw_preview_transport(calibration, preview_size):
    """Resolve transport values in the RAW preview coordinate system.

    Formal calibration geometry is preferred when present. A partial
    calibration may instead carry an explicitly preview-scoped observational
    pixels/step measurement. That measurement is not promoted to steps_per_px.
    """
    values = calibration.values
    steps_per_pitch = values.get('steps_per_pitch')
    if steps_per_pitch is None:
        raise ValueError(
            f"{calibration.film_format} calibration is missing steps_per_pitch"
        )
    nominal_steps = int(round(float(steps_per_pitch)))

    preview_width, preview_height = (int(preview_size[0]), int(preview_size[1]))
    preview_pitch = None
    pixels_per_step = None
    pixels_per_step_source = None

    calibration_resolution = values.get('calibration_resolution')
    if calibration_resolution and len(calibration_resolution) == 2:
        calibration_height = float(calibration_resolution[1])
        if calibration_height > 0:
            preview_scale = preview_height / calibration_height
            if values.get('sprocket_pitch_px') is not None:
                preview_pitch = float(values['sprocket_pitch_px']) * preview_scale
            if values.get('steps_per_px') is not None:
                steps_per_px = float(values['steps_per_px'])
                if steps_per_px <= 0:
                    raise ValueError('steps_per_px must be positive')
                pixels_per_step = (1.0 / steps_per_px) * preview_scale
                pixels_per_step_source = 'formal_steps_per_px'

    observation = values.get('transport_observation')
    if pixels_per_step is None and isinstance(observation, dict):
        observation_size = observation.get('raw_preview_size')
        observation_pixels_per_step = observation.get('pixels_per_step')
        if observation_size is not None and observation_pixels_per_step is not None:
            if tuple(int(value) for value in observation_size) != (preview_width, preview_height):
                raise ValueError(
                    'transport_observation raw_preview_size does not match active RAW preview size'
                )
            pixels_per_step = float(observation_pixels_per_step)
            if pixels_per_step <= 0:
                raise ValueError('transport_observation pixels_per_step must be positive')
            pixels_per_step_source = 'observational_preview_measurement'

    return {
        'steps_per_pitch': nominal_steps,
        'preview_sprocket_pitch_px': preview_pitch,
        'preview_pixels_per_step': pixels_per_step,
        'pixels_per_step_source': pixels_per_step_source,
    }
