"""Film-format calibration routing and pending-proposal validation."""

from dataclasses import dataclass
import os

from film_calibration import calibration_path_for_format, normalize_film_format


CALIBRATION_MODES = {
    'regular8': 'regular8_pair',
    'super8': 'super8_physical_track',
}


def calibration_route(film_format, base_dir='.'):
    normalized = normalize_film_format(film_format)
    return {
        'film_format': normalized,
        'mode': CALIBRATION_MODES[normalized],
        'destination': calibration_path_for_format(normalized, base_dir),
    }


@dataclass(frozen=True)
class PendingCalibrationProposal:
    project_path: str
    film_format: str
    destination: str
    mode: str
    calibration_version: int
    values: dict

    @classmethod
    def create(cls, project_path, film_format, destination, mode, values):
        return cls(
            project_path=os.path.abspath(project_path),
            film_format=normalize_film_format(film_format),
            destination=os.path.abspath(destination),
            mode=str(mode),
            calibration_version=int(values.get('calibration_version', 0)),
            values=dict(values),
        )

    def validate(self, project_path, film_format, destination, mode):
        if os.path.abspath(project_path) != self.project_path:
            return False, 'active_project_changed_since_calibration_sweep'
        if normalize_film_format(film_format) != self.film_format:
            return False, 'film_format_changed_since_calibration_sweep'
        if os.path.abspath(destination) != self.destination:
            return False, 'calibration_destination_mismatch'
        if str(mode) != self.mode:
            return False, 'calibration_mode_mismatch'
        return True, None
