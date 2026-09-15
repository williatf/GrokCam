import json
import os
import tempfile
import unittest

from film_calibration import (
    FILM_FORMAT_REGULAR8,
    FILM_FORMAT_SUPER8,
    load_film_calibration,
    normalize_film_format,
    resolve_raw_preview_transport,
)


class FilmCalibrationTests(unittest.TestCase):
    def write_json(self, directory, filename, payload):
        path = os.path.join(directory, filename)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
        return path

    def test_missing_project_format_defaults_to_regular8(self):
        self.assertEqual(normalize_film_format(None), FILM_FORMAT_REGULAR8)

    def test_legacy_calibration_is_regular8_and_keeps_277(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_json(directory, 'calibration.json', {
                'calibration_resolution': [2028, 1520],
                'sprocket_pitch_px': 781.5,
                'steps_per_pitch': 277,
                'steps_per_px': 0.3544465770953295,
            })
            calibration = load_film_calibration(FILM_FORMAT_REGULAR8, directory)
            resolved = resolve_raw_preview_transport(calibration, (760, 570))
            self.assertTrue(calibration.legacy_unnamespaced)
            self.assertEqual(resolved['steps_per_pitch'], 277)
            self.assertEqual(resolved['pixels_per_step_source'], 'formal_steps_per_px')

    def test_super8_uses_303_without_regular8_geometry_inheritance(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_json(directory, 'calibration.json', {
                'calibration_resolution': [2028, 1520],
                'sprocket_pitch_px': 781.5,
                'steps_per_pitch': 277,
                'steps_per_px': 0.3544465770953295,
            })
            self.write_json(directory, 'calibration.super8.json', {
                'film_format': 'super8',
                'status': 'provisional',
                'steps_per_pitch': 303,
                'transport_observation': {
                    'raw_preview_size': [760, 570],
                    'pixels_per_step': 1.0522,
                },
            })
            calibration = load_film_calibration(FILM_FORMAT_SUPER8, directory)
            resolved = resolve_raw_preview_transport(calibration, (760, 570))
            self.assertFalse(calibration.legacy_unnamespaced)
            self.assertEqual(resolved['steps_per_pitch'], 303)
            self.assertIsNone(resolved['preview_sprocket_pitch_px'])
            self.assertEqual(
                resolved['pixels_per_step_source'],
                'observational_preview_measurement',
            )

    def test_unnamespaced_calibration_cannot_be_used_as_super8(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_json(directory, 'calibration.super8.json', {
                'steps_per_pitch': 277,
            })
            with self.assertRaises(ValueError):
                load_film_calibration(FILM_FORMAT_SUPER8, directory)


if __name__ == '__main__':
    unittest.main()
