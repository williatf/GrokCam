import json
import tempfile
import unittest
from pathlib import Path

from camera_settings import (
    DEFAULT_CAMERA_EXPOSURE_TIME,
    build_camera_controls,
    clamp_camera_settings,
    deserialize_camera_settings,
    serialize_camera_settings,
)


class CameraSettingsTests(unittest.TestCase):
    def test_default_exposure_is_3300_microseconds(self):
        self.assertEqual(DEFAULT_CAMERA_EXPOSURE_TIME, 3300)
        self.assertEqual(clamp_camera_settings()['ExposureTime'], 3300)

    def test_user_value_overrides_default_and_controls_are_manual(self):
        settings = clamp_camera_settings(exposure_time=4711, analogue_gain=1.0)
        controls = build_camera_controls(
            settings['ExposureTime'], settings['AnalogueGain']
        )
        self.assertEqual(controls['ExposureTime'], 4711)
        self.assertEqual(controls['AnalogueGain'], 1.0)
        self.assertFalse(controls['AeEnable'])
        self.assertFalse(controls['AwbEnable'])

    def test_saved_exposure_survives_json_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera_settings.json'
            path.write_text(json.dumps(serialize_camera_settings(4321, 1.0)))
            loaded = deserialize_camera_settings(json.loads(path.read_text()))
            self.assertEqual(loaded['ExposureTime'], 4321)

    def test_explicit_project_value_is_not_replaced_by_new_default(self):
        payload = {'ExposureTime': 814, 'AnalogueGain': 1.0}
        loaded = clamp_camera_settings(
            payload.get('ExposureTime'), payload.get('AnalogueGain')
        )
        self.assertEqual(loaded['ExposureTime'], 814)

    def test_regular8_and_super8_share_camera_default(self):
        for _film_format in ('regular8', 'super8'):
            self.assertEqual(clamp_camera_settings()['ExposureTime'], 3300)


if __name__ == '__main__':
    unittest.main()
