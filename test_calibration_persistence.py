import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from calibration_persistence import save_calibration_atomic


class CalibrationPersistenceTests(unittest.TestCase):
    def test_atomic_save_replaces_destination_and_preserves_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'calibration.super8.json'
            destination.write_text('{"status": "provisional"}\n')

            with mock.patch(
                'calibration_persistence.os.replace',
                wraps=__import__('os').replace,
            ) as replace:
                backup = save_calibration_atomic(
                    {'film_format': 'super8', 'status': 'calibrated'},
                    destination,
                )

            replace.assert_called_once()
            self.assertEqual(json.loads(destination.read_text())['status'], 'calibrated')
            self.assertEqual(json.loads(Path(backup).read_text())['status'], 'provisional')
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
