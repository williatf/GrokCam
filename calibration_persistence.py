"""Safe persistence shared by film-format calibration strategies."""

import json
import os
import shutil
import tempfile
import time


def save_calibration_atomic(calibration, destination):
    """Back up an existing calibration and atomically replace its JSON file."""
    destination = os.path.abspath(destination)
    directory = os.path.dirname(destination) or '.'
    os.makedirs(directory, exist_ok=True)

    backup_path = None
    if os.path.exists(destination):
        root, ext = os.path.splitext(destination)
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        backup_path = f"{root}.backup.{timestamp}{ext or '.json'}"
        shutil.copy2(destination, backup_path)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix='.tmp',
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(calibration, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

    return backup_path
