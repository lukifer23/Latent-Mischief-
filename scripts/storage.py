"""Atomic JSON persistence so interruption does not destroy the last complete record."""
import json
import os
import time
from pathlib import Path
import tempfile


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(9):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(min(.025 * 2**attempt, .4))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
