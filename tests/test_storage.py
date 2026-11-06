import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import storage


class RecordingTests(unittest.TestCase):
    def test_failed_replacement_preserves_prior_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'outputs.json'
            storage.write_json(path, {'complete': ['prior output']})
            with patch.object(storage.os, 'replace', side_effect=PermissionError('simulated interruption')):
                with self.assertRaises(PermissionError):
                    storage.write_json(path, {'complete': ['new output']})
            self.assertEqual(json.loads(path.read_text()), {'complete': ['prior output']})
            self.assertEqual([p.name for p in path.parent.iterdir()], ['outputs.json'])

    def test_invalid_nonfinite_metric_does_not_replace_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'outputs.json'
            storage.write_json(path, {'score': 0})
            with self.assertRaises(ValueError):
                storage.write_json(path, {'score': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'score': 0})
