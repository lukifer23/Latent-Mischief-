import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_run import verify


class EvidenceTests(unittest.TestCase):
    def fixture(self, folder):
        (folder / 'source').mkdir()
        (folder / 'source/code.py').write_bytes(b'original source\n')
        records = {
            'metadata.json': {'source_hashes': {'code.py': hashlib.sha256(b'original source\n').hexdigest()}},
            'checks.json': {'neutral': True, 'cleanup': True},
            'summary.json': {'completed': True, 'intervention_replies': 2},
            'trial-order.json': [{'candidate': 'L0N1', 'gain': g, 'prompt_id': 'p'} for g in (0, 2)],
            'outputs.json': [{'condition': 'channel-gain', 'candidate': 'L0N1', 'gain': g, 'prompt_id': 'p'} for g in (0, 2)]}
        for name, value in records.items():
            (folder / name).write_text(json.dumps(value), encoding='utf-8')

    def test_modified_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            self.assertEqual(verify(folder)['verified_intervention_trials'], 2)
            (folder / 'source/code.py').write_bytes(b'changed source\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify(folder)

    def test_duplicate_output_does_not_pass_count_only(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            output = json.loads((folder / 'outputs.json').read_text())
            output[1] = output[0]
            (folder / 'outputs.json').write_text(json.dumps(output))
            with self.assertRaisesRegex(ValueError, 'schedule'):
                verify(folder)

    def test_offset_trials_verify_signed_strength_and_reject_swapped_doses(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            records = [{'condition': 'channel-offset', 'candidate': 'L0N1', 'offset': x, 'prompt_id': 'p'} for x in (-.4, .4)]
            for name in ('outputs.json', 'trial-order.json'):
                (folder / name).write_text(json.dumps(records), encoding='utf-8')
            self.assertEqual(verify(folder)['verified_intervention_trials'], 2)
            (folder / 'outputs.json').write_text(json.dumps(list(reversed(records))), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'schedule'):
                verify(folder)
