import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from phase_design import blind_pack, load_plan, task_score
from phase_report import verify_prefix


class PhaseDesignTests(unittest.TestCase):
    def test_plan_is_complete_reproducible_and_does_not_import_torch(self):
        first = load_plan(ROOT, ROOT / 'configs/lfm-phase-timing.json')
        self.assertEqual(first, load_plan(ROOT, ROOT / 'configs/lfm-phase-timing.json'))
        self.assertEqual(first['generations'], 54)
        self.assertEqual(first['maximum_generated_tokens'], 2592)
        probes = [x for x in first['schedule'] if x['record_phase'] == 'probe']
        self.assertEqual(len(probes), 48)
        self.assertEqual(len({(x['condition'], x['prompt_id']) for x in probes}), 48)
        code = "import sys;sys.path.insert(0,'scripts');from phase_design import load_plan;load_plan('.', 'configs/lfm-phase-timing.json');assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_changed_frozen_input_and_duplicate_schedule_are_rejected(self):
        plan = load_plan(ROOT, ROOT / 'configs/lfm-phase-timing.json')
        bad = copy.deepcopy(plan['schedule'][:2]); bad[1] = bad[0]
        with self.assertRaisesRegex(ValueError, 'schedule prefix'):
            verify_prefix(bad, plan['schedule'])
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            path = Path(tmp) / 'config.json'
            cfg = copy.deepcopy(plan['config'])
            cfg['input_sha256']['prompts/phase-evaluation.json'] = 'bad'
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'Frozen input'):
                load_plan(ROOT, path)

    def test_correctness_format_and_ambiguous_answers_remain_separate(self):
        item = {'check': {'kind': 'exact-text', 'answer_kind': 'integer', 'expected': '41'}}
        self.assertEqual(task_score({'text': '41<|im_end|>'}, item)['strict_format_match'], True)
        wrapped = task_score({'text': 'The answer is 41.<|im_end|>'}, item)
        self.assertTrue(wrapped['answer_match'])
        self.assertFalse(wrapped['strict_format_match'])
        ambiguous = task_score({'text': '23 + 18 = 41'}, item)
        self.assertIsNone(ambiguous['answer_match'])
        marked = task_score({'text': '23 plus 18 gives 41.</think>41'}, item)
        self.assertTrue(marked['answer_match'])
        self.assertFalse(marked['strict_format_match'])
        symbol = {'check': {'kind': 'exact-text', 'answer_kind': 'text', 'expected': 'Au'}}
        self.assertTrue(task_score({'text': 'Au'}, symbol)['strict_format_match'])
        self.assertFalse(task_score({'text': 'AU'}, symbol)['answer_match'])

    def test_json_content_extraction_does_not_accept_two_answers_or_changed_types(self):
        item = {'check': {'kind': 'exact-json', 'expected': {'count': 2}}}
        wrapped = task_score({'text': 'Result: {"count":2}'}, item)
        self.assertTrue(wrapped['answer_match']); self.assertFalse(wrapped['strict_format_match'])
        self.assertIsNone(task_score({'text': '{"count":2} {"count":2}'}, item)['answer_match'])
        self.assertFalse(task_score({'text': '{"count":2.0}'}, item)['strict_format_match'])
        self.assertFalse(task_score({'text': '{"count":true}'}, item)['answer_match'])
        self.assertIsNone(task_score({'text': 'Mwahaha!'}, {})['answer_match'])

    def test_review_pack_is_stable_and_hides_condition_labels(self):
        outputs = [{'phase': 'probe', 'cohort': 'fresh-character', 'trial_id': c + ':toaster', 'condition': c,
                    'prompt': 'A toaster speaks.', 'text': 'A reply.'} for c in ['baseline', 'villain-all', 'random-all']]
        pack, key = blind_pack(outputs, 12)
        self.assertEqual((pack, key), blind_pack(outputs, 12))
        self.assertEqual(len(key), 3)
        self.assertTrue(all(set(row) == {'review_id', 'prompt', 'text'} for row in pack))
