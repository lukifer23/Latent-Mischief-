import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from measurements import measure
from phase_design import blind_pack, load_plan, task_score
from phase_experiment import source_names
from phase_report import collect, review_results
from storage import write_json


class PhaseReportTests(unittest.TestCase):
    def fixture(self, folder):
        plan = load_plan(ROOT, ROOT / 'configs/lfm-phase-timing.json')
        cfg = plan['config']
        hashes = {}
        for name in dict.fromkeys([*source_names(plan), 'scripts/runtime.py', 'scripts/token_stream.py']):
            data = (ROOT / name).read_bytes()
            path = folder / 'source' / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        write_json(folder / 'metadata.json', {'model': cfg['model'], 'environment': {'runtime': {'backend': 'mps', 'dtype': 'bfloat16'}},
            'experiment_config': cfg, 'experiment_config_path': plan['config_path'],
            'experiment_config_sha256': plan['config_sha256'], 'source_hashes': hashes, 'git_commit': 'fixture'})
        write_json(folder / 'trial-order.json', plan['schedule'])
        write_json(folder / 'generation-defaults.json', {'eos_token_id': 2, '_tokenizer_eos_token_id': 2})
        profiles = []
        for p in plan['profiles']:
            profiles.append({**p, 'offset': 1., 'offsets': p['native'], 'target_fraction': .05, 'hit_cap': False,
                'screen': [{'prompt_id': item['id'], 'actual_fraction': .05} for item in plan['calibration']], 'mean_actual_fraction': .05})
        write_json(folder / 'profiles.json', profiles)
        (folder / 'calibration.pt').write_bytes(b'test-only calibration artifact')
        items = {x['id']: x for x in plan['prompts']}
        rows = []
        for trial in plan['schedule']:
            item = items[trial['prompt_id']]
            text = json.dumps(item['check']['expected']) if item.get('check', {}).get('kind') == 'exact-json' else item.get('check', {}).get('expected', 'An ordinary reply.')
            trace = None
            if trial.get('intervention_phase') and not trial['condition'].startswith('restored-'):
                trace = {'prefill_forwards': 1, 'decode_forwards': 0,
                    'prefill_interventions': int(trial['intervention_phase'] in ('prefill', 'all')),
                    'decode_interventions': 0, 'intervened_positions': 0 if trial['intervention_phase'] == 'decode' else 2}
            row = {**trial, 'phase': trial['record_phase'], 'cohort': item['cohort'], 'domain': item['domain'],
                'prompt': item['prompt'], 'system': cfg['persona_system'] if trial['condition'] == 'prompt-villain' else cfg['system'],
                'text': text + '<|im_end|>', 'input_token_ids': [3, 4], 'input_tokens': 2,
                'token_ids': [2], 'output_tokens': 1, 'terminated_with_eos': True, 'hit_token_limit': False, 'seconds': .1,
                'token_identical_to_baseline': True, 'intervention_phase': trial.get('intervention_phase'),
                'profile_role': trial.get('profile_role'), 'hook_trace': trace, 'actual_prefill_fraction': 0.}
            row['task_score'], row['measurements'] = task_score(row, item), measure(row, item)
            rows.append(row)
        write_json(folder / 'outputs.json', rows)
        checks = {kind + '-' + phase + '_token_identical': True for phase in ('prefill', 'decode', 'all') for kind in ('zero', 'restored')}
        checks['parameter_versions_unchanged'] = True
        write_json(folder / 'checks.json', checks)
        write_json(folder / 'summary.json', {'completed': True, 'generations': 54, 'generated_tokens': 54, 'checks': checks,
            'profiles_sha256': hashlib.sha256((folder / 'profiles.json').read_bytes()).hexdigest(),
            'calibration_sha256': hashlib.sha256((folder / 'calibration.pt').read_bytes()).hexdigest()})
        return rows

    def test_complete_verification_and_duplicate_phase_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); rows = self.fixture(folder)
            data, _ = collect(folder)
            self.assertEqual(data['verification']['recorded_trials'], 54)
            self.assertEqual(len(data['groups']), 8)
            broken = copy.deepcopy(rows); broken[13] = broken[12]
            write_json(folder / 'outputs.json', broken)
            with self.assertRaisesRegex(ValueError, 'schedule prefix'): collect(folder)
            changed = copy.deepcopy(rows)
            row = next(x for x in changed if x['condition'] == 'villain-decode')
            row['hook_trace']['prefill_interventions'] = 1
            write_json(folder / 'outputs.json', changed)
            with self.assertRaisesRegex(ValueError, 'wrong forward phase'): collect(folder)
            changed = copy.deepcopy(rows); changed[0]['cohort'] = 'fresh-check'
            write_json(folder / 'outputs.json', changed)
            with self.assertRaisesRegex(ValueError, 'cohort/domain'): collect(folder)

    def test_ratings_required_and_individual_regression_blocks_nomination(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); self.fixture(folder)
            data, outputs = collect(folder)
            self.assertIn('required', data['manual_review'])
            pack, key = blind_pack(outputs, data['config']['review_seed'])
            ratings = {'outputs_sha256': data['outputs_sha256'],
                'pack_sha256': hashlib.sha256(json.dumps(pack, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), 'ratings': []}
            for identifier, info in key.items():
                is_target = outputs[info['output_index']]['condition'] == 'villain-prefill'
                ratings['ratings'].append({'review_id': identifier, 'villain_style': 2 if is_target else 0,
                    'coherence': 2, 'task_relevance': 2})
            path = folder / 'ratings.json'; write_json(path, ratings)
            self.assertTrue(review_results(data, outputs, path)['candidates'][0]['candidate_for_fresh_replication'])
            target = next(x for x in outputs if x['condition'] == 'villain-prefill' and x['cohort'] == 'fresh-check')
            target['task_score']['strict_format_match'] = False
            verdict = review_results(data, outputs, path)['candidates'][0]
            self.assertFalse(verdict['candidate_for_fresh_replication'])
            self.assertTrue(verdict['lost_task_checks'])
            ratings['ratings'][0]['review_id'] = ratings['ratings'][1]['review_id']; write_json(path, ratings)
            with self.assertRaisesRegex(ValueError, 'one unique rating'): review_results(data, outputs, path)

    def test_primary_timing_requires_superiority_over_both_without_completion_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); self.fixture(folder)
            data, outputs = collect(folder)
            pack, key = blind_pack(outputs, data['config']['review_seed'])
            ratings = {'outputs_sha256': data['outputs_sha256'],
                'pack_sha256': hashlib.sha256(json.dumps(pack, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                'ratings': [{'review_id': identifier,
                    'villain_style': 2 if outputs[info['output_index']]['condition'] == 'villain-decode' else 0,
                    'coherence': 2, 'task_relevance': 2} for identifier, info in key.items()]}
            path = folder / 'ratings.json'; write_json(path, ratings)
            verdict = review_results(data, outputs, path)
            self.assertTrue(verdict['primary_timing_comparison']['criterion_met'])
            for row in ratings['ratings']:
                if outputs[key[row['review_id']]['output_index']]['condition'] == 'villain-all':
                    row['villain_style'] = 2
            write_json(path, ratings)
            self.assertFalse(review_results(data, outputs, path)['primary_timing_comparison']['criterion_met'])
            for row in ratings['ratings']:
                if outputs[key[row['review_id']]['output_index']]['condition'] == 'villain-all':
                    row['villain_style'] = 0
            write_json(path, ratings)
            # Both and decode now share a baseline failure; regression only against both must still block.
            target = next(x for x in outputs if x['condition'] == 'villain-decode')
            target['terminated_with_eos'] = False
            next(x for x in outputs if x['condition'] == 'baseline' and x['prompt_id'] == target['prompt_id'])['terminated_with_eos'] = False
            verdict = review_results(data, outputs, path)
            self.assertTrue(verdict['candidates'][1]['candidate_for_fresh_replication'])
            self.assertFalse(verdict['primary_timing_comparison']['criterion_met'])
            self.assertEqual(verdict['primary_timing_comparison']['added_no_eos_vs_both'], [target['prompt_id']])
