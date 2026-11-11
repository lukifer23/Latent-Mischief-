"""Verify phase runs and export numeric comparisons or a private blinded review pack."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from measurements import measure, repeated_fourgrams
from phase_design import blind_pack, load_plan, read, sha, task_score
from phase_experiment import source_names, validate_trace
from storage import write_json

ROOT = Path(__file__).resolve().parents[1]
RATING_FIELDS = ('villain_style', 'coherence', 'task_relevance')


def verify_prefix(outputs, schedule):
    actual = [x['trial_id'] for x in outputs]
    expected = [x['trial_id'] for x in schedule]
    if actual != expected[:len(actual)] or len(actual) != len(set(actual)):
        raise ValueError('Outputs are not a unique frozen schedule prefix')


def collect(folder):
    folder = Path(folder).resolve()
    meta = read(folder / 'metadata.json')
    for name, digest in meta['source_hashes'].items():
        source = (folder / 'source' / name).resolve()
        if not source.is_relative_to((folder / 'source').resolve()) or sha(source) != digest:
            raise ValueError('Archived source mismatch: ' + name)
    config_name = meta['experiment_config_path']
    if config_name not in meta['source_hashes']:
        raise ValueError('Config is absent from the source archive')
    plan = load_plan(folder / 'source', folder / 'source' / config_name)
    cfg = plan['config']
    if set(meta['source_hashes']) != set(source_names(plan)) | {'scripts/runtime.py', 'scripts/token_stream.py'}:
        raise ValueError('Required source provenance is incomplete')
    if {k: meta['model'].get(k) for k in cfg['model']} != cfg['model']:
        raise ValueError('Recorded model differs from frozen identity')
    if any(meta['environment']['runtime'].get(k) != v for k, v in {'backend': cfg.get('backend', 'mps'), 'dtype': 'bfloat16'}.items()):
        raise ValueError('Recorded backend/dtype differs from the protocol')
    if plan['config_sha256'] != meta['experiment_config_sha256'] or cfg != meta['experiment_config']:
        raise ValueError('Recorded config differs from its archive')
    if read(folder / 'trial-order.json') != plan['schedule']:
        raise ValueError('Trial schedule differs from the complete design')
    outputs = read(folder / 'outputs.json')
    verify_prefix(outputs, plan['schedule'])
    items = {p['id']: p for p in plan['prompts']}
    baselines, rows = {}, []
    defaults = read(folder / 'generation-defaults.json') if outputs else {}
    eos = defaults.get('eos_token_id', [])
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(defaults.get('_tokenizer_eos_token_id'))
    eos.discard(None)
    for row, trial in zip(outputs, plan['schedule']):
        item = items[trial['prompt_id']]
        expected_system = cfg['persona_system'] if trial['condition'] == 'prompt-villain' else cfg['system']
        if any(row.get(k) != trial.get(k) for k in ('condition', 'prompt_id', 'intervention_phase', 'profile_role')):
            raise ValueError('Recorded condition differs from frozen trial')
        if row['phase'] != trial['record_phase'] or row['prompt'] != item['prompt'] or row['system'] != expected_system:
            raise ValueError('Prompt/system/phase mismatch')
        if any(row[k] != item[k] for k in ('cohort', 'domain')):
            raise ValueError('Recorded prompt cohort/domain differs from the frozen input')
        if row['output_tokens'] != len(row['token_ids']) or not 1 <= row['output_tokens'] <= cfg['max_new_tokens']:
            raise ValueError('Invalid output token count')
        if row['input_tokens'] != len(row['input_token_ids']) or row['input_tokens'] > cfg['max_input_tokens']:
            raise ValueError('Invalid input token count')
        if row['terminated_with_eos'] != (row['token_ids'][-1] in eos):
            raise ValueError('EOS flag differs from recorded token sequence')
        if row['hit_token_limit'] != (row['output_tokens'] == cfg['max_new_tokens']):
            raise ValueError('Token cap flag mismatch')
        if row['task_score'] != task_score(row, item) or row['measurements'] != measure(row, item):
            raise ValueError('Stored task/surface score differs from recomputation')
        if row['condition'] == 'baseline':
            baselines[item['id']] = row
        base = baselines[item['id']]
        if row['condition'] != 'prompt-villain' and row['input_token_ids'] != base['input_token_ids']:
            raise ValueError('Internal comparison input tokens differ')
        if row['token_identical_to_baseline'] != (row['token_ids'] == base['token_ids']):
            raise ValueError('Stored token equality differs')
        if trial['record_phase'] == 'check' and row['token_ids'] != base['token_ids']:
            raise ValueError('Neutral/removal reply differs from baseline')
        if trial.get('intervention_phase') == 'decode' and row['token_ids'][0] != base['token_ids'][0]:
            raise ValueError('Decode-only first reply token differs')
        if trial.get('intervention_phase') and not row['condition'].startswith('restored-'):
            validate_trace(row['hook_trace'], trial['intervention_phase'], row['output_tokens'], row['input_tokens'])
        elif row['hook_trace'] is not None:
            raise ValueError('Unexpected hook on an unsteered trial')
        rows.append({k: row[k] for k in ('trial_id', 'condition', 'prompt_id', 'cohort', 'phase', 'input_tokens', 'output_tokens',
            'terminated_with_eos', 'hit_token_limit', 'seconds', 'task_score', 'intervention_phase', 'profile_role',
            'hook_trace', 'actual_prefill_fraction', 'token_identical_to_baseline')})
        rows[-1]['repeated_fourgram_fraction'] = repeated_fourgrams(row['token_ids'])
        rows[-1]['text_sha256'] = hashlib.sha256(row['text'].encode()).hexdigest()
    summary = read(folder / 'summary.json') if (folder / 'summary.json').exists() else None
    checks = read(folder / 'checks.json')
    expected_checks = {kind + '-' + phase + '_token_identical' for phase in ('prefill', 'decode', 'all') for kind in ('zero', 'restored')}
    if summary:
        if summary.get('completed') is not True or len(outputs) != len(plan['schedule']) or not all(checks.values()):
            raise ValueError('Completed run has missing trials or failed checks')
        if set(checks) != expected_checks | {'parameter_versions_unchanged'} or summary['checks'] != checks:
            raise ValueError('Completed run omits required hook/parameter checks')
        if summary['generations'] != len(outputs) or summary['generated_tokens'] != sum(x['output_tokens'] for x in outputs):
            raise ValueError('Completed count mismatch')
        for name in ('calibration', 'profiles'):
            path = folder / ('calibration.pt' if name == 'calibration' else 'profiles.json')
            if sha(path) != summary[name + '_sha256']:
                raise ValueError('Calibration artifact changed')
    if outputs and len(outputs) > 6:
        if any(checks.get(name) is not True for name in expected_checks if any(x['condition'] + '_token_identical' == name for x in outputs)):
            raise ValueError('A recorded neutral/removal check failed')
    profiles = read(folder / 'profiles.json') if (folder / 'profiles.json').exists() else []
    if len(outputs) > 0:
        verify_profiles(profiles, plan)
    grouped = defaultdict(list)
    for row in rows:
        if row['phase'] == 'probe': grouped[row['condition']].append(row)
    groups = []
    for condition, values in grouped.items():
        scored = [x for x in values if x['cohort'] == 'fresh-check']
        groups.append({'condition': condition, 'n': len(values),
            'changed': sum(not x['token_identical_to_baseline'] for x in values),
            'eos': sum(x['terminated_with_eos'] for x in values),
            'mean_tokens': statistics.mean(x['output_tokens'] for x in values),
            'answer_pass': sum(x['task_score']['answer_match'] is True for x in scored),
            'answer_ungraded': sum(x['task_score']['answer_match'] is None for x in scored),
            'strict_pass': sum(x['task_score']['strict_format_match'] is True for x in scored), 'task_n': len(scored)})
    return {'schema': 1, 'experiment': cfg['experiment_id'], 'run': folder.name, 'model': meta['model'], 'config': cfg,
        'report_generator_sha256': sha(Path(__file__)),
        'environment': meta['environment'], 'predeclared_commit': meta['git_commit'], 'source_hashes': meta['source_hashes'],
        'outputs_sha256': sha(folder / 'outputs.json'), 'checks': checks, 'summary': summary, 'profiles': profiles,
        'rows': rows, 'groups': groups, 'manual_review': 'required; no automatic personality score',
        'interruption': read(folder / 'interruption.json') if (folder / 'interruption.json').exists() else None,
        'verification': {'source_files': len(meta['source_hashes']), 'recorded_trials': len(outputs), 'planned_trials': len(plan['schedule'])},
        'limits': cfg['scope']}, outputs


def verify_profiles(profiles, plan):
    cfg = plan['config']
    if {p['role'] for p in profiles} != {'villain', 'random'} or len(profiles) != 2:
        raise ValueError('Both calibrated profiles are required before generation')
    for row in profiles:
        source = next(p for p in plan['profiles'] if p['role'] == row['role'])
        if any(row[k] != source[k] for k in source):
            raise ValueError('Frozen sparse direction changed')
        if row['hit_cap'] or not 0 < row['offset'] <= cfg['max_amplitude'] or row['target_fraction'] != cfg['target_fraction']:
            raise ValueError('Invalid calibration amplitude/target')
        if len(row['offsets']) != len(row['native']) or any(not math.isclose(x, y * row['offset'], rel_tol=2e-6, abs_tol=1e-9) for x, y in zip(row['offsets'], row['native'])):
            raise ValueError('Native offsets differ from calibrated amplitude')
        if [x['prompt_id'] for x in row['screen']] != [p['id'] for p in plan['calibration']]:
            raise ValueError('Incomplete calibration screen')
        actual = statistics.mean(x['actual_fraction'] for x in row['screen'])
        if not math.isclose(actual, row['mean_actual_fraction'], abs_tol=1e-9) or abs(actual / cfg['target_fraction'] - 1) > cfg['calibration_tolerance_fraction']:
            raise ValueError('Calibration screen is outside the matched strength tolerance')


def review_results(data, outputs, ratings_path):
    if not data['summary']:
        raise ValueError('Complete the run before blinded style review')
    pack, key = blind_pack(outputs, data['config']['review_seed'])
    ratings = read(ratings_path)
    digest = hashlib.sha256(json.dumps(pack, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if ratings['outputs_sha256'] != data['outputs_sha256'] or ratings['pack_sha256'] != digest:
        raise ValueError('Ratings refer to a different output/review pack')
    actual = [r['review_id'] for r in ratings['ratings']]
    if len(actual) != len(key) or set(actual) != set(key):
        raise ValueError('Every blinded item needs one unique rating')
    lookup = {}
    for row in ratings['ratings']:
        if any(type(row.get(field)) is not int or not 0 <= row[field] <= 2 for field in RATING_FIELDS):
            raise ValueError('Style/coherence/task relevance ratings must be integers 0–2')
        lookup[key[row['review_id']]['trial_id']] = row
    base = {x['prompt_id']: x for x in outputs if x['condition'] == 'baseline' and x['phase'] == 'probe'}
    candidates = []
    for phase in ('prefill', 'decode', 'all'):
        def qualified(condition):
            return [x for x in outputs if x['condition'] == condition and x['cohort'] == 'fresh-character'
                    and all(lookup[x['trial_id']][field] == 2 for field in RATING_FIELDS)]
        target = [x for x in outputs if x['condition'] == 'villain-' + phase]
        losses = [x['prompt_id'] + ':' + field for x in target for field in ('answer_match', 'strict_format_match')
                  if base[x['prompt_id']]['task_score'][field] is True and x['task_score'][field] is not True]
        added_caps = [x['prompt_id'] for x in target if base[x['prompt_id']]['terminated_with_eos'] and not x['terminated_with_eos']]
        count, control = len(qualified('villain-' + phase)), len(qualified('random-' + phase))
        candidates.append({'phase': phase, 'qualified_villain_replies': count, 'qualified_random_replies': control,
            'lost_task_checks': losses, 'added_no_eos': added_caps,
            'candidate_for_fresh_replication': count >= 2 and count > control and not losses and not added_caps})
    by_phase = {row['phase']: row for row in candidates}
    all_rows = {x['prompt_id']: x for x in outputs if x['condition'] == 'villain-all'}
    decode_rows = [x for x in outputs if x['condition'] == 'villain-decode']
    lost_vs_all = [x['prompt_id'] + ':' + field for x in decode_rows for field in ('answer_match', 'strict_format_match')
                   if all_rows[x['prompt_id']]['task_score'][field] is True and x['task_score'][field] is not True]
    caps_vs_all = [x['prompt_id'] for x in decode_rows if all_rows[x['prompt_id']]['terminated_with_eos'] and not x['terminated_with_eos']]
    decode, both = by_phase['decode'], by_phase['all']
    primary = {'question': 'decode-only versus both phases', 'lost_task_checks_vs_both': lost_vs_all,
               'added_no_eos_vs_both': caps_vs_all,
               'criterion_met': decode['candidate_for_fresh_replication']
                   and decode['qualified_villain_replies'] > both['qualified_villain_replies']
                   and decode['qualified_villain_replies'] - decode['qualified_random_replies']
                       > both['qualified_villain_replies'] - both['qualified_random_replies']
                   and not lost_vs_all and not caps_vs_all}
    return {'ratings_sha256': sha(ratings_path), 'candidates': candidates, 'primary_timing_comparison': primary,
            'scope': 'Human-rated exploratory nomination on three fresh scenarios; no validated persona or capability result.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--publish', action='store_true', help='Write reviewed numeric metrics into experiments/evidence/<experiment>')
    parser.add_argument('--blind-review', action='store_true')
    parser.add_argument('--ratings', type=Path)
    args = parser.parse_args()
    folder = (ROOT / 'runs' / args.run).resolve()
    if folder.parent != (ROOT / 'runs').resolve():
        raise ValueError('Use a local run directory name')
    data, outputs = collect(folder)
    if args.blind_review:
        if not data['summary']:
            raise ValueError('A complete run is required for a stable review pack')
        pack, key = blind_pack(outputs, data['config']['review_seed'])
        write_json(folder / 'blind-review.json', pack)
        write_json(folder / 'review-key.json', key)
        write_json(folder / 'review-instructions.json', {'outputs_sha256': data['outputs_sha256'],
            'pack_sha256': hashlib.sha256(json.dumps(pack, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'required_fields': ['review_id', *RATING_FIELDS], 'rating_range': [0, 2],
            'rubric': {'villain_style': ['ordinary voice', 'isolated stylistic cue', 'sustained theatrical characterization'],
                       'coherence': ['broken or incoherent', 'partly coherent/repetitive', 'coherent'],
                       'task_relevance': ['does not answer', 'partly addresses request', 'answers the actual request']}})
    if args.ratings:
        data['manual_review'] = review_results(data, outputs, args.ratings)
    destination = ROOT / 'experiments/evidence' / data['experiment'] / 'metrics.json' if args.publish else folder / 'analysis.json'
    write_json(destination, data)
    print(json.dumps({'destination': str(destination), 'verification': data['verification'], 'groups': data['groups'],
                      'manual_review': data['manual_review']}, indent=2))


if __name__ == '__main__':
    main()
