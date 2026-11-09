"""Offline LFM phase plans, conservative task checks and blinded review identity."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata


CONDITIONS = ('baseline', 'prompt-villain', 'villain-prefill', 'villain-decode', 'villain-all',
              'random-prefill', 'random-decode', 'random-all')
PHASES = ('prefill', 'decode', 'all')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_plan(root, config_path):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    if not config_path.is_relative_to(root):
        raise ValueError('Config must be inside the repository')
    cfg = read(config_path)
    for name, digest in cfg['input_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Frozen input differs: ' + name)
    current = read(root / 'configs/model.json')
    if {key: current[key] for key in cfg['model']} != cfg['model']:
        raise ValueError('Pinned model changed')
    evidence = read(root / cfg['source_evidence'])
    if {key: evidence['model'][key] for key in cfg['model']} != cfg['model']:
        raise ValueError('Source profile belongs to a different model')
    profiles = []
    for role in ('villain', 'random'):
        matches = [x for x in evidence['profiles'] if x['profile_id'] == cfg['source_profiles'][role]]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous frozen profile')
        profile = matches[0]
        if profile['layer'] != cfg['layer'] or len(profile['channels']) != cfg['channels_per_profile']:
            raise ValueError('Source profile size/layer changed')
        if len(set(profile['channels'])) != len(profile['channels']) or len(profile['native']) != len(profile['channels']):
            raise ValueError('Invalid sparse profile')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in profile['native']):
            raise ValueError('Non-finite source profile')
        profiles.append({'role': role, 'source_profile_id': profile['profile_id'], 'layer': profile['layer'],
                         'channels': profile['channels'], 'native': profile['native']})
    if set(profiles[0]['channels']) & set(profiles[1]['channels']):
        raise ValueError('Random control overlaps the target')
    prompts = read(root / cfg['evaluation_prompts'])
    calibration = read(root / cfg['calibration_prompts'])
    ids = [x['id'] for x in prompts]
    if len(ids) != 6 or len(set(ids)) != len(ids) or len(calibration) != 4:
        raise ValueError('Protocol requires six unique evaluation prompts and four calibration inputs')
    if {x['prompt'] for x in prompts} & {x['prompt'] for x in calibration}:
        raise ValueError('Calibration and evaluation overlap')
    if sum(x['cohort'] == 'fresh-character' for x in prompts) != 3 or sum('check' in x for x in prompts) != 3:
        raise ValueError('Three fresh character prompts and three task checks required')
    if cfg['seed'] != current['seed'] or cfg['thinking'] is not False or cfg['max_new_tokens'] != 48:
        raise ValueError('Frozen seed/thinking/token policy changed')
    if cfg['conditions'] != list(CONDITIONS) or cfg['target_fraction'] != .05:
        raise ValueError('Frozen conditions/strength changed')
    baselines = [{'trial_id': 'baseline:' + p['id'], 'condition': 'baseline', 'prompt_id': p['id'],
                  'record_phase': 'probe'} for p in prompts]
    checks = [{'trial_id': kind + ':' + phase, 'condition': kind + '-' + phase, 'prompt_id': ids[0],
               'record_phase': 'check', 'intervention_phase': phase}
              for phase in PHASES for kind in ('zero', 'restored')]
    probes = [{'trial_id': c + ':' + p['id'], 'condition': c, 'prompt_id': p['id'], 'record_phase': 'probe',
               **({'profile_role': c.split('-')[0], 'intervention_phase': c.split('-')[1]}
                  if c.startswith(('villain-', 'random-')) else {})}
              for c in CONDITIONS[1:] for p in prompts]
    random.Random(cfg['schedule_seed']).shuffle(probes)
    schedule = baselines + checks + probes
    if len(schedule) != 54 or cfg['total_generated_tokens'] != len(schedule) * cfg['max_new_tokens']:
        raise ValueError('Generation budget differs from the full design')
    return {'schema': 1, 'experiment': cfg['experiment_id'], 'config': cfg,
            'config_path': config_path.relative_to(root).as_posix(), 'config_sha256': sha(config_path),
            'profiles': profiles, 'prompts': prompts, 'calibration': calibration, 'schedule': schedule,
            'generations': len(schedule), 'maximum_generated_tokens': cfg['total_generated_tokens'],
            'requires_inference': False, 'status': 'planned; no model loaded'}


def task_score(result, item):
    check = item.get('check')
    if check is None:
        return {'answer_match': None, 'strict_format_match': None, 'answer_extraction': 'human review required'}
    plain = result['text'].replace('<|im_end|>', '').strip()
    final = plain.rsplit('</think>', 1)[-1].strip()
    normalize = lambda s: unicodedata.normalize('NFC', s)
    if check['kind'] == 'exact-json':
        equivalent = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False) == json.dumps(check['expected'], sort_keys=True, ensure_ascii=False)
        try:
            strict = equivalent(json.loads(plain))
        except (ValueError, TypeError):
            strict = False
        candidates = []
        decoder = json.JSONDecoder()
        for match in re.finditer(r'\{', final):
            try:
                value, _end = decoder.raw_decode(final[match.start():])
                candidates.append(value)
            except ValueError:
                continue
        answer = equivalent(candidates[0]) if len(candidates) == 1 else None
        return {'answer_match': answer, 'strict_format_match': strict,
                'answer_extraction': 'one JSON object' if len(candidates) == 1 else 'missing or ambiguous JSON object'}
    strict = normalize(plain) == normalize(check['expected'])
    if check['answer_kind'] == 'integer':
        values = re.findall(r'(?<![\w.])[+-]?\d+(?!\w|\.\d)', final)
        answer = int(values[0]) == int(check['expected']) if len(values) == 1 else None
        reason = 'one standalone integer' if len(values) == 1 else 'missing or ambiguous integer'
    else:
        answer = normalize(final.rstrip('.!')) == normalize(check['expected'])
        reason = 'complete final text after the last boundary, optional terminal punctuation'
    return {'answer_match': answer, 'strict_format_match': strict, 'answer_extraction': reason}


def blind_pack(outputs, seed):
    selected = [(i, x) for i, x in enumerate(outputs) if x.get('cohort') == 'fresh-character' and x.get('phase') == 'probe']
    random.Random(seed).shuffle(selected)
    pack, key = [], {}
    for number, (index, row) in enumerate(selected, 1):
        identifier = 'R' + str(number).zfill(3)
        pack.append({'review_id': identifier, 'prompt': row['prompt'], 'text': row['text']})
        key[identifier] = {'output_index': index, 'trial_id': row['trial_id']}
    return pack, key
