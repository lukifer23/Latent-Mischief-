"""Reversible channel exploration and predeclared replication; no persona target."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re

import lab
from design import build_candidates
from measurements import measure

ROOT = lab.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    cfg = json.loads(config_bytes)
    experiment = cfg['experiment_id']
    if not re.fullmatch(r'[A-Z]+-[0-9]+', experiment):
        raise ValueError('Invalid experiment ID')
    prompt_name = cfg['prompt_file']
    prompt_path = (ROOT / prompt_name).resolve()
    if not prompt_path.is_relative_to(ROOT / 'prompts'):
        raise ValueError('Prompt file must be under prompts/')
    prompts = json.loads(prompt_path.read_text(encoding='utf-8'))
    if not prompts or len({p['id'] for p in prompts}) != len(prompts):
        raise ValueError('Nonempty prompt set with unique IDs required')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / 'runs' / f'{stamp}-{experiment}-channel-exploration'
    folder.mkdir(parents=True)
    print(f'RUN {folder}', flush=True)
    model, tokenizer = lab.load()
    candidates = build_candidates(cfg, model.config.layer_types,
                                  [layer.feed_forward.w2.in_features for layer in model.model.layers])
    sources = lab.archive_sources(folder, ['scripts/map_experiment.py', 'scripts/design.py', 'scripts/measurements.py',
        'scripts/lab.py', 'scripts/storage.py', 'uv.lock', 'configs/model.json', prompt_name])
    (folder / 'source/experiment-config.json').write_bytes(config_bytes)
    lab.write_json(folder / 'metadata.json', {
        'experiment': experiment, 'model': lab.config(), 'environment': lab.environment(),
        'experiment_config': cfg, 'experiment_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'source_hashes': sources, 'checkpoint_generation_defaults': model.generation_config.to_dict(),
        'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens']},
        'hook': 'feed_forward.w2 input: actual gated MLP channel activation, before post_ffn_norm',
        'positions': 'all prompt positions during prefill, then every generated-token forward',
        'selection': cfg.get('selection_note', 'uniform seeded sampling without replacement per layer'),
        'scope': cfg['scope'],
        'seed_scope': 'candidate selection and trial order; greedy generation uses the model config seed',
        'measurement_limits': 'phrase/think markers and EOS are surface indicators, not reasoning or accuracy scores',
    })
    lab.write_json(folder / 'candidates.json', candidates)
    outputs, baselines = [], {}
    def record(result, item, **labels):
        result.update(labels)
        result['cohort'] = item.get('cohort', 'exploration')
        result['measurements'] = measure(result, item)
        outputs.append(result)
        lab.write_json(folder / 'outputs.json', outputs)
        print(json.dumps({**labels, 'text': result['text']}, ensure_ascii=True), flush=True)
    def generate(item):
        return lab.generate(model, tokenizer, item['prompt'], system=cfg['system'], max_new_tokens=cfg['max_new_tokens'])
    for item in prompts:
        result = generate(item)
        baselines[item['id']] = result['token_ids'][:]
        record(result, item, phase='probe', condition='baseline', prompt_id=item['id'])
    first = candidates[0]
    with lab.gain_hook(model, first['layer'], first['neuron'], 1.0):
        neutral = generate(prompts[0])
    restored = generate(prompts[0])
    checks = {'gain_one_token_identical': neutral['token_ids'] == baselines[prompts[0]['id']],
              'removed_hook_token_identical': restored['token_ids'] == baselines[prompts[0]['id']]}
    record(neutral, prompts[0], phase='check', condition='gain-one', prompt_id=prompts[0]['id'])
    record(restored, prompts[0], phase='check', condition='removed-hook', prompt_id=prompts[0]['id'])
    lab.write_json(folder / 'checks.json', checks)
    if not all(checks.values()):
        raise RuntimeError('Gain hook neutrality / cleanup failed')
    schedule = [(c, g, p) for c in candidates for g in c['gains'] for p in prompts]
    random.Random(cfg['seed'] + 1).shuffle(schedule)
    lab.write_json(folder / 'trial-order.json', [{'candidate': c['candidate'], 'gain': g, 'prompt_id': p['id']} for c, g, p in schedule])
    for candidate, gain, item in schedule:
        with lab.gain_hook(model, candidate['layer'], candidate['neuron'], gain):
            result = generate(item)
        labels = {key: value for key, value in candidate.items() if key != 'gains'}
        record(result, item, phase='probe', condition='channel-gain', prompt_id=item['id'],
               gain=gain, token_identical_to_baseline=result['token_ids'] == baselines[item['id']], **labels)
    order = list(range(len(outputs)))
    random.Random(cfg['seed'] + 2).shuffle(order)
    lab.write_json(folder / 'review-order.json', order)
    lab.write_json(folder / 'summary.json', {
        'completed': True, 'checks': checks, 'candidate_count': len(candidates),
        'intervention_replies': len(schedule),
        'changed_token_sequences': sum(x.get('token_identical_to_baseline') is False for x in outputs),
        'note': 'Changed tokens do not establish a personality change. Review before naming effects.'})
    print(f'COMPLETE {folder}', flush=True)


if __name__ == '__main__':
    main()
