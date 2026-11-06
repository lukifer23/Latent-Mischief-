"""Sample channel interventions before assigning behavioral labels. No persona target."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import lab

ROOT = lab.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/exploration.json')
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    cfg = json.loads(config_bytes)
    prompts = json.loads((ROOT / 'prompts/open-exploration.json').read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / 'runs' / f'{stamp}-MAP-001-channel-exploration'
    folder.mkdir(parents=True)
    print(f'RUN {folder}', flush=True)
    model, tokenizer = lab.load()
    chooser = random.Random(cfg['seed'])
    candidates = []
    for layer in cfg['layers']:
        width = model.model.layers[layer].feed_forward.w2.in_features
        for neuron in chooser.sample(range(width), cfg['channels_per_layer']):
            candidates.append({'layer': layer, 'neuron': neuron, 'operator': model.config.layer_types[layer],
                               'candidate': f'L{layer:02d}N{neuron:04d}'})
    sources = lab.archive_sources(folder, ['scripts/explore.py', 'scripts/lab.py', 'scripts/storage.py',
                                          'uv.lock', 'configs/model.json', 'configs/exploration.json',
                                          'prompts/open-exploration.json'])
    (folder / 'source/experiment-config.json').write_bytes(config_bytes)
    lab.write_json(folder / 'metadata.json', {
        'experiment': 'MAP-001', 'model': lab.config(), 'environment': lab.environment(),
        'experiment_config': cfg, 'experiment_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'source_hashes': sources, 'checkpoint_generation_defaults': model.generation_config.to_dict(),
        'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens']},
        'hook': 'feed_forward.w2 input: actual gated MLP channel activation, before post_ffn_norm',
        'positions': 'all prompt positions during prefill, then every generated-token forward',
        'selection': 'uniform seeded sampling without replacement per layer; chosen before observing outputs',
        'scope': 'pilot coverage only; labels assigned after inspection; no persona or accuracy target',
    })
    lab.write_json(folder / 'candidates.json', candidates)
    outputs, baselines = [], {}
    def record(result, **labels):
        result.update(labels)
        outputs.append(result)
        lab.write_json(folder / 'outputs.json', outputs)
        print(json.dumps({**labels, 'text': result['text']}, ensure_ascii=True), flush=True)
    def generate(item):
        return lab.generate(model, tokenizer, item['prompt'], system=cfg['system'], max_new_tokens=cfg['max_new_tokens'])
    for item in prompts:
        result = generate(item)
        baselines[item['id']] = result['token_ids'][:]
        record(result, phase='probe', condition='baseline', prompt_id=item['id'])
    first = candidates[0]
    with lab.gain_hook(model, first['layer'], first['neuron'], 1.0):
        neutral = generate(prompts[0])
    restored = generate(prompts[0])
    checks = {'gain_one_token_identical': neutral['token_ids'] == baselines[prompts[0]['id']],
              'removed_hook_token_identical': restored['token_ids'] == baselines[prompts[0]['id']]}
    record(neutral, phase='check', condition='gain-one', prompt_id=prompts[0]['id'])
    record(restored, phase='check', condition='removed-hook', prompt_id=prompts[0]['id'])
    lab.write_json(folder / 'checks.json', checks)
    if not all(checks.values()):
        raise RuntimeError('Gain hook neutrality / cleanup failed')
    for candidate in candidates:
        for gain in cfg['gains']:
            for item in prompts:
                with lab.gain_hook(model, candidate['layer'], candidate['neuron'], gain):
                    result = generate(item)
                record(result, phase='probe', condition='channel-gain', prompt_id=item['id'],
                       gain=gain, token_identical_to_baseline=result['token_ids'] == baselines[item['id']],
                       **candidate)
    order = list(range(len(outputs)))
    chooser.shuffle(order)
    lab.write_json(folder / 'review-order.json', order)
    lab.write_json(folder / 'summary.json', {
        'completed': True, 'checks': checks, 'candidate_count': len(candidates),
        'intervention_replies': sum(x['condition'] == 'channel-gain' for x in outputs),
        'changed_token_sequences': sum(x.get('token_identical_to_baseline') is False for x in outputs),
        'note': 'Changed tokens do not establish a personality change. Review before naming effects.'})
    print(f'COMPLETE {folder}', flush=True)


if __name__ == '__main__':
    main()
