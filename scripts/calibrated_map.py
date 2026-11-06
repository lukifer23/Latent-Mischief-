"""Evaluate frozen calibrated channel offsets on prompt-disjoint diverse tasks."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import lab
from calibration import offset_hook
from measurements import measure

ROOT = lab.ROOT


@contextmanager
def capture_prefill(model, layers):
    captured, handles = {}, []
    def hook(layer):
        def save(_module, _args, output):
            if layer not in captured:
                captured[layer] = output[0].detach().float().cpu().clone()
        return save
    try:
        for layer in layers:
            handles.append(model.model.layers[layer].post_ffn_norm.register_forward_hook(hook(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True, help='Completed local calibration run directory name')
    args = parser.parse_args()
    screen = (ROOT / 'runs' / args.calibration).resolve()
    if screen.parent != (ROOT / 'runs').resolve():
        raise ValueError('Calibration must be a local run ID')
    meta = json.loads((screen / 'metadata.json').read_text(encoding='utf-8'))
    summary = json.loads((screen / 'summary.json').read_text())
    if not summary['completed'] or not all(summary['checks'].values()):
        raise ValueError('Calibration is incomplete or failed')
    for name in ('screening', 'selected'):
        if hashlib.sha256((screen / f'{name}.json').read_bytes()).hexdigest() != summary[f'{name}_sha256']:
            raise ValueError('Calibration artifact hash mismatch')
    if meta['model'] != lab.config() or meta['environment']['lock_sha256'] != lab.environment()['lock_sha256']:
        raise ValueError('Model/runtime lock differs from calibration')
    selected = json.loads((screen / 'selected.json').read_text())
    if not selected:
        raise ValueError('No calibrated candidates qualified; report screen instead of generating')
    config_bytes = (screen / 'source/experiment-config.json').read_bytes()
    cfg = json.loads(config_bytes)
    prompt_name = cfg['evaluation_prompts']
    prompts = json.loads((ROOT / prompt_name).read_text(encoding='utf-8'))
    if hashlib.sha256((ROOT / prompt_name).read_bytes()).hexdigest() != meta['source_hashes'][prompt_name]:
        raise ValueError('Evaluation prompt file changed after screening')
    folder = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-MAP-005-calibrated-offsets')
    folder.mkdir(parents=True)
    print(f'RUN {folder}', flush=True)
    model, tokenizer = lab.load()
    versions = {name: p._version for name, p in model.named_parameters()}
    hashes = lab.archive_sources(folder, ['scripts/calibrated_map.py', 'scripts/calibration.py', 'scripts/measurements.py',
        'scripts/lab.py', 'scripts/storage.py', 'uv.lock', 'configs/model.json', prompt_name])
    for name in ('metadata.json', 'summary.json', 'selected.json'):
        data = (screen / name).read_bytes()
        destination = folder / 'source/calibration' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[f'calibration/{name}'] = hashlib.sha256(data).hexdigest()
    (folder / 'source/experiment-config.json').write_bytes(config_bytes)
    lab.write_json(folder / 'metadata.json', {
        'experiment': 'MAP-005', 'model': lab.config(), 'environment': lab.environment(), 'source_hashes': hashes,
        'experiment_config': cfg, 'experiment_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'calibration_run': screen.name, 'calibration_summary': summary,
        'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens']},
        'checkpoint_generation_defaults': model.generation_config.to_dict(),
        'hook': 'constant native-activation offset at w2 input, before post_ffn_norm',
        'positions': 'all prompt and decode positions; measured effect is non-special prefill positions excluding zero',
        'selection': cfg['selection_rule'], 'scope': cfg['scope'],
        'limits': 'Exploratory evaluation, one greedy trajectory per condition; exact checks do not score all domains.'})
    lab.write_json(folder / 'selected.json', selected)
    baseline_ids, baseline_prefill, outputs = {}, {}, []
    layers = sorted({x['layer'] for x in selected})
    def generate(item):
        return lab.generate(model, tokenizer, item['prompt'], system=cfg['system'], max_new_tokens=cfg['max_new_tokens'])
    def record(result, item, **labels):
        result.update(labels)
        result.update(cohort=item['cohort'], domain=item['domain'], measurements=measure(result, item))
        outputs.append(result)
        lab.write_json(folder / 'outputs.json', outputs)
        print(json.dumps({**labels, 'tokens': result['output_tokens'], 'eos': result['terminated_with_eos']}, ensure_ascii=True), flush=True)
    for item in prompts:
        with capture_prefill(model, layers) as captured:
            result = generate(item)
        baseline_ids[item['id']] = result['token_ids'][:]
        baseline_prefill[item['id']] = captured
        record(result, item, phase='probe', condition='baseline', prompt_id=item['id'])
    torch.save(baseline_prefill, folder / 'baseline-prefill.pt')
    first = selected[0]
    with offset_hook(model, first['layer'], first['neuron'], 0):
        zero = generate(prompts[0])
    restored = generate(prompts[0])
    checks = {'zero_offset_token_identical': zero['token_ids'] == baseline_ids[prompts[0]['id']],
              'removed_hook_token_identical': restored['token_ids'] == baseline_ids[prompts[0]['id']]}
    record(zero, prompts[0], phase='check', condition='zero-offset', prompt_id=prompts[0]['id'])
    record(restored, prompts[0], phase='check', condition='removed-hook', prompt_id=prompts[0]['id'])
    lab.write_json(folder / 'checks.json', checks)
    if not all(checks.values()):
        raise RuntimeError('Offset generation check failed')
    schedule = [(unit, item) for unit in selected for item in prompts]
    random.Random(cfg['seed'] + 17).shuffle(schedule)
    lab.write_json(folder / 'trial-order.json', [{'condition': 'channel-offset', 'candidate': u['candidate'],
        'offset': u['offset'], 'prompt_id': p['id']} for u, p in schedule])
    special = set(tokenizer.all_special_ids)
    for unit, item in schedule:
        with capture_prefill(model, [unit['layer']]) as captured:
            with offset_hook(model, unit['layer'], unit['neuron'], unit['offset']):
                result = generate(item)
        mask = torch.tensor([i > 0 and token not in special for i, token in enumerate(result['input_token_ids'])])
        reference = baseline_prefill[item['id']][unit['layer']][mask]
        changed = captured[unit['layer']][mask]
        actual = ((changed - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)).mean().item()
        record(result, item, phase='probe', condition='channel-offset', prompt_id=item['id'],
            candidate=unit['candidate'], layer=unit['layer'], neuron=unit['neuron'], operator=unit['operator'],
            sign=unit['sign'], target_fraction=unit['target_fraction'], offset=unit['offset'],
            offset_in_rms_units=unit['offset_in_rms_units'], actual_prefill_fraction=actual,
            screen_actual_fraction=unit['mean_actual_fraction'], role='calibrated-median',
            token_identical_to_baseline=result['token_ids'] == baseline_ids[item['id']])
    checks['parameter_version_counters_unchanged'] = versions == {name: p._version for name, p in model.named_parameters()}
    lab.write_json(folder / 'checks.json', checks)
    lab.write_json(folder / 'summary.json', {'completed': True, 'checks': checks, 'calibration_run': screen.name,
        'selected_channels': len(layers), 'intervention_replies': len(schedule), 'generations': len(outputs),
        'changed_token_sequences': sum(x.get('token_identical_to_baseline') is False for x in outputs),
        'note': 'Descriptions require inspection and later replication; this is not a persona or intelligence score.'})
    if not all(checks.values()):
        raise RuntimeError('Parameter version invariant failed')
    print(f'COMPLETE {folder}', flush=True)


if __name__ == '__main__':
    main()
