"""Resume an interrupted calibrated mapping into a new run after exact baseline revalidation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lab
from calibrated_map import capture_prefill
from calibration import offset_hook
from measurements import measure


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    parent = (lab.ROOT / 'runs' / args.run).resolve()
    if parent.parent != (lab.ROOT / 'runs').resolve():
        raise ValueError('Use a local run directory name')
    if (parent / 'summary.json').exists():
        raise ValueError('Resume only an interrupted run, preserving its directory')
    metadata = json.loads((parent / 'metadata.json').read_text(encoding='utf-8'))
    for name, expected in metadata['source_hashes'].items():
        if hashlib.sha256((parent / 'source' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Parent source hash mismatch: {name}')
    if metadata['model'] != lab.config() or metadata['environment']['lock_sha256'] != lab.environment()['lock_sha256']:
        raise ValueError('Model/runtime changed')
    cfg_bytes = (parent / 'source/experiment-config.json').read_bytes()
    if hashlib.sha256(cfg_bytes).hexdigest() != metadata['experiment_config_sha256']:
        raise ValueError('Parent config changed')
    cfg = json.loads(cfg_bytes)
    prompts = json.loads((parent / 'source' / cfg['evaluation_prompts']).read_text(encoding='utf-8'))
    selected = json.loads((parent / 'selected.json').read_text(encoding='utf-8'))
    if hashlib.sha256((parent / 'selected.json').read_bytes()).hexdigest() != metadata['calibration_summary']['selected_sha256']:
        raise ValueError('Frozen offsets changed')
    outputs = json.loads((parent / 'outputs.json').read_text(encoding='utf-8'))
    trials = json.loads((parent / 'trial-order.json').read_text(encoding='utf-8'))
    def identity(x):
        return x['candidate'], x['offset'], x['prompt_id']
    done = [identity(x) for x in outputs if x['condition'] == 'channel-offset']
    if done != [identity(x) for x in trials[:len(done)]] or len(set(done)) != len(done):
        raise ValueError('Parent completed trials are not a unique schedule prefix')
    baseline = {x['prompt_id']: x for x in outputs if x['condition'] == 'baseline' and x['phase'] == 'probe'}
    if set(baseline) != {x['id'] for x in prompts}:
        raise ValueError('Parent does not have all baselines')
    inherited_checks = json.loads((parent / 'checks.json').read_text())
    if not inherited_checks or not all(x is True for x in inherited_checks.values()):
        raise ValueError('Parent initial hook checks failed')
    folder = lab.ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-MAP-005-resumed-offsets')
    folder.mkdir()
    hashes = lab.archive_sources(folder, ['scripts/resume_calibrated.py', 'scripts/calibrated_map.py',
        'scripts/calibration.py', 'scripts/lab.py', 'scripts/storage.py', 'scripts/measurements.py',
        'uv.lock', 'configs/model.json', cfg['evaluation_prompts']])
    for name, expected in metadata['source_hashes'].items():
        new_name = f'parent-source/{name}'
        destination = folder / 'source' / new_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((parent / 'source' / name).read_bytes())
        hashes[new_name] = expected
    for name in ['metadata.json', 'outputs.json', 'checks.json', 'trial-order.json', 'selected.json', 'baseline-prefill.pt']:
        data = (parent / name).read_bytes()
        destination = folder / 'source/parent' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[f'parent/{name}'] = hashlib.sha256(data).hexdigest()
    (folder / 'source/experiment-config.json').write_bytes(cfg_bytes)
    if (folder / 'source' / cfg['evaluation_prompts']).read_bytes() != (parent / 'source' / cfg['evaluation_prompts']).read_bytes():
        raise ValueError('Current prompts differ from parent archive')
    metadata.update(source_hashes=hashes, resume={'parent_run': parent.name, 'inherited_replies': len(outputs),
        'inherited_intervention_trials': len(done), 'remaining_trials': len(trials)-len(done),
        'reason': 'Parent atomic replacement failed with Windows PermissionError; prior complete JSON preserved.',
        'limitation': 'Interrupted segment has no final parameter-version check. Resumed segment is checked separately.',
        'revalidation_file': 'resume-revalidation.json'})
    lab.write_json(folder / 'metadata.json', metadata)
    lab.write_json(folder / 'outputs.json', outputs)
    lab.write_json(folder / 'trial-order.json', trials)
    lab.write_json(folder / 'selected.json', selected)
    print(f'RUN {folder}', flush=True)
    model, tokenizer = lab.load()
    versions = {name: p._version for name, p in model.named_parameters()}
    reference = torch.load(folder / 'source/parent/baseline-prefill.pt', map_location='cpu', weights_only=True)
    layers = sorted({x['layer'] for x in selected})
    revalidation, checks = [], dict(inherited_checks)
    checks.update(replayed_baseline_tokens_identical=True, replayed_prefill_tensors_identical=True)
    def generate(item):
        return lab.generate(model, tokenizer, item['prompt'], system=cfg['system'], max_new_tokens=cfg['max_new_tokens'])
    for item in prompts:
        with capture_prefill(model, layers) as captured:
            result = generate(item)
        checks['replayed_baseline_tokens_identical'] &= result['token_ids'] == baseline[item['id']]['token_ids']
        checks['replayed_prefill_tensors_identical'] &= all(torch.equal(captured[layer], reference[item['id']][layer]) for layer in layers)
        result.update(prompt_id=item['id'], phase='resume-check', condition='replayed-baseline')
        revalidation.append(result)
        lab.write_json(folder / 'resume-revalidation.json', revalidation)
        print(f"REVALIDATE {item['id']}", flush=True)
    first = selected[0]
    with offset_hook(model, first['layer'], first['neuron'], 0):
        zero = generate(prompts[0])
    restored = generate(prompts[0])
    checks['resumed_zero_offset_token_identical'] = zero['token_ids'] == baseline[prompts[0]['id']]['token_ids']
    checks['resumed_removed_hook_token_identical'] = restored['token_ids'] == baseline[prompts[0]['id']]['token_ids']
    for name, result in [('zero-offset', zero), ('removed-hook', restored)]:
        result.update(prompt_id=prompts[0]['id'], phase='resume-check', condition=name)
        revalidation.append(result)
    lab.write_json(folder / 'resume-revalidation.json', revalidation)
    lab.write_json(folder / 'checks.json', checks)
    if not all(checks.values()):
        raise RuntimeError('Exact resumed baseline/prefill/hook revalidation failed; no new interventions generated')
    lookup = {(x['candidate'], x['offset']): x for x in selected}
    prompt_lookup = {x['id']: x for x in prompts}
    special = set(tokenizer.all_special_ids)
    for index, trial in enumerate(trials[len(done):], start=len(done)+1):
        unit, item = lookup[trial['candidate'], trial['offset']], prompt_lookup[trial['prompt_id']]
        with capture_prefill(model, [unit['layer']]) as captured:
            with offset_hook(model, unit['layer'], unit['neuron'], unit['offset']):
                result = generate(item)
        mask = torch.tensor([i > 0 and token not in special for i, token in enumerate(result['input_token_ids'])])
        original = reference[item['id']][unit['layer']][mask]
        actual = ((captured[unit['layer']][mask] - original).norm(dim=-1) / original.norm(dim=-1).clamp_min(1e-12)).mean().item()
        result.update(phase='probe', condition='channel-offset', prompt_id=item['id'], cohort=item['cohort'], domain=item['domain'],
            **{key: unit[key] for key in ('candidate', 'layer', 'neuron', 'operator', 'sign', 'target_fraction', 'offset', 'offset_in_rms_units')},
            actual_prefill_fraction=actual, screen_actual_fraction=unit['mean_actual_fraction'], role='calibrated-median',
            token_identical_to_baseline=result['token_ids'] == baseline[item['id']]['token_ids'], measurements=measure(result, item))
        outputs.append(result)
        lab.write_json(folder / 'outputs.json', outputs)
        print(f"TRIAL {index}/{len(trials)} {unit['candidate']} {item['id']}", flush=True)
    checks['resumed_parameter_version_counters_unchanged'] = versions == {name: p._version for name, p in model.named_parameters()}
    lab.write_json(folder / 'checks.json', checks)
    lab.write_json(folder / 'summary.json', {'completed': all(checks.values()), 'checks': checks,
        'calibration_run': metadata['calibration_run'], 'selected_channels': len(layers),
        'intervention_replies': len(trials), 'generations': len(outputs), 'extra_resume_check_generations': len(revalidation),
        'changed_token_sequences': sum(x.get('token_identical_to_baseline') is False for x in outputs),
        'resume': metadata['resume']})
    if not all(checks.values()):
        raise RuntimeError('Resumed parameter version check failed')
    print(f'COMPLETE {folder}', flush=True)


if __name__ == '__main__':
    main()
