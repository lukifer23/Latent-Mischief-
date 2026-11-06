"""Collect activation statistics, calibrate offsets, and screen their actual forward effects."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import lab
from calibration import calibrate_offset, offset_hook, select_candidates

ROOT = lab.ROOT


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/calibrated-mapping.json')
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    cfg = json.loads(config_bytes)
    prompts = json.loads((ROOT / cfg['calibration_prompts']).read_text(encoding='utf-8'))
    if not set(cfg['screen_prompt_ids']) <= {x['id'] for x in prompts}:
        raise ValueError('Screen IDs must belong to calibration inputs')
    folder = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-CAL-001-calibration')
    folder.mkdir(parents=True)
    print(f'RUN {folder}', flush=True)
    model, tokenizer = lab.load()
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    hashes = lab.archive_sources(folder, ['scripts/calibrate.py', 'scripts/calibration.py', 'scripts/lab.py',
        'scripts/storage.py', 'uv.lock', 'configs/model.json', cfg['calibration_prompts'], cfg['evaluation_prompts']])
    (folder / 'source/experiment-config.json').write_bytes(config_bytes)
    lab.write_json(folder / 'metadata.json', {
        'experiment': 'CAL-001', 'config': cfg, 'model': lab.config(), 'environment': lab.environment(),
        'source_hashes': hashes, 'experiment_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'calibration_unit': 'mean relative L2 change of post_ffn_norm output, not whole residual stream',
        'rms_positions': 'all non-special input positions except position zero, including neutral system tokens',
        'kl': 'KL(baseline next-token distribution || intervened next-token distribution), natural logarithm',
        'note': 'No generated text is used for screening or candidate selection. Float32 local approximation is checked by actual BF16 forwards.',
    })
    prepared, masks = {}, {}
    special = set(tokenizer.all_special_ids)
    for item in prompts:
        messages = [{'role': 'system', 'content': cfg['system']}, {'role': 'user', 'content': item['prompt']}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=lab.config()['thinking'], return_tensors='pt', return_dict=True).to(model.device)
        ids = inputs['input_ids'][0].tolist()
        mask = torch.tensor([i > 0 and token not in special for i, token in enumerate(ids)], device=model.device)
        if not mask.any():
            raise ValueError('No calibration positions')
        prepared[item['id']], masks[item['id']] = inputs, mask
    lab.write_json(folder / 'calibration-inputs.json', {key: {'input_ids': value['input_ids'][0].tolist(),
        'selected_positions': masks[key].nonzero().flatten().tolist()} for key, value in prepared.items()})
    sums, maxima, counts, raw, post = {}, {}, {}, {}, {}
    for i, layer in enumerate(model.model.layers):
        width = layer.feed_forward.w2.in_features
        sums[i] = torch.zeros(width, device=model.device)
        maxima[i] = torch.zeros(width, device=model.device)
        counts[i] = 0
        if i in cfg['layers']:
            raw[i], post[i] = {}, {}
    active, handles = {}, []
    def statistics(layer):
        def capture(_module, args):
            values = args[0][0, active['mask']].detach().float()
            sums[layer].add_(values.square().sum(dim=0))
            maxima[layer].copy_(torch.maximum(maxima[layer], values.abs().amax(dim=0)))
            counts[layer] += values.shape[0]
        return capture
    def cache(layer, destination):
        def capture(_module, _args, output):
            destination[layer][active['id']] = output[0, active['mask']].detach().float().clone()
        return capture
    baseline_logs = {}
    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.feed_forward.w2.register_forward_pre_hook(statistics(i)))
            if i in cfg['layers']:
                handles.append(layer.feed_forward.w2.register_forward_hook(cache(i, raw)))
                handles.append(layer.post_ffn_norm.register_forward_hook(cache(i, post)))
        with torch.inference_mode():
            for item in prompts:
                key = item['id']
                active.update(id=key, mask=masks[key])
                logits = model(**prepared[key], use_cache=False, logits_to_keep=1).logits[0, -1].float()
                baseline_logs[key] = logits.log_softmax(dim=-1).clone()
    finally:
        for handle in handles:
            handle.remove()
    rms = {i: (sums[i] / counts[i]).sqrt() for i in sums}
    torch.save({'rms': {i: x.cpu() for i, x in rms.items()},
                'max_abs': {i: x.cpu() for i, x in maxima.items()}, 'position_counts': counts}, folder / 'activation-statistics.pt')
    lab.write_json(folder / 'activation-summary.json', {
        'layers': len(sums), 'channels': sum(x.numel() for x in sums.values()), 'position_counts': counts,
        'per_layer_rms_min_median_max': {i: [rms[i].min().item(), rms[i].median().item(), rms[i].max().item()] for i in rms}})
    chooser = random.Random(cfg['seed'])
    candidates = [{'layer': layer, 'neuron': neuron, 'candidate': f'L{layer:02d}N{neuron:04d}',
                   'operator': model.config.layer_types[layer]}
                  for layer in cfg['layers'] for neuron in chooser.sample(range(rms[layer].numel()), cfg['candidates_per_layer'])]
    lab.write_json(folder / 'candidates.json', candidates)
    check_prompt = cfg['screen_prompt_ids'][0]
    first = candidates[0]
    with torch.inference_mode():
        with offset_hook(model, first['layer'], first['neuron'], 0.0):
            zero = model(**prepared[check_prompt], use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        restored = model(**prepared[check_prompt], use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
    checks = {'zero_offset_logits_identical': torch.equal(zero, baseline_logs[check_prompt]),
              'removed_hook_logits_identical': torch.equal(restored, baseline_logs[check_prompt])}
    lab.write_json(folder / 'checks.json', checks)
    if not all(checks.values()):
        raise RuntimeError('Offset neutrality/cleanup failed')
    combined = {i: torch.cat(list(raw[i].values()), dim=0) for i in cfg['layers']}
    rows = []
    with torch.inference_mode():
        for index, candidate in enumerate(candidates):
            layer, neuron = candidate['layer'], candidate['neuron']
            block = model.model.layers[layer]
            direction = block.feed_forward.w2.weight[:, neuron].detach().float()
            weight = block.post_ffn_norm.weight.detach().float()
            epsilon = block.post_ffn_norm.variance_epsilon
            native_rms = rms[layer][neuron].item()
            for target in cfg['target_fractions']:
                for sign in (-1, 1):
                    row = {**candidate, 'target_fraction': target, 'sign': sign, 'activation_rms': native_rms}
                    if native_rms <= 1e-12:
                        row.update(offset=0, hit_cap=True, estimated_fraction=0, mean_actual_fraction=None,
                                   mean_kl=None, per_prompt=[], skip_reason='zero calibration RMS')
                        rows.append(row)
                        continue
                    row.update(calibrate_offset(combined[layer], direction, weight, epsilon, native_rms,
                        target, sign, cfg['max_offset_rms'], cfg['search_iterations']))
                    row['offset_in_rms_units'] = row['offset'] / native_rms
                    observed = []
                    for key in cfg['screen_prompt_ids']:
                        captured = {}
                        def capture(_module, _args, output):
                            captured['post'] = output[0, masks[key]].detach().float()
                        handle = block.post_ffn_norm.register_forward_hook(capture)
                        try:
                            with offset_hook(model, layer, neuron, row['offset']):
                                q = model(**prepared[key], use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
                        finally:
                            handle.remove()
                        p = baseline_logs[key]
                        kl = (p.exp() * (p - q)).sum().item()
                        if not torch.isfinite(q).all() or kl < -1e-5:
                            raise RuntimeError('Invalid screened distribution')
                        fraction = ((captured['post'] - post[layer][key]).norm(dim=-1) /
                                    post[layer][key].norm(dim=-1).clamp_min(1e-12)).mean().item()
                        observed.append({'prompt_id': key, 'actual_fraction': fraction, 'kl': max(0.0, kl), 'kl_roundoff_clamped': kl < 0})
                    row.update(per_prompt=observed, mean_actual_fraction=sum(x['actual_fraction'] for x in observed) / len(observed),
                               mean_kl=sum(x['kl'] for x in observed) / len(observed))
                    rows.append(row)
            lab.write_json(folder / 'screening.json', rows)
            print(f"SCREEN {index + 1}/{len(candidates)} {candidate['candidate']}", flush=True)
    selected = select_candidates(rows, cfg['layers'], cfg['target_fractions'],
                                 cfg['selection_tolerance_fraction'], cfg['selection_max_mean_kl'])
    lab.write_json(folder / 'selected.json', selected)
    checks['parameter_version_counters_unchanged'] = versions == {name: p._version for name, p in model.named_parameters()}
    lab.write_json(folder / 'checks.json', checks)
    lab.write_json(folder / 'summary.json', {
        'completed': True, 'stage': 'calibration-screen', 'activation_channels': sum(x.numel() for x in sums.values()),
        'sampled_channels': len(candidates), 'screen_conditions': len(rows),
        'screen_forwards': sum(len(x['per_prompt']) for x in rows), 'selected_channels': len(selected) // 2,
        'checks': checks, 'screening_sha256': hashlib.sha256((folder / 'screening.json').read_bytes()).hexdigest(),
        'selected_sha256': hashlib.sha256((folder / 'selected.json').read_bytes()).hexdigest()})
    if not all(checks.values()):
        raise RuntimeError('Parameter version invariant failed')
    print(f'COMPLETE {folder}', flush=True)


if __name__ == '__main__':
    main()
