"""Plan or run LFM-012: does an ordinary persona prompt transfer as a residual-stream direction?"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import lab
from phase_design import task_score
from storage import write_json
from transfer_design import PERSONAS, TEXTS, grid_table, load_plan, select_cell, sha

SOURCES = ['scripts/transfer_experiment.py', 'scripts/transfer_design.py', 'scripts/transfer_steering.py',
           'scripts/transfer_report.py', 'scripts/phase_design.py', 'scripts/lab.py', 'scripts/resource_guard.py',
           'scripts/storage.py', 'scripts/measurements.py', 'scripts/verify_model.py', 'scripts/viewer.py',
           'scripts/runtime.py', 'scripts/token_stream.py', 'pyproject.toml', 'uv.lock']


def source_names(plan):
    return list(dict.fromkeys([*SOURCES, plan['config_path'], plan['config']['protocol_report'], *plan['config']['input_sha256']]))


def build_vectors(features, extract_ids, cfg, hidden, layers_total):
    """features[(prompt_id, text)] -> tensor[layers, hidden]. Returns unit directions, common scale and diagnostics."""
    import torch
    stack = lambda text, ids: torch.stack([features[(i, text)] for i in ids])
    neutral = stack('neutral', extract_ids)
    raw, diagnostics = {}, {}
    for persona in PERSONAS:
        pairs = stack(persona, extract_ids) - neutral
        raw[persona] = pairs.mean(dim=0)
        diagnostics[persona] = {'pair_cosine_to_mean': torch.nn.functional.cosine_similarity(
            pairs, raw[persona][None], dim=-1).mean(dim=0).tolist()}
    half = len(extract_ids) // 2
    raw['content'] = neutral[:half].mean(dim=0) - neutral[half:].mean(dim=0)
    generator = torch.Generator(device='cpu').manual_seed(cfg['random_seed'])
    for i in range(cfg['random_vectors']):
        raw['random-' + str(i + 1)] = torch.randn(layers_total, hidden, generator=generator)
    norms = {name: value.norm(dim=-1) for name, value in raw.items()}
    if any((n <= 1e-8).any() for n in norms.values()):
        raise ValueError('Degenerate direction')
    unit = {name: value / norms[name][:, None] for name, value in raw.items()}
    scale = (norms['villain'] + norms['hamster']) / 2
    cosine = lambda a, b: (unit[a] * unit[b]).sum(dim=-1).tolist()
    diagnostics.update(natural_norm={k: v.tolist() for k, v in norms.items() if not k.startswith('random-')},
                       common_scale=scale.tolist(), mean_neutral_reply_state_norm=neutral.mean(dim=0).norm(dim=-1).tolist(),
                       cosine={'villain~hamster': cosine('villain', 'hamster'), 'villain~content': cosine('villain', 'content'),
                               'hamster~content': cosine('hamster', 'content'), 'villain~random-1': cosine('villain', 'random-1')})
    return unit, scale, diagnostics


def run(plan):
    import psutil
    import torch
    import runtime
    from measurements import measure
    from resource_guard import ResourceGuard, ResourceLimitError, snapshot
    from transfer_steering import forced_nll, reply_means, residual_hook
    from verify_model import verify_identity
    from viewer import gpu_job_running
    cfg = plan['config']
    selected = runtime.select()
    if selected.backend != 'mps' or selected.dtype_name != 'bfloat16' or sys.platform != 'darwin':
        raise ValueError('LFM-012 is frozen for macOS MPS/BF16; predeclare a separate backend experiment elsewhere')
    if selected.identity()['mps_cpu_fallback'] != '0' or selected.identity()['mps_fast_math'] != '0':
        raise ValueError('Default math and no MPS CPU fallback are required')
    own = [os.getpid(), *[p.pid for p in psutil.Process().parents()]]
    if gpu_job_running(ignore_pids=own):
        raise RuntimeError('Another lab inference process is active')
    sources = source_names(plan)
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *sources], cwd=lab.ROOT)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=lab.ROOT, text=True).splitlines())
    if diff or any(name not in tracked for name in sources):
        raise ValueError('Commit the frozen implementation, config and protocol before inference')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    os.nice(10)
    torch.mps.set_per_process_memory_fraction(cfg['resources']['mps_memory_fraction'])
    folder = lab.ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + cfg['experiment_id'] + '-prompt-transfer')
    folder.mkdir()
    hashes = lab.archive_sources(folder, sources)
    write_json(folder / 'metadata.json', {'experiment': cfg['experiment_id'], 'model_key': 'lfm3', 'model': lab.config(),
        'environment': lab.environment(), 'experiment_config': cfg, 'experiment_config_path': plan['config_path'],
        'experiment_config_sha256': plan['config_sha256'], 'source_hashes': hashes,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=lab.ROOT, text=True).strip(),
        'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens'], 'seed': cfg['seed'], 'use_cache': True},
        'allocator_limit_bytes': int(torch.mps.recommended_max_memory() * cfg['resources']['mps_memory_fraction']),
        'process_nice': psutil.Process().nice(), 'cpu_threads': torch.get_num_threads(), 'scope': cfg['scope']})
    started = time.perf_counter()
    guard = ResourceGuard(folder, selected, cfg['resources'], started + cfg['wall_seconds'])
    outputs, context_rows, grid_rows, test_rows, checks, timings = [], [], [], [], {}, {}
    model = tokenizer = None
    status, stop_reason = 'running', None

    def progress():
        write_json(folder / 'progress.json', {'status': status, 'stage': guard.stage, 'generations': len(outputs),
            'planned_generations': plan['generations'], 'grid_forwards': len(grid_rows), 'test_forwards': len(test_rows),
            'elapsed_seconds': time.perf_counter() - started})

    def stage(name):
        guard.check()
        timings[guard.stage] = time.perf_counter() - started
        guard.stage = name
        progress()
        print('STAGE ' + name, flush=True)

    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('Signal requested stop')))
    print('RUN ' + folder.name, flush=True)
    try:
        guard.start()
        stage('weight identity')
        write_json(folder / 'identity.json', verify_identity())
        stage('loading model')
        model, tokenizer = lab.load()
        if model.config.model_type != 'lfm3' or model.generation_config.num_beams not in (None, 1):
            raise ValueError('Unexpected model or beam policy')
        if max(cfg['layers']) >= len(model.model.layers):
            raise ValueError('Frozen layer outside checkpoint')
        versions = {name: p._version for name, p in model.named_parameters()}
        write_json(folder / 'generation-defaults.json', {**model.generation_config.to_dict(), '_tokenizer_eos_token_id': tokenizer.eos_token_id})
        special = set(tokenizer.all_special_ids)
        items = {p['id']: p for p in plan['prompts']}
        split = lambda name: [p['id'] for p in plan['prompts'] if p['split'] == name]
        contexts = {}

        def context(prompt_id, system_key):
            key = (prompt_id, system_key)
            if key not in contexts:
                ids = tokenizer.apply_chat_template([{'role': 'system', 'content': cfg['systems'][system_key]},
                    {'role': 'user', 'content': items[prompt_id]['prompt']}], tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors='pt', return_dict=True)['input_ids'][0].tolist()
                if len(ids) > cfg['max_input_tokens']:
                    raise ValueError('Tokenized input exceeds the frozen cap')
                contexts[key] = ids
            return contexts[key]

        def generate(prompt_id, system_key, condition, **labels):
            guard.check()
            item = items[prompt_id]
            result = lab.generate(model, tokenizer, item['prompt'], system=cfg['systems'][system_key],
                max_new_tokens=cfg['max_new_tokens'], check=guard.check, max_input_tokens=cfg['max_input_tokens'])
            if result['input_token_ids'] != context(prompt_id, system_key):
                raise RuntimeError('Generation and scoring contexts differ')
            result.update(prompt_id=prompt_id, split=item['split'], condition=condition, system_key=system_key,
                task_score=task_score(result, item), measurements=measure(result, item), **labels)
            outputs.append(result)
            write_json(folder / 'outputs.json', outputs)
            progress()
            return result

        def score_contexts(result):
            """Unsteered likelihood of one generated reply under each system prompt."""
            if not result['token_ids']:
                raise ValueError('Empty reply cannot be scored')
            nll = {}
            for key in TEXTS:
                guard.check()
                nll[key] = forced_nll(model, context(result['prompt_id'], key), result['token_ids'])
                context_rows.append({'prompt_id': result['prompt_id'], 'split': result['split'], 'condition': result['condition'],
                                     'context': key, **nll[key]})
            result['context_nll'] = nll
            result['style_llr_per_token'] = {p: (nll['neutral']['nll_sum'] - nll[p]['nll_sum']) / nll[p]['tokens'] for p in PERSONAS}

        stage('reference generation')
        reference = {}
        for prompt_id in items:
            for key in TEXTS:
                reference[(prompt_id, key)] = generate(prompt_id, key, 'reference-' + key)
        stage('extraction')
        features = {}
        for prompt_id in split('extract'):
            for key in TEXTS:
                guard.check()
                row = reference[(prompt_id, key)]
                positions = [row['input_tokens'] + i for i, token in enumerate(row['token_ids']) if token not in special]
                if not positions:
                    raise ValueError('No non-special reply tokens for extraction')
                ids = torch.tensor([row['input_token_ids'] + row['token_ids']], device=model.device)
                features[(prompt_id, key)] = reply_means(model, ids, positions)
                row['feature_tokens'] = len(positions)
        unit, scale, diagnostics = build_vectors(features, split('extract'), cfg, model.config.hidden_size, len(model.model.layers))
        torch.save({'unit': unit, 'common_scale': scale, 'model': lab.config()}, folder / 'vectors.pt')
        write_json(folder / 'vector-diagnostics.json', diagnostics)
        stage('reference scoring')
        for prompt_id in [*split('select'), *split('test'), *split('check')]:
            for key in TEXTS:
                score_contexts(reference[(prompt_id, key)])
        write_json(folder / 'context-nll.json', context_rows)
        write_json(folder / 'outputs.json', outputs)
        table = lambda context_key: {(r['prompt_id'], r['condition'].removeprefix('reference-')): {'nll_sum': r['nll_sum'], 'tokens': r['tokens']}
                                     for r in context_rows if r['condition'].startswith('reference-') and
                                     (r['context'] == r['condition'].removeprefix('reference-') if context_key == 'own' else r['context'] == 'neutral')}
        baseline, prompted = table('neutral'), table('own')

        def steered_rows(prompt_ids, layer, alpha, sink, split_name):
            for vector in plan['vectors']:
                for prompt_id in prompt_ids:
                    for text in TEXTS:
                        guard.check()
                        reply = reference[(prompt_id, text)]['token_ids']
                        with residual_hook(model, layer, unit[vector][layer], alpha * scale[layer].item()) as trace:
                            value = forced_nll(model, context(prompt_id, 'neutral'), reply)
                        if trace['forwards'] != 1 or trace['positions'] != len(context(prompt_id, 'neutral')) + len(reply):
                            raise RuntimeError('Intervention did not cover every position exactly once')
                        sink.append({'split': split_name, 'layer': layer, 'alpha': alpha, 'vector': vector,
                                     'prompt_id': prompt_id, 'text': text, **value})

        stage('select grid')
        for layer in cfg['layers']:
            for alpha in cfg['alphas']:
                steered_rows(split('select'), layer, alpha, grid_rows, 'select')
                write_json(folder / 'grid-nll.json', grid_rows)
                progress()
                print('CELL L' + str(layer) + ' a' + str(alpha) + ' ' + str(len(grid_rows)) + '/' + str(plan['grid_forwards']), flush=True)
        grid = grid_table(grid_rows, baseline, prompted, plan['vectors'])
        chosen = {p: {k: select_cell(grid, p)[k] for k in ('layer', 'alpha')} for p in PERSONAS}
        write_json(folder / 'selection.json', {'rule': 'argmax own-vector gap closed on the select split; ties to smaller alpha, then earlier layer',
                                                'selected': chosen, 'grid': grid})
        print('SELECTED ' + json.dumps(chosen), flush=True)
        stage('test teacher-forced')
        for persona in PERSONAS:
            steered_rows(split('test'), chosen[persona]['layer'], chosen[persona]['alpha'], test_rows, 'test')
            for row in test_rows:
                row.setdefault('selected_for', persona)
            write_json(folder / 'test-nll.json', test_rows)
        stage('hook checks')
        first = split('test')[0]
        cell = chosen['villain']
        with residual_hook(model, cell['layer'], unit['villain'][cell['layer']], 0.0):
            zero = generate(first, 'neutral', 'check-zero-scale')
        restored = generate(first, 'neutral', 'check-removed-hook')
        checks['zero_scale_token_identical'] = zero['token_ids'] == reference[(first, 'neutral')]['token_ids']
        checks['removed_hook_token_identical'] = restored['token_ids'] == reference[(first, 'neutral')]['token_ids']
        write_json(folder / 'checks.json', checks)
        if not all(checks.values()):
            raise RuntimeError('Residual hook neutrality/removal check failed')
        stage('steered generation')
        for persona in PERSONAS:
            cell = chosen[persona]
            for vector, condition in ((persona, 'steer-' + persona), ('random-1', 'steer-' + persona + '-random')):
                for prompt_id in [*split('test'), *split('check')]:
                    with residual_hook(model, cell['layer'], unit[vector][cell['layer']], cell['alpha'] * scale[cell['layer']].item()) as trace:
                        result = generate(prompt_id, 'neutral', condition, layer=cell['layer'], alpha=cell['alpha'], vector=vector)
                    if trace['positions'] != result['input_tokens'] + result['output_tokens'] - 1:
                        raise RuntimeError('Generation intervention positions differ from the all-position policy')
                    result['hook_trace'] = trace
                    score_contexts(result)
                    write_json(folder / 'outputs.json', outputs)
                    write_json(folder / 'context-nll.json', context_rows)
        guard.check()
        checks['parameter_versions_unchanged'] = versions == {name: p._version for name, p in model.named_parameters()}
        write_json(folder / 'checks.json', checks)
        timings[guard.stage] = time.perf_counter() - started
        status = 'complete'
        write_json(folder / 'summary.json', {'completed': all(checks.values()), 'generations': len(outputs),
            'generated_tokens': sum(x['output_tokens'] for x in outputs), 'grid_forwards': len(grid_rows),
            'test_forwards': len(test_rows), 'checks': checks, 'selected': chosen, 'stage_end_seconds': timings,
            'vectors_sha256': sha(folder / 'vectors.pt'), 'elapsed_seconds': time.perf_counter() - started})
        progress()
        print('COMPLETE ' + folder.name, flush=True)
    except BaseException as error:
        status, stop_reason = 'interrupted', str(error)
        write_json(folder / 'interruption.json', {'type': type(error).__name__, 'reason': stop_reason, 'stage': guard.stage,
            'generations': len(outputs), 'grid_forwards': len(grid_rows), 'test_forwards': len(test_rows),
            'last_resources': guard.last, 'elapsed_seconds': time.perf_counter() - started})
        progress()
        if not isinstance(error, ResourceLimitError):
            raise
        print('RESOURCE STOP ' + stop_reason, flush=True)
    finally:
        guard.close()
        model = tokenizer = None
        gc.collect()
        selected.empty_cache()
        try:
            cleanup = {'model_reference_released': True, 'resources': snapshot(folder, selected)}
        except Exception as error:
            cleanup = {'model_reference_released': True, 'telemetry_error': str(error)}
        write_json(folder / 'cleanup.json', cleanup)
    return 2 if stop_reason else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('--config', type=Path, default=lab.ROOT / 'configs/lfm-prompt-transfer.json')
    args = parser.parse_args()
    plan = load_plan(lab.ROOT, args.config)
    if args.command == 'plan':
        print(json.dumps({k: v for k, v in plan.items() if k != 'prompts'}, indent=2))
        return 0
    return run(plan)


if __name__ == '__main__':
    raise SystemExit(main())
