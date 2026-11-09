"""Plan or run the frozen, short LFM3 phase-timing comparison."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import lab
from phase_design import load_plan, read, sha, task_score
from phase_timing import timed_sparse_hook
from storage import write_json


def source_names(plan):
    return ['scripts/phase_experiment.py', 'scripts/phase_design.py', 'scripts/phase_timing.py', 'scripts/phase_report.py',
            'scripts/phase_calibration.py', 'scripts/lab.py', 'scripts/calibration.py',
            'scripts/resource_guard.py', 'scripts/storage.py', 'scripts/measurements.py',
            'scripts/verify_model.py', 'scripts/viewer.py', 'pyproject.toml', 'uv.lock',
            plan['config_path'], plan['config']['protocol_report'], *plan['config']['input_sha256']]


def validate_trace(trace, phase, count, input_tokens):
    if trace['prefill_forwards'] != 1 or trace['decode_forwards'] != count - 1:
        raise ValueError('Actual forward phases differ from the cached-generation protocol')
    expected_prefill = int(phase in ('prefill', 'all'))
    expected_decode = count - 1 if phase in ('decode', 'all') else 0
    if trace['prefill_interventions'] != expected_prefill or trace['decode_interventions'] != expected_decode:
        raise ValueError('Intervention occurred during the wrong forward phase')
    if trace['intervened_positions'] != expected_prefill * input_tokens + expected_decode:
        raise ValueError('Intervened token positions differ from the phase policy')


def run(plan):
    import psutil
    import torch
    import runtime
    from measurements import measure
    from phase_calibration import calibrate_profiles, capture_post
    from resource_guard import ResourceGuard, ResourceLimitError, snapshot
    from verify_model import verify_identity
    from viewer import gpu_job_running
    cfg = plan['config']
    selected = runtime.select()
    if sys.platform != 'darwin' or selected.backend != 'mps' or selected.dtype_name != 'bfloat16':
        raise ValueError('This frozen experiment requires macOS MPS/BF16')
    if selected.identity()['mps_cpu_fallback'] != '0' or selected.identity()['mps_fast_math'] != '0':
        raise ValueError('Default math and no MPS CPU fallback are required')
    own = [os.getpid(), *[p.pid for p in psutil.Process().parents()]]
    if gpu_job_running(ignore_pids=own):
        raise RuntimeError('Another lab inference process is active')
    sources = list(dict.fromkeys([*source_names(plan), 'scripts/runtime.py', 'scripts/token_stream.py']))
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *sources], cwd=lab.ROOT)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=lab.ROOT, text=True).splitlines())
    if diff or any(name not in tracked for name in sources):
        raise ValueError('Commit the frozen implementation, config and protocol before inference')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    os.nice(10)
    torch.mps.set_per_process_memory_fraction(cfg['resources']['mps_memory_fraction'])
    folder = lab.ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-LFM-010-phase-timing')
    folder.mkdir()
    hashes = lab.archive_sources(folder, sources)
    write_json(folder / 'metadata.json', {'experiment': 'LFM-010', 'model_key': 'lfm3', 'model': lab.config(),
        'environment': lab.environment(), 'experiment_config': cfg, 'experiment_config_path': plan['config_path'],
        'experiment_config_sha256': plan['config_sha256'],
        'source_hashes': hashes, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=lab.ROOT, text=True).strip(),
        'generation': {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens'], 'seed': cfg['seed'], 'use_cache': True},
        'mps_allocator_limit_bytes': int(torch.mps.recommended_max_memory() * cfg['resources']['mps_memory_fraction']),
        'process_nice': psutil.Process().nice(), 'cpu_threads': torch.get_num_threads(), 'scope': cfg['scope']})
    write_json(folder / 'trial-order.json', plan['schedule'])
    write_json(folder / 'outputs.json', [])
    write_json(folder / 'checks.json', {})
    started = time.perf_counter()
    deadline = started + cfg['wall_seconds']
    guard = ResourceGuard(folder, selected, cfg['resources'], deadline)
    outputs, checks, baseline_tokens, baseline_post = [], {}, {}, {}
    model, tokenizer, active = None, None, None
    active_inputs, active_tokens = [], []
    status, stop_reason = 'running', None
    def progress():
        write_json(folder / 'progress.json', {'status': status, 'completed': len(outputs), 'planned': len(plan['schedule']),
            'active_trial': active, 'active_tokens': len(active_tokens), 'stage': guard.stage,
            'elapsed_seconds': time.perf_counter() - started})
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal requested stop: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    print('RUN ' + folder.name, flush=True)
    try:
        guard.start()
        guard.check()
        guard.stage = 'weight identity'
        progress()
        write_json(folder / 'identity.json', verify_identity())
        guard.check()
        guard.stage = 'loading model'
        model, tokenizer = lab.load()
        guard.check()
        if model.config.model_type != 'lfm3' or model.generation_config.num_beams != 1:
            raise ValueError('Unexpected model or beam policy')
        versions = {name: p._version for name, p in model.named_parameters()}
        write_json(folder / 'generation-defaults.json', {**model.generation_config.to_dict(),
            '_tokenizer_eos_token_id': tokenizer.eos_token_id})
        guard.stage = 'MPS calibration'
        profiles = calibrate_profiles(model, tokenizer, plan, guard.check,
            lambda rows: write_json(folder / 'profiles.json', rows), folder)
        guard.check()
        profiles = {p['role']: p for p in profiles}
        items = {p['id']: p for p in plan['prompts']}
        special = set(tokenizer.all_special_ids)
        for trial in plan['schedule']:
            guard.check()
            active, active_inputs, active_tokens = trial, [], []
            item = items[trial['prompt_id']]
            guard.stage = trial['trial_id']
            progress()
            profile = profiles.get(trial.get('profile_role', 'villain'))
            phase = trial.get('intervention_phase')
            condition = trial['condition']
            hook = phase is not None and not condition.startswith('restored-')
            offsets = [0.] * len(profile['channels']) if condition.startswith('zero-') else profile['offsets']
            context = timed_sparse_hook(model, profile['layer'], profile['channels'], offsets, phase) if hook else nullcontext(None)
            system = cfg['persona_system'] if condition == 'prompt-villain' else cfg['system']
            def checkpoint(tokens, text):
                nonlocal active_tokens
                active_tokens = list(tokens)
                write_json(folder / 'active.json', {'trial': active, 'prompt': item['prompt'], 'system': system,
                    'input_token_ids': active_inputs, 'token_ids': active_tokens, 'text': text})
                progress()
            def inputs(ids):
                nonlocal active_inputs
                active_inputs = ids
                checkpoint([], '')
            with capture_post(model, cfg['layer']) as captured:
                with context as trace:
                    result = lab.generate(model, tokenizer, item['prompt'], system=system, max_new_tokens=cfg['max_new_tokens'],
                        check=guard.check, on_tokens=checkpoint, on_inputs=inputs, max_input_tokens=cfg['max_input_tokens'])
            checkpoint(result['token_ids'], result['text'])
            if trace is not None:
                validate_trace(trace, phase, result['output_tokens'], result['input_tokens'])
            if phase == 'decode' and not condition.startswith('restored-') and result['token_ids'][0] != baseline_tokens[item['id']][0]:
                raise RuntimeError('Decode-only intervention changed the first reply token')
            if condition == 'baseline':
                baseline_tokens[item['id']] = result['token_ids']
                baseline_post[item['id']] = captured[0]
            if trial['record_phase'] == 'check':
                checks[condition + '_token_identical'] = result['token_ids'] == baseline_tokens[item['id']]
                write_json(folder / 'checks.json', checks)
                if not checks[condition + '_token_identical']:
                    raise RuntimeError('Phase neutrality/removal check failed')
            mask = torch.tensor([i > 0 and token not in special for i, token in enumerate(result['input_token_ids'])])
            # Prompted system instructions have different inputs; no paired activation ratio.
            fraction = None
            if condition != 'prompt-villain':
                ref = baseline_post[item['id']][0, mask]
                changed = captured[0][0, mask]
                fraction = ((changed - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).mean().item()
            result.update(trial_id=trial['trial_id'], phase=trial['record_phase'], condition=condition,
                prompt_id=item['id'], cohort=item['cohort'], domain=item['domain'], intervention_phase=phase,
                profile_role=trial.get('profile_role'), hook_trace=trace, task_score=task_score(result, item),
                measurements=measure(result, item), actual_prefill_fraction=fraction,
                token_identical_to_baseline=result['token_ids'] == baseline_tokens[item['id']])
            if versions != {name: p._version for name, p in model.named_parameters()}:
                raise RuntimeError('Parameters changed during the experiment')
            outputs.append(result)
            write_json(folder / 'outputs.json', outputs)
            active, active_inputs, active_tokens = None, [], []
            progress()
            print('SAVED ' + str(len(outputs)) + '/' + str(len(plan['schedule'])) + ' ' + trial['trial_id'], flush=True)
        guard.check()
        checks['parameter_versions_unchanged'] = versions == {name: p._version for name, p in model.named_parameters()}
        write_json(folder / 'checks.json', checks)
        status = 'complete'
        write_json(folder / 'summary.json', {'completed': all(checks.values()), 'generations': len(outputs),
            'generated_tokens': sum(x['output_tokens'] for x in outputs), 'checks': checks,
            'calibration_sha256': sha(folder / 'calibration.pt'), 'profiles_sha256': sha(folder / 'profiles.json')})
        progress()
    except BaseException as error:
        status, stop_reason = 'interrupted', str(error)
        write_json(folder / 'interruption.json', {'type': type(error).__name__, 'reason': stop_reason,
            'active_trial': active, 'completed': len(outputs), 'planned': len(plan['schedule']),
            'stage': guard.stage, 'last_resources': guard.last, 'elapsed_seconds': time.perf_counter() - started})
        progress()
        if not isinstance(error, ResourceLimitError):
            raise
        print('RESOURCE STOP ' + stop_reason, flush=True)
    finally:
        guard.close()
        model, tokenizer = None, None
        gc.collect()
        selected.empty_cache()
        try:
            cleanup = {'model_reference_released': model is None, 'resources': snapshot(folder, selected)}
        except Exception as error:
            cleanup = {'model_reference_released': model is None, 'telemetry_error': str(error)}
        write_json(folder / 'cleanup.json', cleanup)
    return 2 if stop_reason else 0


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('--config', type=Path, default=lab.ROOT / 'configs/lfm-phase-timing.json')
    args = parser.parse_args()
    plan = load_plan(lab.ROOT, args.config)
    if args.command == 'plan':
        print(json.dumps({k: v for k, v in plan.items() if k not in ['profiles', 'schedule', 'calibration', 'prompts']}, indent=2))
        return 0
    return run(plan)


if __name__ == '__main__':
    raise SystemExit(main())
