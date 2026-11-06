"""Export reviewed numeric evidence and reproducible scientific figures, without raw replies."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from calibration import select_candidates
from measurements import measure, repeated_fourgrams
from storage import write_json
from verify_run import verify

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_run(name):
    path = (ROOT / 'runs' / name).resolve()
    if path.parent != (ROOT / 'runs').resolve():
        raise ValueError('Use a local run directory name')
    return path


def collect(calibration, evaluation):
    screen, run = local_run(calibration), local_run(evaluation)
    meta = json.loads((screen / 'metadata.json').read_text(encoding='utf-8'))
    summary = json.loads((screen / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('completed') is not True or not all(x is True for x in summary['checks'].values()):
        raise ValueError('Calibration incomplete or failed')
    for name, expected in meta['source_hashes'].items():
        if digest(screen / 'source' / name) != expected:
            raise ValueError(f'Calibration archive changed: {name}')
    if digest(screen / 'source/experiment-config.json') != meta['experiment_config_sha256']:
        raise ValueError('Calibration config changed')
    for name in ('screening', 'selected'):
        if digest(screen / f'{name}.json') != summary[f'{name}_sha256']:
            raise ValueError(f'Calibration {name} changed')
    rows = json.loads((screen / 'screening.json').read_text(encoding='utf-8'))
    chosen = json.loads((screen / 'selected.json').read_text(encoding='utf-8'))
    cfg = meta['config']
    candidates = json.loads((screen / 'candidates.json').read_text(encoding='utf-8'))
    expected_conditions = {(x['candidate'], target, sign) for x in candidates for target in cfg['target_fractions'] for sign in (-1, 1)}
    observed_conditions = [(x['candidate'], x['target_fraction'], x['sign']) for x in rows]
    if len(observed_conditions) != len(set(observed_conditions)) or set(observed_conditions) != expected_conditions:
        raise ValueError('Calibration screen is missing or duplicates conditions')
    for row in rows:
        if row.get('skip_reason'):
            continue
        per_prompt = row['per_prompt']
        if [x['prompt_id'] for x in per_prompt] != cfg['screen_prompt_ids']:
            raise ValueError('Calibration prompt schedule differs')
        for field, source in [('mean_actual_fraction', 'actual_fraction'), ('mean_kl', 'kl')]:
            if not math.isclose(row[field], statistics.mean(x[source] for x in per_prompt), abs_tol=1e-9):
                raise ValueError('Stored screen mean differs from individual measurements')
    recomputed = select_candidates(rows, cfg['layers'], cfg['target_fractions'],
                                  cfg['selection_tolerance_fraction'], cfg['selection_max_mean_kl'])
    if recomputed != chosen:
        raise ValueError('Saved selection does not match predeclared rule')
    verified = verify(run)
    run_meta = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    if run_meta['calibration_run'] != calibration or digest(run / 'selected.json') != summary['selected_sha256']:
        raise ValueError('Evaluation does not use this calibration')
    archived_prompts = json.loads((run / 'source' / cfg['evaluation_prompts']).read_text(encoding='utf-8'))
    prompt_lookup = {x['id']: x for x in archived_prompts}
    outputs = json.loads((run / 'outputs.json').read_text(encoding='utf-8'))
    expected_trials = {(x['candidate'], x['offset'], p['id']) for x in chosen for p in archived_prompts}
    actual_trials = [(x['candidate'], x['offset'], x['prompt_id']) for x in outputs if x['condition'] == 'channel-offset']
    baselines = [x['prompt_id'] for x in outputs if x['condition'] == 'baseline' and x['phase'] == 'probe']
    if set(actual_trials) != expected_trials or len(actual_trials) != len(expected_trials) or sorted(baselines) != sorted(prompt_lookup):
        raise ValueError('Evaluation does not cover the frozen candidate/prompt product and baselines')
    if run_meta.get('resume'):
        inherited = json.loads((run / 'source/parent/outputs.json').read_text(encoding='utf-8'))
        revalidation = json.loads((run / 'resume-revalidation.json').read_text(encoding='utf-8'))
        baseline_ids = {x['prompt_id']: x['token_ids'] for x in outputs if x['condition'] == 'baseline' and x['phase'] == 'probe'}
        if outputs[:len(inherited)] != inherited or len(revalidation) != len(archived_prompts) + 2:
            raise ValueError('Resume changed inherited evidence or omitted revalidation replies')
        if [x['prompt_id'] for x in revalidation[:len(archived_prompts)]] != [x['id'] for x in archived_prompts]:
            raise ValueError('Resume baseline replay schedule differs')
        if any(x['token_ids'] != baseline_ids[x['prompt_id']] for x in revalidation):
            raise ValueError('Resume token revalidation differs from baseline')
    groups, reply_metrics = defaultdict(list), []
    for item in outputs:
        if item['phase'] != 'probe':
            continue
        if measure(item, prompt_lookup[item['prompt_id']]) != item['measurements']:
            raise ValueError('Stored reply measurements differ from recomputation')
        key = 'baseline' if item['condition'] == 'baseline' else f"{item['candidate']} {'+' if item['sign'] > 0 else '-'}"
        row = {k: item[k] for k in ('prompt_id', 'cohort', 'domain', 'output_tokens', 'terminated_with_eos')}
        row.update(condition=key, repeated_fourgram_fraction=repeated_fourgrams(item['token_ids']),
                   changed=item.get('token_identical_to_baseline') is False,
                   actual_prefill_fraction=item.get('actual_prefill_fraction'))
        checks = [value for key, value in item['measurements'].items() if key in ('exact_json_match', 'exact_text_match')]
        row['exact_check'] = checks[0] if checks else None
        groups[key].append(row)
        reply_metrics.append(row)
    aggregates = []
    for key in ['baseline'] + sorted(x for x in groups if x != 'baseline'):
        items = groups[key]
        checks = [x['exact_check'] for x in items if x['exact_check'] is not None]
        strengths = [x['actual_prefill_fraction'] for x in items if x['actual_prefill_fraction'] is not None]
        aggregates.append({'condition': key, 'n': len(items), 'eos': sum(x['terminated_with_eos'] for x in items),
            'changed': sum(x['changed'] for x in items), 'mean_tokens': statistics.mean(x['output_tokens'] for x in items),
            'mean_repeated_fourgram_fraction': statistics.mean(x['repeated_fourgram_fraction'] for x in items),
            'exact_pass': sum(checks), 'exact_n': len(checks),
            'mean_actual_prefill_fraction': statistics.mean(strengths) if strengths else None,
            'min_actual_prefill_fraction': min(strengths) if strengths else None,
            'max_actual_prefill_fraction': max(strengths) if strengths else None})
    return {'schema': 1, 'model': meta['model'], 'calibration_run': calibration, 'evaluation_run': evaluation,
        'runtime_lock_sha256': meta['environment']['lock_sha256'],
        'report_generator_sha256': digest(Path(__file__)),
        'calibration_config': cfg,
        'activation_summary': json.loads((screen / 'activation-summary.json').read_text(encoding='utf-8')),
        'resume': run_meta.get('resume'),
        'extra_resume_check_generations': json.loads((run / 'summary.json').read_text(encoding='utf-8')).get('extra_resume_check_generations', 0),
        'calibration_summary': summary, 'evaluation_verification': verified,
        'artifact_hashes': {f'{folder.name}/{name}': digest(folder / name) for folder, names in
            [(screen, ['metadata.json', 'screening.json', 'selected.json', 'summary.json']),
             (run, ['outputs.json', 'metadata.json', 'trial-order.json'])] for name in names},
        'screening': rows, 'selected': chosen, 'groups': aggregates, 'replies': reply_metrics,
        'prompts': [{'id': p['id'], 'cohort': p['cohort'], 'domain': p['domain'],
                     'exact_check': p.get('check')} for p in prompt_lookup.values()],
        'limits': ['One pinned model, one greedy trajectory per prompt/condition; exploratory, no confidence intervals.',
                   'Exact checks combine answer and strict format; other domains have no automatic correctness grade.',
                   'Relative L2 is measured at the intervened block post_ffn_norm, not the whole residual stream.',
                   'Calibration statistics include neutral system tokens; decode-position strengths are not measured.',
                   'Screen selection uses calibration KL and strength only; evaluation text does not select units.']}


def plot(data, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    destination.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for extension in ('png', 'svg'):
            fig.savefig(destination / f'{name}.{extension}', dpi=170, facecolor='white')
        plt.close(fig)
    rows, selected = data['screening'], data['selected']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout='constrained')
    colors = {-1: '#b84c32', 1: '#247c9e'}
    for sign in (-1, 1):
        for target, marker in ((.05, 'o'), (.2, '^')):
            subset = [x for x in rows if x['sign'] == sign and x['target_fraction'] == target]
            axes[0].scatter([100*x['estimated_fraction'] for x in subset], [100*x['mean_actual_fraction'] for x in subset],
                s=25, alpha=.6, c=colors[sign], marker=marker, label=f"{'Negative' if sign < 0 else 'Positive'}, target {target:.0%} (n={len(subset)})")
    caps = [x for x in rows if x['hit_cap']]
    axes[0].scatter([100*x['estimated_fraction'] for x in caps], [100*x['mean_actual_fraction'] for x in caps],
                    s=65, facecolors='none', edgecolors='black', label=f'Offset cap reached (n={len(caps)})')
    top = max(22, max(x['mean_actual_fraction'] for x in rows)*105)
    axes[0].plot([0, top], [0, top], color='#777777', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Float32 predicted local change (%)', ylabel='Actual BF16 mean local change (%)',
                title='Calibration screen: all 384 conditions', xlim=(0, top), ylim=(0, top))
    axes[0].legend(fontsize=8, loc='upper left')
    for sign in (-1, 1):
        subset = sorted([x for x in selected if x['sign'] == sign], key=lambda x: x['layer'])
        axes[1].plot([x['layer'] for x in subset], [x['mean_kl'] for x in subset], 'o-',
                     color=colors[sign], label=f"{'Negative' if sign < 0 else 'Positive'} offset")
    axes[1].set(yscale='log', xlabel='Selected channel layer', ylabel='Mean next-token KL (nats; log scale)',
                xticks=sorted({x['layer'] for x in selected}), title='Equal local targets, unequal distribution shifts')
    axes[1].legend(fontsize=9)
    fig.suptitle('LFM3-1.7B · CAL-001 · 96 channels, four screen prompts per condition', fontsize=13)
    fig.supxlabel('Local change = mean relative L2 at post_ffn_norm. Selected units target 20%; KL is not a quality score.', fontsize=9)
    save(fig, 'calibration')

    groups = data['groups']
    labels = [x['condition'] for x in groups]
    y = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.4), sharey=True, layout='constrained')
    palette = ['#777777' if x == 'baseline' else colors[-1 if x.endswith('-') else 1] for x in labels]
    axes[0].barh(y, [x['changed'] for x in groups], color=palette)
    axes[0].set(xlim=(0, 17), xlabel='Changed token sequences / 16 prompts', yticks=y, yticklabels=labels)
    for index, row in enumerate(groups):
        axes[0].text(row['changed']+.15, index, 'reference' if index == 0 else str(row['changed']), va='center', fontsize=8)
    axes[1].barh(y, [x['exact_pass'] for x in groups], color=palette)
    axes[1].set(xlim=(0, 6.8), xlabel='Strict exact checks passed / 6 checks')
    for index, row in enumerate(groups):
        axes[1].text(row['exact_pass']+.1, index, str(row['exact_pass']), va='center', fontsize=8)
    axes[2].barh(y, [16-x['eos'] for x in groups], color=palette)
    axes[2].set(xlim=(0, max(3, max(16-x['eos'] for x in groups)+1)), xlabel='Replies without EOS / 16 prompts')
    for index, row in enumerate(groups):
        axes[2].text(16-row['eos']+.1, index, str(16-row['eos']), va='center', fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('MAP-005 · Diverse evaluation: sequence changes, strict checks, and completion', fontsize=13)
    fig.supxlabel('16 disjoint prompts; six strict checks. Greedy, cap 192. No personality score. Resumed run: see provenance in report.'
                  if data.get('resume') else '16 disjoint prompts; six strict checks. Greedy, cap 192. No personality score.', fontsize=9)
    save(fig, 'evaluation')

    labels = labels[1:]
    prompts = [x['id'] for x in data['prompts']]
    lookup = {(x['condition'], x['prompt_id']): x for x in data['replies']}
    matrix = np.array([[int(lookup[label, prompt]['changed']) for prompt in prompts] for label in labels])
    fig, ax = plt.subplots(figsize=(13, 6.8), layout='constrained')
    im = ax.imshow(matrix, cmap=ListedColormap(['#ececec', '#247c9e']), vmin=-.5, vmax=1.5, aspect='auto')
    ax.set(xticks=range(len(prompts)), xticklabels=prompts, yticks=range(len(labels)), yticklabels=labels,
           title='MAP-005 · Token-sequence changes by prompt and signed offset')
    plt.setp(ax.get_xticklabels(), rotation=50, ha='right', rotation_mode='anchor')
    ax.axvline(5.5, color='#222222', linewidth=1.5)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=.55)
    cbar.ax.set_yticklabels(['Identical', 'Changed'])
    fig.supxlabel('First six columns: strict checks. Remaining ten: exploratory domains. Blue measures difference, not quality.', fontsize=9)
    save(fig, 'change-map')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration')
    parser.add_argument('--evaluation')
    parser.add_argument('--evidence', type=Path, help='Redraw published numeric evidence without local model/run files')
    parser.add_argument('--destination', type=Path, default=ROOT / 'experiments/evidence/CAL-001')
    args = parser.parse_args()
    if args.evidence:
        data = json.loads(args.evidence.read_text(encoding='utf-8'))
    else:
        if not args.calibration or not args.evaluation:
            parser.error('Provide --evidence or both --calibration and --evaluation')
        data = collect(args.calibration, args.evaluation)
        write_json(args.destination / 'metrics.json', data)
    plot(data, args.destination)
    print(args.destination)


if __name__ == '__main__':
    main()
