"""Verify an LFM-012 run and recompute every published number from its raw rows. No torch, no model."""
import argparse
import json
from pathlib import Path

from storage import write_json
from transfer_design import (PERSONAS, SPLITS, TEXTS, generation_verdict, grid_table, load_plan, nominate, read,
                             select_cell, sha, teacher_forced_verdict)

ROOT = Path(__file__).resolve().parents[1]


def verify_sources(folder, metadata):
    for name, digest in metadata['source_hashes'].items():
        if sha(folder / 'source' / name) != digest:
            raise ValueError('Archived source differs from its recorded hash: ' + name)
    return len(metadata['source_hashes'])


def reference_tables(context_rows):
    baseline, prompted = {}, {}
    for r in context_rows:
        if not r['condition'].startswith('reference-'):
            continue
        text = r['condition'].removeprefix('reference-')
        value = {'nll_sum': r['nll_sum'], 'tokens': r['tokens']}
        if r['context'] == 'neutral':
            baseline[(r['prompt_id'], text)] = value
        if r['context'] == text:
            prompted[(r['prompt_id'], text)] = value
    return baseline, prompted


def check_product(rows, cells, prompt_ids, vectors):
    expected = {(layer, alpha, v, p, t) for layer, alpha in cells for v in vectors for p in prompt_ids for t in TEXTS}
    actual = [(r['layer'], r['alpha'], r['vector'], r['prompt_id'], r['text']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Teacher-forced rows are not exactly the frozen product')


def neutral_shift(rows, baseline, vectors):
    """How much each vector raises the per-token NLL of the model's own default reply."""
    result = {}
    for v in vectors:
        chosen = [r for r in rows if r['vector'] == v and r['text'] == 'neutral']
        tokens = sum(r['tokens'] for r in chosen)
        result[v] = (sum(r['nll_sum'] for r in chosen) - sum(baseline[(r['prompt_id'], 'neutral')]['nll_sum'] for r in chosen)) / tokens
    return result


def analyse(folder):
    folder = Path(folder)
    metadata = read(folder / 'metadata.json')
    cfg = metadata['experiment_config']
    plan = load_plan(folder / 'source', folder / 'source' / metadata['experiment_config_path'])
    if plan['config'] != cfg or plan['config_sha256'] != metadata['experiment_config_sha256']:
        raise ValueError('Archived config differs from the recorded experiment config')
    sources = verify_sources(folder, metadata)
    summary = read(folder / 'summary.json') if (folder / 'summary.json').exists() else None
    if summary is None:
        interruption = read(folder / 'interruption.json') if (folder / 'interruption.json').exists() else None
        return {'experiment': cfg['experiment_id'], 'run': folder.name, 'status': 'interrupted', 'sources_verified': sources,
                'interruption': interruption, 'scientific_result': None}
    outputs, context_rows = read(folder / 'outputs.json'), read(folder / 'context-nll.json')
    grid_rows, test_rows = read(folder / 'grid-nll.json'), read(folder / 'test-nll.json')
    checks, diagnostics = read(folder / 'checks.json'), read(folder / 'vector-diagnostics.json')
    vectors = plan['vectors']
    split = lambda name: [p['id'] for p in plan['prompts'] if p['split'] == name]
    if len(outputs) != plan['generations'] or not all(checks.values()) or len(checks) != 3:
        raise ValueError('Generation count or hook/parameter checks incomplete')
    for row in outputs:
        if row['output_tokens'] > cfg['max_new_tokens'] or row['input_tokens'] > cfg['max_input_tokens']:
            raise ValueError('Token budget exceeded')
    references = {(r['prompt_id'], r['system_key']): r for r in outputs if r['condition'].startswith('reference-')}
    if len(references) != plan['reference_generations']:
        raise ValueError('Reference generations incomplete or duplicated')
    for r in outputs:
        if r['condition'].startswith(('steer-', 'check-')) and r['input_token_ids'] != references[(r['prompt_id'], 'neutral')]['input_token_ids']:
            raise ValueError('A steered generation did not use the default prompt tokens')
    baseline, prompted = reference_tables(context_rows)
    cells = [(layer, alpha) for layer in cfg['layers'] for alpha in cfg['alphas']]
    check_product(grid_rows, cells, split('select'), vectors)
    grid = grid_table(grid_rows, baseline, prompted, vectors)
    selected = {p: {k: select_cell(grid, p)[k] for k in ('layer', 'alpha')} for p in PERSONAS}
    if selected != read(folder / 'selection.json')['selected'] or selected != summary['selected']:
        raise ValueError('Recomputed selection differs from the run')
    result = {}
    for persona in PERSONAS:
        cell = selected[persona]
        rows = [r for r in test_rows if r['selected_for'] == persona]
        check_product(rows, [(cell['layer'], cell['alpha'])], split('test'), vectors)
        entry = grid_table(rows, baseline, prompted, vectors)[0]
        select_entry = next(e for e in grid if (e['layer'], e['alpha']) == (cell['layer'], cell['alpha']))
        forced = teacher_forced_verdict(entry, persona, cfg)
        generated = generation_verdict([r for r in outputs if r['split'] in ('test', 'check')], persona, cfg)
        steered = [r for r in outputs if r['condition'] == 'steer-' + persona]
        matched = [r for r in outputs if r['condition'] == 'steer-' + persona + '-random']
        default = [r for r in outputs if r['condition'] == 'reference-neutral' and r['split'] in ('test', 'check')]
        ordinary = [r for r in outputs if r['condition'] == 'reference-' + persona and r['split'] in ('test', 'check')]
        describe = lambda group: {
            'eos': sum(r['terminated_with_eos'] for r in group), 'replies': len(group),
            'mean_tokens': sum(r['output_tokens'] for r in group) / len(group),
            'mean_repeated_fourgram_fraction': sum(r['measurements']['repeated_fourgram_fraction'] for r in group) / len(group),
            'check_answer_matches': sum(r['task_score']['answer_match'] is True for r in group if r['split'] == 'check'),
            'check_strict_matches': sum(r['task_score']['strict_format_match'] is True for r in group if r['split'] == 'check'),
            'token_identical_to_default': sum(r['token_ids'] == references[(r['prompt_id'], 'neutral')]['token_ids'] for r in group),
            'default_context_nll_per_token': sum(r['context_nll']['neutral']['nll_sum'] for r in group) / sum(r['output_tokens'] for r in group)}
        result[persona] = {
            'selected_cell': cell,
            'select_split': {'own_gap_closed': select_entry['gap'][persona][persona]['gap_closed'],
                             'controls_gap_closed': {v: g['gap_closed'] for v, g in select_entry['gap'][persona].items() if v != persona}},
            'test_split': {'teacher_forced': forced, 'gap_closed_by_vector': {v: g for v, g in entry['gap'][persona].items()},
                           'other_persona_text_gap_closed_by_vector': {v: g['gap_closed'] for v, g in entry['gap'][[p for p in PERSONAS if p != persona][0]].items()},
                           'default_reply_nll_per_token_increase': neutral_shift(rows, baseline, vectors)},
            'generation': {**generated, 'default': describe(default), 'prompted': describe(ordinary),
                           'steered': describe(steered), 'random_matched': describe(matched),
                           'rows': [{'prompt_id': r['prompt_id'], 'split': r['split'], 'condition': r['condition'],
                                     'output_tokens': r['output_tokens'], 'eos': r['terminated_with_eos'],
                                     'style_llr_per_token': r['style_llr_per_token'], 'task_score': r['task_score'],
                                     'default_context_nll_per_token': r['context_nll']['neutral']['nll_sum'] / r['output_tokens']}
                                    for r in [*default, *ordinary, *steered, *matched]]},
            'nominated_for_fresh_replication': nominate(forced, generated)}
    layer_view = lambda values: {str(layer): values[layer] for layer in cfg['layers']}
    return {'experiment': cfg['experiment_id'], 'run': folder.name, 'status': 'complete', 'git_commit': metadata['git_commit'],
            'model': metadata['model'], 'runtime': metadata['environment']['runtime'], 'packages': metadata['environment']['packages'],
            'config_sha256': metadata['experiment_config_sha256'], 'sources_verified': sources, 'checks': checks,
            'counts': {'generations': len(outputs), 'generated_tokens': sum(r['output_tokens'] for r in outputs),
                       'select_grid_forwards': len(grid_rows), 'test_forwards': len(test_rows),
                       'elapsed_seconds': summary['elapsed_seconds'], 'stage_end_seconds': summary['stage_end_seconds']},
            'design': {'layers': cfg['layers'], 'alphas': cfg['alphas'], 'vectors': vectors, 'splits': SPLITS, 'criteria': cfg['criteria']},
            'vector_diagnostics': {'common_scale': layer_view(diagnostics['common_scale']),
                                   'mean_default_reply_state_norm': layer_view(diagnostics['mean_neutral_reply_state_norm']),
                                   'natural_norm': {k: layer_view(v) for k, v in diagnostics['natural_norm'].items()},
                                   'cosine': {k: layer_view(v) for k, v in diagnostics['cosine'].items()},
                                   'pair_cosine_to_mean': {p: layer_view(diagnostics[p]['pair_cosine_to_mean']) for p in PERSONAS}},
            'select_grid': [{'layer': e['layer'], 'alpha': e['alpha'],
                             'gap_closed': {t: {v: g['gap_closed'] for v, g in e['gap'][t].items()} for t in PERSONAS}} for e in grid],
            'result': result,
            'limits': 'One model, one backend, two prompted characters, greedy decoding, eight test prompts. The style score is the '
                      "model's own likelihood ratio, not a human rating. No capability claim."}


def figure(metrics, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    design = metrics['design']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    cell = lambda t, v, layer, alpha: next(e for e in metrics['select_grid'] if (e['layer'], e['alpha']) == (layer, alpha))['gap_closed'][t][v]
    for axis, persona in zip(axes, PERSONAS):
        values = [[cell(persona, persona, layer, alpha) for alpha in design['alphas']] for layer in design['layers']]
        image = axis.imshow(values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                axis.text(j, i, f'{max(value, -9.99):.2f}', ha='center', va='center', fontsize=9,
                          color='white' if abs(value) > .6 else '#1a1a1a')
        chosen = metrics['result'][persona]['selected_cell']
        axis.add_patch(plt.Rectangle((design['alphas'].index(chosen['alpha']) - .5, design['layers'].index(chosen['layer']) - .5), 1, 1,
                                     fill=False, edgecolor='#1a1a1a', linewidth=2.2))
        axis.set_xticks(range(len(design['alphas'])), [str(a) for a in design['alphas']])
        axis.set_yticks(range(len(design['layers'])), ['L' + str(x) for x in design['layers']])
        axis.set_xlabel('strength (multiples of the natural shift)')
        axis.set_title(persona + ' direction: share of prompt gap closed\n(select prompts; box = frozen selection)', fontsize=10)
    fig.colorbar(image, ax=axes[:2], fraction=.03, pad=.02, label='1 = equals the system prompt, 0 = no effect')
    axis = axes[2]
    names = design['vectors']
    width = .38
    for offset, persona, colour in ((-width / 2, 'villain', '#b2452f'), (width / 2, 'hamster', '#2f6f9f')):
        gaps = metrics['result'][persona]['test_split']['gap_closed_by_vector']
        axis.bar([i + offset for i in range(len(names))], [gaps[v]['gap_closed'] for v in names], width, color=colour,
                 label=persona + ' replies, cell L' + str(metrics['result'][persona]['selected_cell']['layer']) + ' x' + str(metrics['result'][persona]['selected_cell']['alpha']))
    axis.axhline(0, color='#1a1a1a', linewidth=.8)
    axis.axhline(design['criteria']['min_gap_closed'], color='#666666', linewidth=.8, linestyle='--')
    axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
    axis.set_ylabel('share of prompt gap closed')
    axis.set_title('Held-out test prompts: each direction applied\nat the selected cell (dashed = frozen threshold)', fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for suffix in ('png', 'svg'):
        fig.savefig(Path(target) / ('transfer.' + suffix), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--publish', action='store_true', help='write numeric evidence and figure under experiments/evidence')
    args = parser.parse_args()
    folder = ROOT / 'runs' / args.run
    metrics = analyse(folder)
    write_json(folder / 'report.json', metrics)
    if args.publish:
        target = ROOT / 'experiments/evidence' / metrics['experiment']
        write_json(target / 'metrics.json', metrics)
        if metrics['status'] == 'complete':
            figure(metrics, target)
    brief = {'run': metrics['run'], 'status': metrics['status']}
    if metrics['status'] == 'complete':
        brief['result'] = {p: {'cell': v['selected_cell'], 'test': v['test_split']['teacher_forced'],
                               'style': v['generation']['style_llr_mean'], 'nominated': v['nominated_for_fresh_replication']}
                           for p, v in metrics['result'].items()}
    print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()
