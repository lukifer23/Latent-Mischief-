"""Offline LFM-012 plan and arithmetic: frozen inputs, grid, gap-closed metric, selection and criteria.

Imports no torch. Every number the report publishes is recomputed from raw rows by these functions.
"""
import hashlib
import json
from pathlib import Path

PERSONAS = ('villain', 'hamster')
TEXTS = ('neutral', *PERSONAS)
SPLITS = {'extract': 16, 'select': 8, 'test': 8, 'check': 4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def vector_names(cfg):
    return [*PERSONAS, 'content', *['random-' + str(i + 1) for i in range(cfg['random_vectors'])]]


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
    prompts = read(root / cfg['prompts'])
    ids = [x['id'] for x in prompts]
    if len(set(ids)) != len(ids) or len({x['prompt'] for x in prompts}) != len(prompts):
        raise ValueError('Prompt ids and texts must be unique')
    counts = {name: sum(x['split'] == name for x in prompts) for name in SPLITS}
    if counts != SPLITS or len(prompts) != sum(SPLITS.values()):
        raise ValueError('Prompt splits differ from the frozen design')
    if any(('check' in x) != (x['split'] == 'check') for x in prompts):
        raise ValueError('Only the check split carries exact answers')
    if set(cfg['systems']) != set(TEXTS) or cfg['seed'] != current['seed'] or cfg['thinking'] is not False:
        raise ValueError('Frozen systems/seed/thinking policy changed')
    layers, alphas = cfg['layers'], cfg['alphas']
    if sorted(set(layers)) != layers or sorted(set(alphas)) != alphas or min(alphas) <= 0:
        raise ValueError('Layers and positive alphas must be sorted and distinct')
    vectors = vector_names(cfg)
    behavioural = SPLITS['extract'] + SPLITS['select'] + SPLITS['test']
    reference = (behavioural + SPLITS['check']) * len(TEXTS)
    steered = len(PERSONAS) * 2 * (SPLITS['test'] + SPLITS['check'])
    hook_checks = 2
    generations = reference + steered + hook_checks
    grid_forwards = len(layers) * len(alphas) * len(vectors) * len(TEXTS) * SPLITS['select']
    test_forwards = len(PERSONAS) * len(vectors) * len(TEXTS) * SPLITS['test']
    if generations != cfg['planned_generations'] or generations * cfg['max_new_tokens'] != cfg['total_generated_tokens']:
        raise ValueError('Generation budget differs from the full design')
    if grid_forwards + test_forwards != cfg['planned_steered_forwards']:
        raise ValueError('Teacher-forced budget differs from the full design')
    return {'schema': 1, 'experiment': cfg['experiment_id'], 'config': cfg,
            'config_path': config_path.relative_to(root).as_posix(), 'config_sha256': sha(config_path),
            'prompts': prompts, 'vectors': vectors, 'generations': generations,
            'reference_generations': reference, 'steered_generations': steered, 'hook_checks': hook_checks,
            'grid_forwards': grid_forwards, 'test_forwards': test_forwards,
            'maximum_generated_tokens': cfg['total_generated_tokens'],
            'requires_inference': False, 'status': 'planned; no model loaded'}


def cell_key(row):
    return (row['layer'], row['alpha'])


def gap_closed(rows, baseline, prompted, text, vector):
    """Share of the prompted-vs-default likelihood gap recovered by an intervention.

    rows: steered teacher-forced rows for ONE cell and split. baseline/prompted map
    (prompt_id, text) -> {'nll_sum', 'tokens'} for the same split. Ratio of summed
    per-token-weighted NLL differences, so long replies do not dominate short ones
    more than their token count warrants. 1.0 equals the ordinary system prompt;
    0.0 equals no intervention; negative means the reply became less likely.
    """
    chosen = [r for r in rows if r['text'] == text and r['vector'] == vector]
    if not chosen or len({r['prompt_id'] for r in chosen}) != len(chosen):
        raise ValueError('Missing or duplicated rows for ' + text + '/' + vector)
    recovered = total = 0.
    improved = 0
    for r in chosen:
        key = (r['prompt_id'], text)
        base, best = baseline[key], prompted[key]
        if not base['tokens'] == best['tokens'] == r['tokens']:
            raise ValueError('Token counts differ across contexts for ' + r['prompt_id'])
        recovered += base['nll_sum'] - r['nll_sum']
        total += base['nll_sum'] - best['nll_sum']
        improved += r['nll_sum'] < base['nll_sum']
    if total <= 0:
        raise ValueError('The ordinary prompt does not make its own reply more likely; gap undefined')
    return {'gap_closed': recovered / total, 'prompts_improved': improved, 'prompts': len(chosen),
            'baseline_nll_per_token': sum(baseline[(r['prompt_id'], text)]['nll_sum'] for r in chosen) / sum(r['tokens'] for r in chosen),
            'prompted_nll_per_token': sum(prompted[(r['prompt_id'], text)]['nll_sum'] for r in chosen) / sum(r['tokens'] for r in chosen),
            'steered_nll_per_token': sum(r['nll_sum'] for r in chosen) / sum(r['tokens'] for r in chosen)}


def grid_table(rows, baseline, prompted, vectors):
    """Per cell: gap closed for every (target text, vector) pair on persona texts."""
    cells = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    table = []
    for (layer, alpha), members in sorted(cells.items()):
        entry = {'layer': layer, 'alpha': alpha, 'gap': {}}
        for text in PERSONAS:
            entry['gap'][text] = {v: gap_closed(members, baseline, prompted, text, v) for v in vectors}
        table.append(entry)
    return table


def select_cell(table, persona):
    """Frozen rule: largest own-vector gap closed on the select split; ties to the weaker, earlier cell."""
    return min(table, key=lambda e: (-e['gap'][persona][persona]['gap_closed'], e['alpha'], e['layer']))


def controls(entry, persona):
    return {v: g['gap_closed'] for v, g in entry['gap'][persona].items() if v != persona}


def teacher_forced_verdict(entry, persona, cfg):
    own = entry['gap'][persona][persona]
    others = controls(entry, persona)
    strongest = max(others, key=others.get)
    rule = cfg['criteria']
    result = {'own_gap_closed': own['gap_closed'], 'strongest_control': strongest,
              'strongest_control_gap_closed': others[strongest],
              'margin_over_strongest_control': own['gap_closed'] - others[strongest],
              'prompts_improved': own['prompts_improved'], 'prompts': own['prompts'],
              'C1_transfer': own['gap_closed'] >= rule['min_gap_closed'],
              'C2_specific': own['gap_closed'] - others[strongest] >= rule['min_margin_over_controls'],
              'C3_consistent': own['prompts_improved'] >= rule['min_prompts_improved']}
    return result


def generation_verdict(rows, persona, cfg):
    """rows: scored test/check generations for one persona's selected cell plus references."""
    by = {}
    for r in rows:
        by.setdefault(r['condition'], {})[r['prompt_id']] = r
    style = lambda condition: {k: v['style_llr_per_token'][persona] for k, v in by[condition].items() if v['split'] == 'test'}
    base, steered, random_ = style('reference-neutral'), style('steer-' + persona), style('steer-' + persona + '-random')
    prompted = style('reference-' + persona)
    if not set(base) == set(steered) == set(random_) == set(prompted) or len(base) != SPLITS['test']:
        raise ValueError('Incomplete generation cohort for ' + persona)
    mean = lambda values: sum(values.values()) / len(values)
    above = sum(steered[k] > base[k] for k in base)
    lost = sorted(k for k, v in by['reference-neutral'].items() if v['split'] == 'check'
                  and v['task_score']['answer_match'] is True and by['steer-' + persona][k]['task_score']['answer_match'] is not True)
    return {'style_llr_mean': {'default': mean(base), 'prompted': mean(prompted), 'steered': mean(steered), 'random_matched': mean(random_)},
            'steered_above_default_prompts': above, 'prompts': len(base), 'lost_baseline_passed_checks': lost,
            'C4_style': above >= cfg['criteria']['min_prompts_improved'] and mean(steered) > mean(random_),
            'C5_no_lost_check': not lost}


def nominate(forced, generated):
    keys = ['C1_transfer', 'C2_specific', 'C3_consistent']
    return all(forced[k] for k in keys) and generated['C4_style'] and generated['C5_no_lost_check']
