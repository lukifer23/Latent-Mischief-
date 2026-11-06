"""Recompute transparent per-condition metrics from a local run. No quality grading."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from storage import write_json

ROOT = Path(__file__).resolve().parents[1]
GROUP_FIELDS = ('condition', 'persona', 'layer', 'dose', 'candidate', 'gain', 'offset', 'sign', 'target_fraction', 'cohort', 'role')


def repeated_fourgrams(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='Local run directory name, not a path')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if Path(args.run).name != args.run or '/' in args.run or '\\' in args.run:
        raise ValueError('Supply a run directory name')
    folder = (ROOT / 'runs' / args.run).resolve()
    if folder.parent != (ROOT / 'runs').resolve():
        raise ValueError('Run outside runs directory')
    records = json.loads((folder / 'outputs.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for record in records:
        if record.get('phase') != 'probe':
            continue
        key = tuple(record.get(field) for field in GROUP_FIELDS)
        groups[key].append(record)
    result = []
    for key, items in groups.items():
        row = {**dict(zip(GROUP_FIELDS, key)),
                       'n': len(items), 'eos': sum(x['terminated_with_eos'] for x in items),
                       'mean_tokens': sum(x['output_tokens'] for x in items) / len(items),
                       'mean_repeated_fourgram_fraction': sum(repeated_fourgrams(x['token_ids']) for x in items) / len(items),
                       'mean_seconds': sum(x['seconds'] for x in items) / len(items),
                       'hamster_keyword_replies': sum(bool(x['indicators']['hamster_terms_present']) for x in items) if all('indicators' in x for x in items) else None}
        for metric in ('reasoning_boundary_present', 'self_check_phrase_present', 'exact_json_match', 'exact_text_match'):
            measured = [x['measurements'][metric] for x in items if metric in x.get('measurements', {})]
            row[metric] = {'positive': sum(measured), 'measured': len(measured)}
        result.append(row)
    write_json(folder / 'analysis.json', {'groups': result, 'note': 'Exploratory indicators; no persona quality or accuracy scoring.'})
    print(str(folder / 'analysis.json') if args.quiet else json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
