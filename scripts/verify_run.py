"""Verify completed mapping evidence against its archived inputs and saved trial schedule."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    sources = metadata.get('source_hashes', metadata.get('hashes', {}))
    if not sources:
        raise ValueError('No archived source hashes')
    for name, digest in sources.items():
        path = (folder / 'source' / name).resolve()
        if not path.is_relative_to((folder / 'source').resolve()):
            raise ValueError('Source outside archive')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Archive hash mismatch: {name}')
    config_digest = metadata.get('experiment_config_sha256')
    if config_digest and hashlib.sha256((folder / 'source/experiment-config.json').read_bytes()).hexdigest() != config_digest:
        raise ValueError('Experiment config hash mismatch')
    checks = json.loads((folder / 'checks.json').read_text())
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError('Hook check failed or absent')
    summary = json.loads((folder / 'summary.json').read_text())
    if summary.get('completed') is not True:
        raise ValueError('Run is not complete')
    outputs = json.loads((folder / 'outputs.json').read_text(encoding='utf-8'))
    trials = json.loads((folder / 'trial-order.json').read_text())
    def identity(x):
        condition = x.get('condition', 'channel-gain')
        strength = 'offset' if condition == 'channel-offset' else 'gain'
        return condition, x['candidate'], x[strength], x['prompt_id']
    actual = [identity(x) for x in outputs if x['condition'] in ('channel-gain', 'channel-offset')]
    expected = [identity(x) for x in trials]
    if actual != expected or len(set(actual)) != len(actual) or len(actual) != summary['intervention_replies']:
        raise ValueError('Recorded trials differ from saved schedule or contain duplicates')
    return {'verified_source_files': len(sources), 'recorded_generations': len(outputs),
            'verified_intervention_trials': len(actual), 'hook_checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    folder = (ROOT / 'runs' / args.run).resolve()
    if folder.parent != (ROOT / 'runs').resolve():
        raise ValueError('Supply a run directory name under runs/')
    print(json.dumps(verify(folder), indent=2))


if __name__ == '__main__':
    main()
