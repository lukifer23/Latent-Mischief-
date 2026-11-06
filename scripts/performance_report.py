"""Export verified completed timing points and explicit interrupted sizes; never infer missing timings."""
import argparse
import hashlib
import json
from pathlib import Path

from storage import write_json

ROOT=Path(__file__).resolve().parents[1]


def collect(name):
    folder=(ROOT/'runs'/name).resolve()
    if folder.parent!=(ROOT/'runs').resolve(): raise ValueError('Use a run directory name')
    read=lambda file:json.loads((folder/file).read_text(encoding='utf-8'))
    meta=read('metadata.json'); rows=read('outputs.json'); identity=read('identity.json')
    for name,digest in meta['source_hashes'].items():
        source=(folder/'source'/name).resolve()
        if not source.is_relative_to(folder/'source') or hashlib.sha256(source.read_bytes()).hexdigest()!=digest: raise ValueError('Archived source mismatch')
    config=folder/'source/experiment-config.json'
    if hashlib.sha256(config.read_bytes()).hexdigest()!=meta['experiment_config_sha256']: raise ValueError('Archived experiment config mismatch')
    completed=[x for x in rows if x['status']=='complete']
    assert len({(x['condition'],x['actual_input_tokens']) for x in completed})==len(completed)
    for row in completed:
        if row['actual_input_tokens']!=row['requested_input_tokens']: raise ValueError('Context length mismatch')
        expected=row['decode_tokens_after_first']/row['decode_seconds_after_first'] if row['decode_tokens_after_first'] else None
        if expected is not None and abs(expected-row['decode_tokens_per_second'])>1e-9: raise ValueError('Decode denominator mismatch')
    return {'schema':1,'run':folder.name,'model':meta['model'],'environment':meta['environment'],
        'runtime_attention':meta['runtime_attention'],'config':meta['experiment_config'],'source_hashes':meta['source_hashes'],
        'outputs_sha256':hashlib.sha256((folder/'outputs.json').read_bytes()).hexdigest(),
        'identity_checks':identity['checks'],'weight_sha256':identity.get('weights',{}),
        'preflight':read('preflight.json'),'checks':read('checks.json'),
        'final_parameter_check_available':(folder/'summary.json').exists(),
        'interruption':read('interruption.json') if (folder/'interruption.json').exists() else None,
        'rows':[{k:v for k,v in x.items() if k not in ['text','token_ids']} for x in rows],
        'limits':'One timing per point; synthetic document workload; no long-context reasoning benchmark. Reserved memory differs from allocated; RSS snapshots are not peak. Failed 128K point has no completed timing.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run');args=parser.parse_args()
    data=collect(args.run);dest=ROOT/'experiments/evidence/PERF-001/metrics.json';write_json(dest,data)
    print(json.dumps({'destination':str(dest),'points':len(data['rows']),'maximum_completed_input_tokens':max(x['actual_input_tokens'] for x in data['rows'] if x['status']=='complete')}))


if __name__=='__main__':main()
