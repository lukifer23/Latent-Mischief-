"""Fresh prompt replication using public, frozen calibration evidence and matched random controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import lab
from calibrated_map import capture_prefill
from calibration import offset_hook
from measurements import measure


def select_units(cfg, rows):
    chooser = random.Random(cfg['seed'])
    selected = []
    for target in cfg['targets']:
        eligible = []
        for neuron in sorted({r['neuron'] for r in rows if r['layer']==target['layer']}):
            pair = [r for r in rows if r['layer']==target['layer'] and r['neuron']==neuron and r['sign']==1 and r['target_fraction'] in cfg['strengths']]
            if len(pair)==len(cfg['strengths']) and all(not r['hit_cap'] and abs(r['mean_actual_fraction']/r['target_fraction']-1)<=.25 and r['mean_kl']<=.5 for r in pair):
                if neuron != target['neuron']: eligible.append(neuron)
        controls = chooser.sample(eligible,cfg['random_controls_per_layer'])
        for neuron in [target['neuron'],*controls]:
            for strength in cfg['strengths']:
                row = next(r for r in rows if r['layer']==target['layer'] and r['neuron']==neuron and r['sign']==1 and r['target_fraction']==strength)
                selected.append({**row,'role':'target' if neuron==target['neuron'] else 'same-layer-random'})
    return selected


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=lab.ROOT/'configs/offset-replication.json')
    args=parser.parse_args(); cfg_bytes=args.config.read_bytes(); cfg=json.loads(cfg_bytes)
    evidence=json.loads((lab.ROOT/cfg['source_evidence']).read_text(encoding='utf-8'))
    # The public CAL-001 export was measured on CUDA/BF16, before the adapter.
    from runtime import require_cuda_protocol
    require_cuda_protocol('Frozen CAL-001 offsets')
    if evidence['model']!=lab.config() or evidence['runtime_lock_sha256']!=lab.environment()['lock_sha256']:
        raise ValueError('Model/runtime differs from frozen calibration evidence')
    selected=select_units(cfg,evidence['screening'])
    prompts=json.loads((lab.ROOT/cfg['prompt_file']).read_text(encoding='utf-8'))
    folder=lab.ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-MAP-006-offset-replication')
    folder.mkdir()
    hashes=lab.archive_sources(folder,['scripts/replicate_offsets.py','scripts/calibrated_map.py','scripts/calibration.py',
        'scripts/lab.py','scripts/storage.py','scripts/measurements.py','uv.lock','configs/model.json',cfg['prompt_file'],cfg['source_evidence']])
    (folder/'source/experiment-config.json').write_bytes(cfg_bytes)
    lab.write_json(folder/'metadata.json',{'experiment':'MAP-006','model':lab.config(),'environment':lab.environment(),
        'experiment_config':cfg,'source_hashes':hashes,'experiment_config_sha256':hashlib.sha256(cfg_bytes).hexdigest(),
        'hook':'single-channel constant w2-input offsets, all prompt/decode positions',
        'generation':{'do_sample':False,'max_new_tokens':cfg['max_new_tokens']},'limits':cfg['scope']})
    lab.write_json(folder/'selected.json',selected)
    print(f'RUN {folder}',flush=True)
    model,tokenizer=lab.load(); versions={name:p._version for name,p in model.named_parameters()}
    layers=sorted({r['layer'] for r in selected}); special=set(tokenizer.all_special_ids)
    outputs,baseline,prefill=[],{},{}
    def generate(item): return lab.generate(model,tokenizer,item['prompt'],system=cfg['system'],max_new_tokens=cfg['max_new_tokens'])
    def emit(result,item,phase='probe',**labels):
        result.update(phase=phase,prompt_id=item['id'],cohort=item['cohort'],domain=item['domain'],measurements=measure(result,item),**labels)
        outputs.append(result); lab.write_json(folder/'outputs.json',outputs)
        print(f'REPLY {len(outputs)}/158 {item["id"]}',flush=True)
    for item in prompts:
        with capture_prefill(model,layers) as captured: result=generate(item)
        baseline[item['id']],prefill[item['id']]=result['token_ids'][:],captured
        emit(result,item,condition='baseline')
    first=selected[0]
    with offset_hook(model,first['layer'],first['neuron'],0): zero=generate(prompts[0])
    restored=generate(prompts[0])
    checks={'zero_offset_token_identical':zero['token_ids']==baseline[prompts[0]['id']],
        'removed_hook_token_identical':restored['token_ids']==baseline[prompts[0]['id']]}
    emit(zero,prompts[0],phase='check',condition='zero-offset'); emit(restored,prompts[0],phase='check',condition='removed-hook')
    lab.write_json(folder/'checks.json',checks)
    if not all(checks.values()): raise RuntimeError('Hook check failed')
    schedule=[(u,p) for u in selected for p in prompts]; random.Random(cfg['seed']+1).shuffle(schedule)
    lab.write_json(folder/'trial-order.json',[{'condition':'channel-offset','candidate':u['candidate'],'offset':u['offset'],'prompt_id':p['id']} for u,p in schedule])
    for unit,item in schedule:
        with capture_prefill(model,[unit['layer']]) as captured:
            with offset_hook(model,unit['layer'],unit['neuron'],unit['offset']): result=generate(item)
        mask=torch.tensor([i>0 and token not in special for i,token in enumerate(result['input_token_ids'])])
        ref=prefill[item['id']][unit['layer']][mask]
        actual=((captured[unit['layer']][mask]-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).mean().item()
        emit(result,item,condition='channel-offset',**{k:unit[k] for k in ['candidate','layer','neuron','offset','sign','target_fraction','offset_in_rms_units','role']},
            actual_prefill_fraction=actual,token_identical_to_baseline=result['token_ids']==baseline[item['id']])
    checks['parameter_version_counters_unchanged']=versions=={name:p._version for name,p in model.named_parameters()}
    lab.write_json(folder/'checks.json',checks)
    lab.write_json(folder/'summary.json',{'completed':all(checks.values()),'checks':checks,'generations':len(outputs),'intervention_replies':len(schedule)})
    if not all(checks.values()): raise RuntimeError('Parameter versions changed')
    print(f'COMPLETE {folder}',flush=True)


if __name__=='__main__': main()
