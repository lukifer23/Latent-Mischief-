"""Combine frozen sparse profiles across layers; retain random controls and dose damage."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random

import lab
from calibrated_map import capture_prefill
from character_lab import sparse_hook
from measurements import measure
from personas import indicators


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=lab.ROOT/'configs/joint-characters.json')
    args=parser.parse_args(); cfg_bytes=args.config.read_bytes(); cfg=json.loads(cfg_bytes)
    parent=(lab.ROOT/'runs'/cfg['source_run']).resolve()
    if parent.parent!=(lab.ROOT/'runs').resolve(): raise ValueError('Invalid source run')
    old=json.loads((parent/'metadata.json').read_text(encoding='utf-8'))
    from runtime_identity import require_same_backend
    require_same_backend(old['environment'],lab.environment())
    summary=json.loads((parent/'summary.json').read_text())
    if summary.get('completed') is not True or not all(summary['checks'].values()): raise ValueError('Source run incomplete')
    if hashlib.sha256((parent/'profiles.json').read_bytes()).hexdigest()!=summary['profiles_sha256']: raise ValueError('Profiles changed')
    if old['model']!=lab.config() or old['environment']['lock_sha256']!=lab.environment()['lock_sha256']: raise ValueError('Runtime changed')
    profiles=json.loads((parent/'profiles.json').read_text())
    source_cfg=old['experiment_config']; prompts=json.loads((lab.ROOT/cfg['prompt_file']).read_text(encoding='utf-8'))
    groups={}
    for name in ['villain','hamster','random']:
        role='sparse-random' if name=='random' else 'style-profile'
        groups[name]=[p for p in profiles if p['role']==role and (name=='random' or p['persona']==name)
                      and p['layer'] in cfg['layers'] and p['target_fraction']==cfg['source_target']]
        if sorted(p['layer'] for p in groups[name])!=sorted(cfg['layers']): raise ValueError('Missing or duplicate source layer')
    folder=lab.ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-CHAR-002-joint-characters'); folder.mkdir()
    hashes=lab.archive_sources(folder,['scripts/joint_characters.py','scripts/character_lab.py','scripts/calibrated_map.py',
        'scripts/calibration.py','scripts/lab.py','scripts/measurements.py','scripts/storage.py','scripts/personas.py',
        'scripts/runtime_identity.py','uv.lock','configs/model.json',cfg['prompt_file']])
    for name in ['metadata.json','summary.json','profiles.json']:
        data=(parent/name).read_bytes(); path=folder/'source/parent'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        hashes['parent/'+name]=hashlib.sha256(data).hexdigest()
    (folder/'source/experiment-config.json').write_bytes(cfg_bytes)
    lab.write_json(folder/'metadata.json',{'experiment':'CHAR-002','model':lab.config(),'environment':lab.environment(),
        'experiment_config':cfg,'source_hashes':hashes,'experiment_config_sha256':hashlib.sha256(cfg_bytes).hexdigest(),
        'source_persona_config':source_cfg,'hook':'four simultaneous sparse w2-input offsets, all prompt/decode positions',
        'limits':cfg['scope'],'generation':{'do_sample':False,'max_new_tokens':cfg['max_new_tokens']}})
    print(f'RUN {folder}',flush=True)
    model,tokenizer=lab.load(); versions={n:p._version for n,p in model.named_parameters()}; special=set(tokenizer.all_special_ids)
    outputs,baseline,prefill=[],{},{}
    def context(name,multiplier):
        stack=ExitStack()
        try:
            for p in groups[name]: stack.enter_context(sparse_hook(model,p['layer'],p['channels'],[x*multiplier for x in p['offsets']]))
        except BaseException: stack.close(); raise
        return stack
    def generate(item,system=None): return lab.generate(model,tokenizer,item['prompt'],system=system or source_cfg['system'],max_new_tokens=cfg['max_new_tokens'])
    def emit(result,item,phase='probe',**labels):
        result.update(phase=phase,prompt_id=item['id'],cohort=item['cohort'],domain=item['domain'],**labels)
        result['measurements'],result['indicators']=measure(result,item),indicators(result)
        outputs.append(result); lab.write_json(folder/'outputs.json',outputs)
        print(f'REPLY {len(outputs)}/86 {item["id"]}',flush=True)
    for item in prompts:
        with capture_prefill(model,cfg['layers']) as captured: result=generate(item)
        baseline[item['id']],prefill[item['id']]=result['token_ids'][:],captured
        emit(result,item,condition='baseline')
    with context('villain',0): zero=generate(prompts[0])
    restored=generate(prompts[0])
    checks={'zero_offset_token_identical':zero['token_ids']==baseline[prompts[0]['id']],
            'removed_hook_token_identical':restored['token_ids']==baseline[prompts[0]['id']]}
    emit(zero,prompts[0],phase='check',condition='zero-offset'); emit(restored,prompts[0],phase='check',condition='removed-hook')
    lab.write_json(folder/'checks.json',checks)
    if not all(checks.values()): raise RuntimeError('Joint hook check failed')
    schedule=[{'condition':'joint-profile','persona':name if name!='random' else None,'candidate':'joint-'+name,
               'dose':dose,'prompt_id':p['id']} for name in groups for dose in cfg['multipliers'] for p in prompts]
    schedule += [{'condition':'prompt-persona','persona':name,'prompt_id':p['id']} for name in source_cfg['personas'] for p in prompts]
    random.Random(cfg['seed']).shuffle(schedule); lab.write_json(folder/'trial-order.json',schedule)
    lookup={p['id']:p for p in prompts}
    for trial in schedule:
        item=lookup[trial['prompt_id']]
        if trial['condition']=='prompt-persona':
            emit(generate(item,source_cfg['system']+' '+source_cfg['personas'][trial['persona']]),item,condition='prompt-persona',persona=trial['persona'],role='prompt-control')
            continue
        name=trial['persona'] or 'random'
        with capture_prefill(model,cfg['layers']) as captured:
            with context(name,trial['dose']): result=generate(item)
        mask=torch.tensor([i>0 and token not in special for i,token in enumerate(result['input_token_ids'])])
        actual={}
        for layer in cfg['layers']:
            ref=prefill[item['id']][layer][mask]
            actual[layer]=((captured[layer][mask]-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).mean().item()
        emit(result,item,condition='joint-profile',persona=trial['persona'],candidate=trial['candidate'],dose=trial['dose'],
            profile_id=trial['candidate']+'@'+str(trial['dose']),role='sparse-random' if name=='random' else 'joint-style',
            actual_prefill_by_layer=actual,token_identical_to_baseline=result['token_ids']==baseline[item['id']])
    checks['parameter_version_counters_unchanged']=versions=={n:p._version for n,p in model.named_parameters()}
    lab.write_json(folder/'checks.json',checks); lab.write_json(folder/'summary.json',{'completed':all(checks.values()),
        'checks':checks,'generations':len(outputs),'scheduled_trials':len(schedule),'note':cfg['scope']})
    if not all(checks.values()): raise RuntimeError('Parameter versions changed')
    print(f'COMPLETE {folder}',flush=True)


if __name__=='__main__': main()
