"""Stable exact-recipe identities, scoped baseline comparisons, and persona documentation."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from measurements import repeated_fourgrams
from storage import write_json

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={'channel-gain','channel-offset','sparse-profile','joint-profile','contrast-direction','random-direction','prompt-persona'}
REPORTS={'CHAR-001':'CHAR-001-sparse-characters.md','CHAR-002':'CHAR-002-joint-characters.md',
    'MAP-006':'MAP-006-offset-replication.md','MAP-005':'MAP-005-calibrated-effects.md',
    'MAP-004':'MAP-004-expanded-sample.md','MAP-003':'MAP-003-completion-replication.md',
    'MAP-001':'MAP-001-effect-first-pilot.md'}


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def recipe_id(recipe): return 'LM-'+hashlib.sha256(canonical(recipe).encode()).hexdigest()[:16]
def read(path,default=None): return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_public(data):
    """A fresh clone has reviewed recipes but no private historical raw runs."""
    public=read(ROOT/'experiments/personas/registry.json',{})
    entries={entry['id']:entry for entry in public.get('recipes',[])}
    for entry in data['recipes']:
        previous=entries.get(entry['id'])
        if previous:
            if previous['recipe']!=entry['recipe']:raise ValueError('Recipe ID collision')
            methods=previous['methods']+[m for m in entry['methods'] if m not in previous['methods']]
            replaced={(o['run'],o['cohort']) for o in entry['observations']}
            observations=[o for o in previous['observations'] if (o['run'],o['cohort']) not in replaced]+entry['observations']
            entry={**entry,'methods':methods,'observations':observations}
        entries[entry['id']]=entry
    return {**data,'recipes':sorted(entries.values(),key=lambda x:x['id']),
            'runs':{**public.get('runs',{}),**data['runs']},
            'run_refs':{**public.get('run_refs',{}),**data['run_refs']}}


def implementation(folder,file,function):
    tree=ast.parse((folder/'source'/file).read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==function)
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def build():
    public=read(ROOT/'experiments/personas/registry.json',{})
    entries={entry['id']:{**entry,'methods':list(entry['methods']),'observations':[]} for entry in public.get('recipes',[])}
    run_refs,runs,skipped={}, {}, []
    names=read(ROOT/'experiments/personas/names.json',{})
    models=read(ROOT/'configs/models.json',{})
    if models.get('qwen08',{}).get('published_edit'):
        qwen=models['qwen08']; variant=qwen['published_edit']
        recipe={'schema':1,'model':{'model_id':qwen['model_id'],'revision':qwen['revision'],'dtype':qwen['dtype'],'attention':qwen['live_attention']},
            'family':'published-checkpoint','operations':[],'positions':'checkpoint variant','hook_location':None,
            'variant':variant}
        identifier=recipe_id(recipe)
        entries[identifier]={'id':identifier,'codename':'Mischief-'+identifier[3:9].upper(),'method_label':'Qwen3.5-0.8B · published anti-loop weight edit',
            'observed_name':'Loop Brake (published)','name_status':'reference-only','naming_notes':'Published weight-space edit, not a local benchmark reproduction. Only Qwen3.5-0.8B; not the 4B Evil Qwen neuron recipe.',
            'recipe':recipe,'methods':[{'source':variant['source'],'selection':'Published checkpoint; two edited down_proj tensors are digest-verified by prepare_models.py.'}],'observations':[]}
    for folder in sorted((ROOT/'runs').iterdir()):
        if not folder.is_dir(): continue
        meta=read(folder/'metadata.json',{}); summary=read(folder/'summary.json',{})
        if summary.get('completed') is not True or not (folder/'outputs.json').exists(): continue
        outputs=read(folder/'outputs.json',[])
        probes=[(i,x) for i,x in enumerate(outputs) if x.get('phase','probe')=='probe']
        if not any(x.get('condition') in FAMILIES for _,x in probes): continue
        sources=meta.get('source_hashes',{})
        for name,digest in sources.items():
            path=(folder/'source'/name).resolve()
            if not path.is_relative_to((folder/'source').resolve()) or sha(path)!=digest: raise ValueError(f'Source integrity failure: {folder.name} {name}')
        model=meta.get('model',meta.get('config',{}))
        identity={key:model.get(key) for key in ['model_id','revision','dtype','attention']}
        identity['dtype']=meta.get('environment',{}).get('runtime',{}).get('dtype',identity['dtype'])
        if not identity['model_id'] or not identity['revision']: raise ValueError('Model identity missing')
        experiment=meta.get('experiment') or ('LFM-002' if 'persona' in folder.name else 'unknown')
        runs[folder.name]={'experiment':experiment,'model':model,'config':meta.get('experiment_config',{}),
            'generation':meta.get('generation',{}),'environment':meta.get('environment',{}),'source_hashes':sources,
            'outputs_sha256':sha(folder/'outputs.json'),'report':REPORTS.get(experiment),'checks':read(folder/'checks.json',{})}
        baselines={x.get('prompt_id',x.get('id')):x for _,x in probes if x.get('condition')=='baseline'}
        profiles={x['profile_id']:x for x in read(folder/'profiles.json',[])}
        parent_profiles=read(folder/'source/parent/profiles.json',[])
        direction_data=None; random_data=None
        grouped=defaultdict(list); run_refs[folder.name]={}; implementation_cache={}
        for index,x in probes:
            condition=x.get('condition')
            if condition not in FAMILIES: continue
            recipe={'schema':1,'model':identity,'family':condition,'operations':[],
                'positions':'all prompt positions and every decode position','hook_location':'gated MLP input before w2 and post_ffn_norm'}
            method={'intended_character':x.get('persona'),'selection':'See frozen source configuration and extraction artifacts.'}
            file,function='scripts/calibration.py','offset_hook'
            if condition=='channel-gain':
                file,function='scripts/lab.py','gain_hook'
                recipe['operations']=[{'layer':x['layer'],'kind':'gain','channels':[x['neuron']],'gain':x['gain']}]
                label=f"L{x['layer']:02d} N{x['neuron']:04d} · gain {x['gain']}"
            elif condition=='channel-offset':
                recipe['operations']=[{'layer':x['layer'],'kind':'offset','channels':[x['neuron']],'offsets':[x['offset']]}]
                label=f"{x['candidate']} · {'+' if x['offset']>=0 else ''}{x['offset']:.6g} offset"
                method.update(calibration_target=x.get('target_fraction'),native_rms_units=x.get('offset_in_rms_units'))
            elif condition in ['sparse-profile','joint-profile']:
                file,function='scripts/character_lab.py','sparse_hook'
                if condition=='sparse-profile':
                    parts=[profiles[x['profile_id']]]; multiplier=1; label=x['profile_id']
                else:
                    cfg=meta['experiment_config']; role='sparse-random' if x['role']=='sparse-random' else 'style-profile'
                    parts=[p for p in parent_profiles if p['role']==role and p['target_fraction']==cfg['source_target'] and
                           (role=='sparse-random' or p['persona']==x['persona']) and p['layer'] in cfg['layers']]
                    multiplier=x['dose']; label=x['profile_id']
                if not parts: raise ValueError('Exact sparse recipe missing')
                recipe['operations']=[{'layer':p['layer'],'kind':'offset','channels':p['channels'],
                    'offsets':[value*multiplier for value in p['offsets']]} for p in sorted(parts,key=lambda p:p['layer'])]
                method.update(profile_id=x['profile_id'],nominal_target=x.get('target_fraction'),amplitude_multiplier=multiplier)
            elif condition in ['contrast-direction','random-direction']:
                import torch
                file,function='scripts/personas.py','direction_hook'
                if direction_data is None: direction_data=torch.load(folder/'directions.pt',map_location='cpu',weights_only=True)
                if condition=='contrast-direction':
                    vector=direction_data['directions'][x['persona']][x['layer']]; artifact='directions.pt'; key=f"{x['persona']}:{x['layer']}"
                else:
                    if random_data is None: random_data=torch.load(folder/'random-vectors.pt',map_location='cpu',weights_only=True)
                    key=f"{x['layer']}:{x['control']}"; vector=random_data[key]; artifact='random-vectors.pt'
                amplitude=x['dose']*direction_data['neutral_norms'][x['layer']]
                recipe.update(positions='last position of each forward: final prompt token and incremental decode tokens',hook_location='decoder output after residual addition')
                recipe['operations']=[{'layer':x['layer'],'kind':'residual-vector','vector_sha256':hashlib.sha256(vector.float().contiguous().numpy().tobytes()).hexdigest(),
                    'amplitude':amplitude,'dtype':'float32 reference; cast delta to model dtype'}]
                method.update(vector_artifact={'run':folder.name,'file':artifact,'key':key,'file_sha256_at_catalog_build':sha(folder/artifact)},dose=x['dose'])
                label=f"{x.get('persona','random')} · layer {x['layer']} · residual dose {x['dose']}"
            else:
                recipe.update(positions='system instruction',hook_location=None,system=x['system'])
                label=f"Prompted {x.get('persona','character')}"; file=function=None
            if file:
                if (file,function) not in implementation_cache: implementation_cache[file,function]=implementation(folder,file,function)
                recipe['hook_function_ast_sha256']=implementation_cache[file,function]
            identifier=recipe_id(recipe); run_refs[folder.name][str(index)]=identifier
            if identifier not in entries:
                annotation=names.get(identifier,{})
                entries[identifier]={'id':identifier,'codename':'Mischief-'+identifier[3:9].upper(),'method_label':label,
                    'observed_name':annotation.get('observed_name',''),'name_status':annotation.get('name_status','unreviewed'),
                    'naming_notes':annotation.get('naming_notes',''),'recipe':recipe,'methods':[],'observations':[]}
            entry=entries[identifier]
            origin={'run':folder.name,**method}
            if origin not in entry['methods']: entry['methods'].append(origin)
            grouped[identifier,x.get('cohort','legacy-exploration')].append((index,x))
        for (identifier,cohort),items in grouped.items():
            paired=[(i,x,baselines.get(x.get('prompt_id',x.get('id')))) for i,x in items]
            if any(b is None for _,_,b in paired): raise ValueError('Missing matched default')
            checks=[(x['measurements'][key],b.get('measurements',{}).get(key)) for _,x,b in paired
                    for key in ['exact_json_match','exact_text_match'] if key in x.get('measurements',{})]
            observation={'run':folder.name,'cohort':cohort,'n':len(paired),'output_indices':[i for i,_,_ in paired],
                'changed_sequences':sum(x['token_ids']!=b['token_ids'] for _,x,b in paired),
                'mean_tokens':statistics.mean(x['output_tokens'] for _,x,_ in paired),
                'default_mean_tokens':statistics.mean(b['output_tokens'] for _,_,b in paired),
                'eos':sum(x['terminated_with_eos'] for _,x,_ in paired),'default_eos':sum(b['terminated_with_eos'] for _,_,b in paired),
                'mean_repetition':statistics.mean(repeated_fourgrams(x['token_ids']) for _,x,_ in paired),
                'default_mean_repetition':statistics.mean(repeated_fourgrams(b['token_ids']) for _,_,b in paired),
                'exact_pass':sum(v for v,_ in checks) if checks else None,'exact_n':len(checks),
                'default_exact_pass':sum(v for _,v in checks) if checks and all(v is not None for _,v in checks) else None,
                'system':paired[0][1].get('system'),'default_system':paired[0][2].get('system'),
                'prompt_sha256':hashlib.sha256(canonical([x['prompt'] for _,x,_ in paired]).encode()).hexdigest(),
                'generation':{'do_sample':False,'seed':model.get('seed'),'max_new_tokens':meta.get('generation',{}).get('max_new_tokens',meta.get('experiment_config',{}).get('max_new_tokens'))}}
            entries[identifier]['observations'].append(observation)
    for folder in sorted((ROOT/'runs').glob('*-LIVE-*')):
        if read(folder/'summary.json',{}).get('completed') is not True: continue
        meta=read(folder/'metadata.json',{}); identifier=meta.get('recipe_id')
        if identifier not in entries: continue
        outputs=read(folder/'outputs.json',[])
        if len(outputs)!=2: continue
        for name,digest in meta.get('source_hashes',{}).items():
            path=(folder/'source'/name).resolve()
            if not path.is_relative_to((folder/'source').resolve()) or sha(path)!=digest: raise ValueError(f'Live source integrity failure: {folder.name} {name}')
        b,x=outputs;request=meta['request']
        runs[folder.name]={'experiment':'LIVE','model':meta['model'],'config':request,'generation':{'do_sample':False,'max_new_tokens':request['max_new_tokens']},'environment':meta['environment'],'source_hashes':meta['source_hashes'],'outputs_sha256':sha(folder/'outputs.json'),'report':None,'checks':{'parameter_version_checks':True}}
        run_refs[folder.name]={'1':identifier}
        entries[identifier]['observations'].append({'run':folder.name,'cohort':'live-chat','n':1,'output_indices':[1],
            'changed_sequences':int(x['token_ids']!=b['token_ids']),'mean_tokens':x['output_tokens'],'default_mean_tokens':b['output_tokens'],
            'eos':int(x['terminated_with_eos']),'default_eos':int(b['terminated_with_eos']),
            'mean_repetition':repeated_fourgrams(x['token_ids']),'default_mean_repetition':repeated_fourgrams(b['token_ids']),
            'exact_pass':None,'exact_n':0,'default_exact_pass':None,'system':x['system'],'default_system':b['system'],
            'prompt_sha256':hashlib.sha256(x['prompt'].encode()).hexdigest(),'generation':{'do_sample':False,'seed':request['seed'],'max_new_tokens':request['max_new_tokens']},
            'performance':{name:{k:out.get(k) for k in ['input_tokens','time_to_first_token_seconds','decode_tokens_per_second','peak_gpu_allocated_bytes','load_seconds']} for name,out in [('default',b),('edited',x)]}})
    return merge_public({'schema':1,'recipes':sorted(entries.values(),key=lambda x:x['id']),'runs':runs,'run_refs':run_refs,'skipped':skipped,
        'naming_rule':'Codename is neutral and deterministic. Observed nickname is a scoped human annotation, not automatic persona validation.',
        'limits':'Do not pool different prompt cohorts, instructions, runtimes or token caps. Historical residual vectors require local artifacts; hashes identify exact vectors.'})


def publish(data):
    # Interactive conversations stay local, including private system prompts/history.
    import copy
    data=copy.deepcopy(data)
    private_runs={name for name,run in data['runs'].items() if run['experiment']=='LIVE'}
    data['runs']={name:run for name,run in data['runs'].items() if name not in private_runs}
    data['run_refs']={name:refs for name,refs in data['run_refs'].items() if name not in private_runs}
    for entry in data['recipes']:
        entry['observations']=[o for o in entry['observations'] if o['run'] not in private_runs]
    directory=ROOT/'experiments/personas'; directory.mkdir(exist_ok=True)
    write_json(directory/'registry.json',data)
    cards=directory/'cards'; cards.mkdir(exist_ok=True)
    index=['# Persona and recipe library','','Every exact intervention has a stable ID. Nicknames describe reviewed observations; intended styles are separate. See [naming methodology](../../docs/PERSONAS.md).','','| Persona / codename | Method | Review status | Card |','|---|---|---|---|']
    for entry in data['recipes']:
        title=entry['observed_name'] or entry['codename']
        lines=[f'# {title}','',f"Recipe: `{entry['id']}` · {entry['name_status']}",'',entry['method_label'],'',entry['naming_notes'] or 'No observed persona nickname has been assigned.','',
            '## Exact intervention','','```json',json.dumps(entry['recipe'],indent=2,ensure_ascii=False),'```','',
            '## Differences from matched default','','Each row is a separate run/cohort; numbers are descriptive, not personality scores.','','| Run / cohort | n | Changed | Mean tokens: default → edit | EOS: default → edit | Exact checks: default → edit |','|---|---:|---:|---|---|---|']
        for o in entry['observations']:
            exact='not measured' if not o['exact_n'] else f"{o['default_exact_pass']} → {o['exact_pass']} / {o['exact_n']}"
            lines.append(f"| `{o['run']}` / {o['cohort']} | {o['n']} | {o['changed_sequences']} | {o['default_mean_tokens']:.2f} → {o['mean_tokens']:.2f} | {o['default_eos']} → {o['eos']} | {exact} |")
        lines += ['','## Methodology and evidence','','Method derivation, intended character, source runs and local vector references:','','```json',json.dumps(entry['methods'],indent=2,ensure_ascii=False),'```','',
            'Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.']
        (cards/(entry['id']+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        index.append(f"| {title} | {entry['method_label']} | {entry['name_status']} | [Card](cards/{entry['id']}.md) |")
    (directory/'README.md').write_text('\n'.join(index)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--publish',action='store_true'); args=parser.parse_args()
    data=build(); write_json(ROOT/'runs/persona-catalog.json',data)
    if args.publish: publish(data)
    print(json.dumps({'recipes':len(data['recipes']),'runs':len(data['runs']),'observations':sum(len(x['observations']) for x in data['recipes']),'published':args.publish}))


if __name__=='__main__': main()
