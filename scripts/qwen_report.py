"""Independently verify, export and plot paired Qwen pilot outcomes, including missing pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import lab
from math_grade import grade
from measurements import repeated_fourgrams
from qwen_eval import validate_prefix

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(name):
    folder=(lab.ROOT/'runs'/name).resolve()
    if folder.parent!=(lab.ROOT/'runs').resolve():raise ValueError('Use a run name under runs/')
    meta=read(folder/'metadata.json');outputs=read(folder/'outputs.json');cfg=meta['experiment_config']
    for file,digest in meta['source_hashes'].items():
        path=(folder/'source'/file).resolve()
        if not path.is_relative_to(folder/'source') or sha(path)!=digest:raise ValueError('Archived source mismatch: '+file)
    if sha(lab.ROOT/'scripts/math_grade.py')!=meta['source_hashes']['scripts/math_grade.py']:raise ValueError('Current grader differs from archived grader')
    if sha(folder/'source'/cfg['manifest'])!=cfg['manifest_sha256']:raise ValueError('Problem manifest mismatch')
    if sha(folder/'audit.json')!=cfg['audit_sha256']:raise ValueError('Model audit mismatch')
    manifest=read(folder/'source'/cfg['manifest']);validate_prefix(outputs,cfg['schedule'])
    rows=[]
    for output,trial in zip(outputs,cfg['schedule']):
        problem=next(x for x in manifest['problems'] if x['unique_id']==trial['prompt_id'])
        if output['prompt']!=problem['problem']+cfg['prompt_suffix']:raise ValueError('Prompt differs from frozen manifest')
        if output['output_tokens']!=len(output['token_ids']) or output['output_tokens']>cfg['max_new_tokens']:raise ValueError('Token count mismatch')
        if output['input_tokens']!=len(output['input_token_ids']):raise ValueError('Input count mismatch')
        rescored=grade(output['text'],problem['answer'])
        if rescored!=output['grading']:raise ValueError('Independent regrading differs')
        repeated=repeated_fourgrams(output['token_ids'])
        if repeated!=output['measurements']['repeated_fourgram_fraction']:raise ValueError('Repetition calculation differs')
        if not output['parameter_version_unchanged']:raise ValueError('Parameter version check failed')
        if output['decode_tokens_after_first']!=max(output['output_tokens']-1,0):raise ValueError('Decode denominator token count differs')
        rate=output['decode_tokens_after_first']/output['decode_seconds_after_first'] if output['decode_tokens_after_first'] else None
        if rate is not None and abs(rate-output['decode_tokens_per_second'])>1e-9:raise ValueError('Throughput calculation differs')
        fields=['condition','prompt_id','seed','model_id','revision','input_tokens','output_tokens','termination','terminated_with_eos',
            'hit_token_limit','seconds','prefill_seconds','time_to_first_token_seconds','decode_tokens_per_second','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','load_seconds']
        rows.append({**{k:output[k] for k in fields},'level':problem['level'],'correct':rescored['correct'],'grading':rescored,
            'repeated_fourgram_fraction':repeated,'thinking_closed':'</think>' in output['text'],
            'text_sha256':hashlib.sha256(output['text'].encode()).hexdigest()})
    pairs=[];missing=[]
    for problem in manifest['problems']:
        actual=[x for x in outputs if x['prompt_id']==problem['unique_id']]
        if len(actual)!=2 or any(x['termination']=='wall-limit' for x in actual):missing.append(problem['unique_id']);continue
        base=next(x for x in actual if x['condition']=='baseline');edited=next(x for x in actual if x['condition']=='published-checkpoint')
        if base['input_token_ids']!=edited['input_token_ids']:raise ValueError('Checkpoint tokenized inputs differ within pair')
        pairs.append({'prompt_id':problem['unique_id'],'level':problem['level'],
            'correctness':{'baseline':base['grading']['correct'],'edited':edited['grading']['correct']},
            'eos':{'baseline':base['terminated_with_eos'],'edited':edited['terminated_with_eos']},
            'tokens':{'baseline':base['output_tokens'],'edited':edited['output_tokens']}})
    summary=read(folder/'summary.json') if (folder/'summary.json').exists() else None
    if summary and (summary['generations']!=len(rows) or len(rows)!=len(cfg['schedule'])):raise ValueError('Completed schedule count mismatch')
    preflight=read(folder/'preflight.json') if (folder/'preflight.json').exists() else {}
    if cfg['experiment_id']=='QWN-003':
        preflight={condition:{k:v for k,v in item.items() if k not in ['samples','sample_messages','sample_token_ids','intermediate_prefill','sample_settings']} for condition,item in preflight.items()}
        if summary:summary={**summary,'preflight_checks':preflight}
    resources=[]
    if (folder/'resources.jsonl').exists():
        for line in (folder/'resources.jsonl').read_text().splitlines():
            row=json.loads(line)
            resources.append({k:v for k,v in row.items() if k not in ['thermal_report','monotonic_seconds']})
    interruption=read(folder/'interruption.json') if (folder/'interruption.json').exists() else None
    active=read(folder/'active.json') if interruption and interruption.get('active_trial') and (folder/'active.json').exists() else None
    partial={'trial':active['trial'],'output_tokens':len(active['token_ids']),'text_sha256':hashlib.sha256(active['text'].encode()).hexdigest()} if active else None
    return {'schema':1,'experiment':cfg['experiment_id'],'run':folder.name,'model':meta['model'],'config':cfg,'environment':meta['environment'],
        'reporter_sha256':sha(Path(__file__)),'math_verify_version':meta['math_verify_version'],
        'source_hashes':meta['source_hashes'],'outputs_sha256':sha(folder/'outputs.json'),'preflight':preflight,
        'summary':summary,'interruption':interruption,'partial':partial,'resources':resources,
        'cleanup':read(folder/'cleanup.json') if (folder/'cleanup.json').exists() else None,
        'rows':rows,'pairs':pairs,'missing_or_wall_interrupted_pairs':missing,
        'verification':{'source_files':len(meta['source_hashes']),'regraded_rows':len(rows),'matched_tokenized_pairs':len(pairs)},
        'limits':cfg.get('scope','Four stratified problems, one sample per checkpoint; 4096 cap/reference kernels differ from upstream. No benchmark percentage replication or speedup claim.')}


def plot(data,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    if not data['rows']:return
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'latent-mischief'})
    problems=list(dict.fromkeys(x['prompt_id'] for x in data['rows']));positions=np.arange(len(problems))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,key,title in zip(axes.flat,['correct','terminated_with_eos','output_tokens','repeated_fourgram_fraction'],
            ['Math-Verify correctness','Completion (EOS)','Output tokens','Repeated token 4-gram fraction']):
        for offset,condition,color in [(-.18,'baseline','#777777'),(.18,'published-checkpoint','#247c9e')]:
            lookup={x['prompt_id']:x for x in data['rows'] if x['condition']==condition}
            values=[lookup[p][key] if p in lookup else float('nan') for p in problems]
            ax.bar(positions+offset,values,width=.34,label=condition,color=color)
            for index,problem in enumerate(problems):
                if problem not in lookup:ax.text(positions[index]+offset,.06,'missing',rotation=90,fontsize=7,ha='center',transform=ax.get_xaxis_transform())
        ax.set(title=title,xticks=positions,xticklabels=['L'+str(next(x['level'] for x in data['rows'] if x['prompt_id']==p))+' / '+p.split('/')[-1].replace('.json','') for p in problems])
        if key in ['correct','terminated_with_eos']:ax.set(ylim=(0,1.2),yticks=[0,1])
        if key=='output_tokens':ax.axhline(data['config']['max_new_tokens'],linestyle=':',color='#b74c33',label='output cap')
        ax.legend(fontsize=8)
    state='Completed' if data['summary'] else 'Interrupted / partial'
    fig.suptitle(f"{data['experiment']} · {state} · {len(data['rows'])}/{len(data['config']['schedule'])} saved · {len(data['pairs'])} matched pairs")
    fig.supxlabel('A cap is not automatically a reasoning loop; EOS is not correctness. No benchmark or speedup claim.',fontsize=9)
    for extension in ['png','svg']:
        path=destination/f'comparisons.{extension}'
        fig.savefig(path,dpi=165,facecolor='white',metadata={'Date':None} if extension=='svg' else None)
        if extension=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run');args=parser.parse_args()
    data=collect(args.run);destination=lab.ROOT/'experiments/evidence'/data['experiment'];destination.mkdir(exist_ok=True)
    lab.write_json(destination/'metrics.json',data);plot(data,destination)
    print(json.dumps({'verification':data['verification'],'missing_pairs':data['missing_or_wall_interrupted_pairs'],
        'rows':[{k:r[k] for k in ['condition','level','output_tokens','correct','termination','repeated_fourgram_fraction']} for r in data['rows']]},indent=2))


if __name__=='__main__':main()
