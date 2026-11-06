"""Verify and export numeric campaign comparisons; raw model replies remain local."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from measurements import measure, repeated_fourgrams
from storage import write_json
from verify_run import verify

ROOT=Path(__file__).resolve().parents[1]


def label(item):
    if item.get('profile_id'): return item['profile_id']
    if item['condition']=='channel-offset': return f"{item['candidate']}@{item['target_fraction']}"
    return item['condition']+((':'+item['persona']) if item.get('persona') else '')


def aggregate(rows):
    checks=[x['exact_check'] for x in rows if x['exact_check'] is not None]
    strength=[x['actual_prefill_fraction'] for x in rows if x['actual_prefill_fraction'] is not None]
    return {'n':len(rows),'eos':sum(x['eos'] for x in rows),'changed':sum(x['changed'] is True for x in rows),
        'exact_pass':sum(checks),'exact_n':len(checks),'mean_tokens':statistics.mean(x['tokens'] for x in rows),
        'mean_repetition':statistics.mean(x['repetition'] for x in rows),
        'mean_actual_prefill_fraction':statistics.mean(strength) if strength else None,
        'mean_seconds':statistics.mean(x['seconds'] for x in rows),
        'think_boundaries':sum(x['think_boundary'] for x in rows),
        'villain_vocabulary_replies':sum(bool(x['villain_terms']) for x in rows),
        'hamster_vocabulary_replies':sum(bool(x['hamster_terms']) for x in rows)}


def replication_criteria(data):
    rows=data['replies']; cfg=data['config']
    base={x['prompt_id']:x for x in rows if x['condition']=='baseline'}
    fresh=[x for x in base.values() if x['cohort']=='fresh-behavior']
    checks=[x for x in base.values() if x['cohort']=='fresh-check']
    baseline_mean=statistics.mean(x['tokens'] for x in fresh)
    lookup={(x['condition'],x['prompt_id']):x for x in rows}
    result={}
    for target in cfg['targets']:
        condition=target['candidate']+'@0.2'
        items=[lookup[condition,x['prompt_id']] for x in fresh]
        same_layer=[u for u in data['selected'] if u['role']=='same-layer-random' and u['layer']==target['layer'] and u['target_fraction']==.2]
        control_means={u['candidate']:statistics.mean(lookup[u['candidate']+'@0.2',x['prompt_id']]['tokens'] for x in fresh) for u in same_layer}
        lost=[x['prompt_id'] for x in checks if x['exact_check'] is True and lookup[condition,x['prompt_id']]['exact_check'] is False]
        added=[x['prompt_id'] for x in fresh if x['eos'] and not lookup[condition,x['prompt_id']]['eos']]
        repaired=[x['prompt_id'] for x in fresh if not x['eos'] and lookup[condition,x['prompt_id']]['eos']]
        mean=statistics.mean(x['tokens'] for x in items)
        shorter=sum(x['tokens']<base[x['prompt_id']]['tokens'] for x in items)
        if target['layer']==4:
            passed=shorter>=4 and mean<=.9*baseline_mean and all(mean<v for v in control_means.values()) and not lost and not added
            status='met' if passed else 'not met'
        else:
            status='inconclusive: no fresh baseline cap' if all(x['eos'] for x in fresh) else ('met' if repaired and not added and not lost else 'not met')
        result[condition]={'status':status,'fresh_behavior_n':len(fresh),'baseline_mean_tokens':baseline_mean,
            'target_mean_tokens':mean,'shorter_replies':shorter,'relative_mean_reduction':1-mean/baseline_mean,
            'same_layer_control_mean_tokens':control_means,'lost_exact_checks':lost,'added_cap_failures':added,
            'repaired_caps':repaired,'baseline_caps':[x['prompt_id'] for x in fresh if not x['eos']]}
    return result


def collect(name):
    folder=(ROOT/'runs'/name).resolve()
    if folder.parent!=(ROOT/'runs').resolve(): raise ValueError('Use a local run ID')
    verified=verify(folder)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    cfg=metadata['experiment_config']; experiment=metadata['experiment']
    prompt_name=cfg.get('evaluation_prompts',cfg.get('prompt_file'))
    prompts=json.loads((folder/'source'/prompt_name).read_text(encoding='utf-8')); prompt_lookup={x['id']:x for x in prompts}
    outputs=json.loads((folder/'outputs.json').read_text(encoding='utf-8'))
    baseline={x['prompt_id']:x for x in outputs if x['condition']=='baseline' and x['phase']=='probe'}
    if set(baseline)!=set(prompt_lookup): raise ValueError('Incomplete baselines')
    rows=[]
    for x in outputs:
        if x['phase']!='probe': continue
        if measure(x,prompt_lookup[x['prompt_id']])!=x['measurements']: raise ValueError('Stored measurements differ')
        exact=[v for k,v in x['measurements'].items() if k in ['exact_json_match','exact_text_match']]
        row={'condition':label(x),'prompt_id':x['prompt_id'],'cohort':x['cohort'],'domain':x['domain'],
            'persona':x.get('persona'),'role':x.get('role'),'layer':x.get('layer'),'target_fraction':x.get('target_fraction'),
            'dose':x.get('dose'),'tokens':x['output_tokens'],'eos':x['terminated_with_eos'],'seconds':x['seconds'],
            'repetition':repeated_fourgrams(x['token_ids']),'exact_check':exact[0] if exact else None,
            'changed':None if x['condition']=='baseline' else x['token_ids']!=baseline[x['prompt_id']]['token_ids'],
            'think_boundary':x['measurements']['reasoning_boundary_present'],
            'actual_prefill_fraction':x.get('actual_prefill_fraction'),'actual_prefill_by_layer':x.get('actual_prefill_by_layer'),
            'villain_terms':x.get('indicators',{}).get('villain_terms_present',[]),
            'hamster_terms':x.get('indicators',{}).get('hamster_terms_present',[])}
        rows.append(row)
    groups=defaultdict(list)
    for row in rows: groups[row['condition']].append(row)
    expected_conditions={'CHAR-001':28,'CHAR-002':12,'MAP-006':13}[experiment]
    if len(groups)!=expected_conditions or any(sorted(x['prompt_id'] for x in group)!=sorted(prompt_lookup) for group in groups.values()):
        raise ValueError('Missing/duplicate condition-prompt records')
    data={'schema':1,'experiment':experiment,'run':name,'model':metadata['model'],'config':cfg,
        'verification':verified,'source_hashes':metadata['source_hashes'],
        'raw_output_sha256':hashlib.sha256((folder/'outputs.json').read_bytes()).hexdigest(),
        'report_generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prompts':prompts,'replies':rows,'groups':[{**{'condition':key},**aggregate(items)} for key,items in sorted(groups.items())],
        'cohorts':[{**{'condition':key,'cohort':cohort},**aggregate([x for x in items if x['cohort']==cohort])}
                   for key,items in sorted(groups.items()) for cohort in sorted({x['cohort'] for x in items})],
        'limits':['One greedy trajectory per prompt/condition. No confidence intervals or validated persona score.',
                  'Character vocabulary misses some clear prompted style and is not a quality measure.',
                  'Local MLP-output change is not whole-residual or behavioral change. Joint layer effects include propagation.']}
    for name in ['profiles','rankings','selected']:
        path=folder/f'{name}.json'
        if path.exists(): data[name]=json.loads(path.read_text(encoding='utf-8'))
    if experiment=='CHAR-001':
        summary=json.loads((folder/'summary.json').read_text())
        if hashlib.sha256((folder/'profiles.json').read_bytes()).hexdigest()!=summary['profiles_sha256']: raise ValueError('Profiles changed')
    if experiment=='MAP-006':
        from replicate_offsets import select_units
        evidence=json.loads((folder/'source'/cfg['source_evidence']).read_text(encoding='utf-8'))
        if select_units(cfg,evidence['screening'])!=data['selected']: raise ValueError('Frozen controls changed')
        data['criteria']=replication_criteria(data)
    return data


def plot(data,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    destination.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for extension in ['png','svg']: fig.savefig(destination/f'{name}.{extension}',dpi=165,facecolor='white')
        plt.close(fig)
    def color(name):
        if name=='baseline': return '#777777'
        if 'random' in name: return '#a8a8a8'
        return '#b74c33' if 'villain' in name or name.startswith('L04N5848') else '#247c9e'
    groups=data['groups']; groups=sorted(groups,key=lambda x:(x['condition']!='baseline',x['condition']))
    if data['experiment']=='MAP-006':
        groups=[x for x in data['cohorts'] if x['cohort']=='fresh-behavior']
        groups=sorted(groups,key=lambda x:(x['condition']!='baseline',x['condition']))
        checks={x['condition']:x for x in data['cohorts'] if x['cohort']=='fresh-check'}
        groups=[{**x,'exact_pass':checks[x['condition']]['exact_pass'],'exact_n':checks[x['condition']]['exact_n']} for x in groups]
    labels=[x['condition'] for x in groups]; y=np.arange(len(groups)); colors=[color(x) for x in labels]
    fig,axes=plt.subplots(1,4,figsize=(17,max(6,len(groups)*.3+1.6)),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['mean_tokens','eos','exact_pass','mean_repetition'],['Mean output tokens','Replies reaching EOS',f'Strict checks passed / {groups[0]["exact_n"]}','Mean repeated 4-gram fraction']):
        ax.barh(y,[x[key] for x in groups],color=colors)
        ax.set(xlabel=title,yticks=y,yticklabels=labels)
        if key in ['eos','exact_pass']: ax.set_xlim(0,(groups[0]['n'] if key=='eos' else groups[0]['exact_n'])+.7)
        for i,row in enumerate(groups): ax.text(row[key],i,(' '+str(row[key]) if key in ['eos','exact_pass'] else f' {row[key]:.2f}'),va='center',fontsize=7)
        ax.margins(x=.17)
    axes[0].invert_yaxis()
    scope='six fresh behavioral prompts' if data['experiment']=='MAP-006' else f"{groups[0]['n']} prompts per condition; task/character cohorts also exported separately"
    fig.suptitle(data['experiment']+' · Direct comparisons · '+scope,fontsize=12)
    fig.supxlabel('Greedy, single trajectory. EOS and repetition are structural indicators; no personality or general quality score.',fontsize=9)
    save(fig,'comparisons')
    prompt_ids=[p['id'] for p in data['prompts']]
    lookup={(x['condition'],x['prompt_id']):x for x in data['replies']}
    labels=sorted({x['condition'] for x in data['replies'] if x['condition']!='baseline'})
    matrix=np.array([[int(lookup[c,p]['changed']) for p in prompt_ids] for c in labels])
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(12,max(5,len(labels)*.29+2)),layout='constrained')
    im=ax.imshow(matrix,cmap=ListedColormap(['#ececec','#247c9e']),vmin=-.5,vmax=1.5,aspect='auto')
    ax.set(xticks=range(len(prompt_ids)),xticklabels=prompt_ids,yticks=range(len(labels)),yticklabels=labels,
           title=data['experiment']+' · Complete token-sequence comparison map')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
    colorbar=fig.colorbar(im,ax=ax,ticks=[0,1],shrink=.4); colorbar.ax.set_yticklabels(['Identical','Changed'])
    fig.supxlabel('Blue measures any sequence difference, not successful character transfer. Prompted controls change the system instruction.',fontsize=9)
    save(fig,'change-map')
    if data.get('rankings'):
        fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
        for row,persona in enumerate(['villain','hamster']):
            for col,layer in enumerate(data['config']['layers']):
                entries=data['rankings'][f'{persona}:{layer}'][:16]
                ax=axes[row,col]; positions=np.arange(len(entries)); values=[x['standardized_difference'] for x in entries]
                ax.barh(positions,values,color=['#b74c33' if x<0 else '#247c9e' for x in values])
                ax.set(yticks=positions,yticklabels=[str(x['neuron']) for x in entries],title=f'{persona} · layer {layer}',xlabel='Paired mean difference / neutral RMS')
                ax.invert_yaxis(); ax.axvline(0,color='#333333',linewidth=.5)
        fig.suptitle('CHAR-001 · Sixteen selected channels per profile · Fixed text associations',fontsize=13)
        fig.supxlabel('Channels selected before generation. Sign is activation contrast, not goodness/evilness. Lexical/content/length confounds remain.',fontsize=9)
        save(fig,'channel-map')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run'); parser.add_argument('--evidence',type=Path); parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    if not args.run and not args.evidence: parser.error('Use --run or --evidence')
    data=json.loads(args.evidence.read_text(encoding='utf-8')) if args.evidence else collect(args.run)
    destination=args.destination or ROOT/'experiments/evidence'/data['experiment']
    if not args.evidence: write_json(destination/'metrics.json',data)
    plot(data,destination)
    print(json.dumps({'destination':str(destination),'verification':data['verification'],'criteria':data.get('criteria')},indent=2))


if __name__=='__main__': main()
