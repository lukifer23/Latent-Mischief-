"""Short cache compatibility checks, loaded Qwen digests and bounded sampling sanity."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import lab
import runtime
from live_worker import Models


def main():
    import torch
    import psutil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['lfm3','qwen08'],default='qwen08')
    parser.add_argument('--attention',choices=['eager','sdpa'])
    args=parser.parse_args()
    from viewer import gpu_job_running
    own=[os.getpid(),*[p.pid for p in psutil.Process().parents()]]
    if gpu_job_running(ignore_pids=own):raise RuntimeError('Wait until the existing inference experiment has exited')
    registry=json.loads((lab.ROOT/'configs/models.json').read_text(encoding='utf-8'))
    spec=registry[args.model];manager=Models();results=[]
    attention=args.attention or spec['live_attention']
    audit_path=lab.ROOT/'experiments/evidence/QWN-002/weights.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8')) if args.model=='qwen08' else None
    variants=[('baseline',spec)]
    if audit:
        if not all(audit['checks'].values()):raise ValueError('Public weight audit failed')
        variants.append(('published-checkpoint',{**spec,**spec['published_edit']}))
        if audit['models']!=[{k:v[k] for k in ['model_id','revision']} for _,v in variants]:raise ValueError('Audit model pins differ')
    folder=lab.ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-BACKEND-001-cache-'+args.model)
    folder.mkdir()
    sources=lab.archive_sources(folder,['scripts/cache_diagnostic.py','scripts/live_worker.py','scripts/lab.py',
        'scripts/qwen_eval.py','scripts/qwen_sampling.py','scripts/math_grade.py','scripts/calibration.py',
        'scripts/character_lab.py','scripts/calibrated_map.py','scripts/personas.py','scripts/storage.py',
        'configs/models.json','configs/model.json','configs/qwen-pilot.json','pyproject.toml','uv.lock',
        *(['experiments/evidence/QWN-002/weights.json'] if audit else [])])
    lab.write_json(folder/'metadata.json',{'experiment':'BACKEND-001','model_key':args.model,'model':spec,
        'runtime_attention':attention,'environment':lab.environment(),'source_hashes':sources,
        'thinking':True,'use_cache':[False,True],'purpose':'Short backend/cache compatibility, not anti-loop evaluation.'})
    print('RUN '+folder.name,flush=True)
    text='Count from one to five and explain what addition means.'
    for label,variant in variants:
        model,tokenizer,load_seconds=manager.load(variant,attention)
        selected=runtime.for_model(model)
        loaded_digests={}
        for entry in audit['changed_tensors'] if audit else []:
            actual=hashlib.sha256(model.get_parameter(entry['tensor']).detach().to(device='cpu',dtype=torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            expected=entry['base_sha256' if label=='baseline' else 'edited_sha256']
            if actual!=expected:raise ValueError('Loaded tensor differs from audited checkpoint: '+entry['tensor'])
            loaded_digests[entry['tensor']]=actual
        version={n:p._version for n,p in model.named_parameters()}
        inputs=tokenizer.apply_chat_template([{'role':'user','content':text}],enable_thinking=True,
            tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to(model.device)
        ids=inputs['input_ids'];length=ids.shape[1];prefix=max(1,length-16);selected.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            reference=model(**inputs,use_cache=False,logits_to_keep=0).logits[0].float()
            out=model(input_ids=ids[:,:prefix],attention_mask=torch.ones((1,prefix),device=model.device,dtype=torch.long),use_cache=True,logits_to_keep=1)
            cache=out.past_key_values;points=[]
            for position in range(prefix-1,length):
                if position>=prefix:
                    out=model(input_ids=ids[:,position:position+1],attention_mask=torch.ones((1,position+1),device=model.device,dtype=torch.long),past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache=out.past_key_values
                a=reference[position];b=out.logits[0,-1].float()
                pa=a.softmax(-1);kl=float((pa*(a.log_softmax(-1)-b.log_softmax(-1))).sum())
                points.append({'prefix_tokens':position+1,'argmax_identical':a.argmax().item()==b.argmax().item(),
                    'max_abs_logit_difference':float((a-b).abs().max()),'reference_to_cached_kl_nats':kl,
                    'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())})
        if version!={n:p._version for n,p in model.named_parameters()}:raise ValueError('Parameters changed during cache diagnostic')
        selected.synchronize()
        result={'condition':label,'model_id':variant['model_id'],'revision':variant['revision'],'input_token_ids':ids[0].tolist(),
            'loaded_tensor_sha256':loaded_digests,'load_seconds':load_seconds,'seconds':time.perf_counter()-start,'points':points,'parameter_versions_unchanged':True}
        if audit:
            from qwen_eval import preflight
            cfg=json.loads((lab.ROOT/'configs/qwen-pilot.json').read_text())
            result['sampling_preflight']=preflight(model,tokenizer,cfg,time.perf_counter()+120)
        results.append(result)
        lab.write_json(folder/'outputs.json',results)
        del model,tokenizer,out,cache,reference,inputs,ids
    result={'scope':'Fixed teacher-forced text; one short input, not general cache proof or generation benchmark.',
        'input_text':text,'environment':lab.environment(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'loader_sha256':hashlib.sha256((lab.ROOT/'scripts/live_worker.py').read_bytes()).hexdigest(),'results':results}
    lab.write_json(folder/'diagnostic.json',result)
    checks={'all_logits_finite':all(p['finite'] for x in results for p in x['points']),
        'all_cached_argmax_identical':all(p['argmax_identical'] for x in results for p in x['points']),
        'parameter_versions_unchanged':all(x['parameter_versions_unchanged'] for x in results)}
    lab.write_json(folder/'checks.json',checks)
    lab.write_json(folder/'summary.json',{'completed':True,'checks':checks,'comparisons':len(results),
        'scope':'Diagnostic finished; inspect each check and numerical differences. Not general cache equivalence.'})
    print(json.dumps({'comparisons':[{'condition':x['condition'],'argmax_matches':sum(p['argmax_identical'] for p in x['points']),
        'positions':len(x['points']),'maximum_abs_difference':max(p['max_abs_logit_difference'] for p in x['points']),
        'maximum_kl_nats':max(p['reference_to_cached_kl_nats'] for p in x['points'])} for x in results]},indent=2))


if __name__=='__main__':main()
