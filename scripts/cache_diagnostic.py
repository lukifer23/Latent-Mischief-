"""Real teacher-forced full-prefix versus incremental-cache logit comparison; no sampling."""
import hashlib
import json
from pathlib import Path
import time

import lab
from live_worker import Models


def main():
    import torch
    from viewer import gpu_job_running
    if gpu_job_running():raise RuntimeError('Wait until the existing GPU experiment has exited')
    registry=json.loads((lab.ROOT/'configs/models.json').read_text(encoding='utf-8'))
    spec=registry['qwen08'];manager=Models();results=[]
    audit=json.loads((lab.ROOT/'runs/qwen-full-audit.json').read_text(encoding='utf-8'))
    text='Count from one to five and explain what addition means.'
    for label,variant in [('baseline',spec),('published-checkpoint',{**spec,**spec['published_edit']})]:
        model,tokenizer,load_seconds=manager.load(variant,'sdpa')
        loaded_digests={}
        for entry in audit['changed_tensors']:
            actual=hashlib.sha256(model.get_parameter(entry['tensor']).detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            expected=entry['base_sha256' if label=='baseline' else 'edited_sha256']
            if actual!=expected:raise ValueError('Loaded GPU tensor differs from audited checkpoint: '+entry['tensor'])
            loaded_digests[entry['tensor']]=actual
        version={n:p._version for n,p in model.named_parameters()}
        inputs=tokenizer.apply_chat_template([{'role':'user','content':text}],enable_thinking=True,
            tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to(model.device)
        ids=inputs['input_ids'];length=ids.shape[1];prefix=max(1,length-16);start=time.perf_counter()
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
        results.append({'condition':label,'model_id':variant['model_id'],'revision':variant['revision'],'input_token_ids':ids[0].tolist(),
            'loaded_gpu_tensor_sha256':loaded_digests,'load_seconds':load_seconds,'seconds':time.perf_counter()-start,'points':points,'parameter_versions_unchanged':True})
        del model,tokenizer,out,cache,reference,inputs,ids
    result={'scope':'Fixed teacher-forced text; one short input, not general cache proof or generation benchmark.',
        'input_text':text,'environment':lab.environment(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'loader_sha256':hashlib.sha256((lab.ROOT/'scripts/live_worker.py').read_bytes()).hexdigest(),'results':results}
    lab.write_json(lab.ROOT/'runs/qwen-cache-diagnostic.json',result)
    print(json.dumps({'comparisons':[{'condition':x['condition'],'argmax_matches':sum(p['argmax_identical'] for p in x['points']),
        'positions':len(x['points']),'maximum_abs_difference':max(p['max_abs_logit_difference'] for p in x['points']),
        'maximum_kl_nats':max(p['reference_to_cached_kl_nats'] for p in x['points'])} for x in results]},indent=2))


if __name__=='__main__':main()
