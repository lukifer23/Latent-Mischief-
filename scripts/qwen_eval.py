"""Bounded, recorded Qwen base/edit sampling pilot. No changes to greedy chat."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import lab
import runtime
from live_worker import Models
from math_grade import grade
from measurements import repeated_fourgrams
from qwen_sampling import OutputPresencePenalty


SOURCES=['scripts/qwen_eval.py','scripts/qwen_sampling.py','scripts/math_grade.py','scripts/qwen_audit.py',
    'scripts/prepare_qwen_pilot.py','scripts/live_worker.py','scripts/lab.py','scripts/calibration.py',
    'scripts/character_lab.py','scripts/personas.py','scripts/catalog.py','scripts/measurements.py','scripts/storage.py',
    'scripts/viewer.py','configs/models.json','configs/qwen-pilot.json','pyproject.toml','uv.lock','.cache/qwn002/problems.json']
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prefix(outputs,schedule):
    actual=[(x['condition'],x['prompt_id'],x['seed']) for x in outputs]
    expected=[(x['condition'],x['prompt_id'],x['seed']) for x in schedule]
    if actual!=expected[:len(actual)] or len(set(actual))!=len(actual):raise ValueError('Saved outputs are not a unique schedule prefix')


def generation_config(model,tokenizer,cfg,limit):
    from transformers import GenerationConfig
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tokenizer.eos_token_id);eos.discard(None)
    return GenerationConfig(do_sample=True,temperature=cfg['temperature'],top_k=cfg['top_k'],top_p=cfg['top_p'],
        repetition_penalty=1.,num_beams=1,max_new_tokens=limit,min_new_tokens=0,
        eos_token_id=sorted(eos),pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        use_cache=True)


class Recorder:
    def __init__(self,tokenizer,callback,selected):
        self.tokenizer=tokenizer;self.callback=callback;self.selected=selected;self.tokens=[];self.prompt_seen=False;self.first=None;self.ended=False
    def put(self,value):
        import torch
        if not self.prompt_seen:self.prompt_seen=True;return
        if self.first is None:self.selected.synchronize();self.first=time.perf_counter()
        self.tokens.extend(value.reshape(-1).tolist())
        if len(self.tokens)%64==0:self.callback(self.tokens,self.tokenizer.decode(self.tokens,skip_special_tokens=False))
    def end(self):self.ended=True


def sampled(model,tokenizer,messages,cfg,limit,deadline,callback):
    import torch
    import psutil
    from transformers import LogitsProcessorList,StoppingCriteria,StoppingCriteriaList
    inputs=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=True,
        return_tensors='pt',return_dict=True).to(model.device)
    prompt_length=inputs['input_ids'].shape[1]
    maximum=model.config.text_config.max_position_embeddings
    if prompt_length+limit>maximum:raise ValueError('Input plus output exceeds model context window')
    class Deadline(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):return torch.full((input_ids.shape[0],),time.perf_counter()>=deadline,device=input_ids.device,dtype=torch.bool)
    selected=runtime.for_model(model)
    recorder=Recorder(tokenizer,callback,selected)
    torch.manual_seed(cfg['seed']);selected.reset_peak_memory();selected.synchronize()
    started=time.perf_counter();prefill_end=[]
    def forward_done(module,args,output):
        if not prefill_end:selected.synchronize();prefill_end.append(time.perf_counter())
    handle=model.register_forward_hook(forward_done)
    try:
        with torch.inference_mode():
            output=model.generate(**inputs,generation_config=generation_config(model,tokenizer,cfg,limit),
                logits_processor=LogitsProcessorList([OutputPresencePenalty(prompt_length,cfg['presence_penalty'])]),
                stopping_criteria=StoppingCriteriaList([Deadline()]),streamer=recorder,logits_to_keep=1)
        selected.synchronize();ended=time.perf_counter()
    except BaseException:
        callback(recorder.tokens,tokenizer.decode(recorder.tokens,skip_special_tokens=False));raise
    finally:handle.remove()
    tokens=output[0,prompt_length:].tolist()
    if tokens!=recorder.tokens or not recorder.ended:raise ValueError('Recorded stream differs from returned tokens or did not finish')
    eos=generation_config(model,tokenizer,cfg,limit).eos_token_id
    termination='eos' if tokens and tokens[-1] in eos else 'wall-limit' if ended>=deadline else 'output-cap'
    return {'text':tokenizer.decode(tokens,skip_special_tokens=False),'token_ids':tokens,'input_token_ids':inputs['input_ids'][0].tolist(),
        'input_tokens':prompt_length,'output_tokens':len(tokens),'seconds':ended-started,
        'prefill_seconds':prefill_end[0]-started,'time_to_first_token_seconds':recorder.first-started,
        'decode_tokens_after_first':max(0,len(tokens)-1),'decode_seconds_after_first':ended-recorder.first,
        'decode_tokens_per_second':(len(tokens)-1)/(ended-recorder.first) if len(tokens)>1 else None,
        'terminated_with_eos':termination=='eos','hit_token_limit':termination=='output-cap','termination':termination,
        'runtime':selected.identity(),**selected.memory(),
        'process_rss_end_bytes':psutil.Process().memory_info().rss,'generation_config':generation_config(model,tokenizer,cfg,limit).to_dict()}


def preflight(model,tokenizer,cfg,deadline):
    import torch
    messages=[{'role':'user','content':'What is 2 + 3?'}]
    on=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    off=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    check={'thinking_template_differs':on!=off,'thinking_prefix_open':on.rfind('<think>')>on.rfind('</think>')}
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=True,return_tensors='pt',return_dict=True).to(model.device)
    with torch.inference_mode():
        a=model(**ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        b=model(**ids,use_cache=True,logits_to_keep=1).logits[0,-1].float()
    check.update(cached_first_argmax_identical=a.argmax().item()==b.argmax().item(),logits_finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
    one=sampled(model,tokenizer,messages,cfg,8,deadline,lambda tokens,text:None)
    two=sampled(model,tokenizer,messages,cfg,8,deadline,lambda tokens,text:None)
    check['seeded_sample_repeat_identical']=one['token_ids']==two['token_ids']
    result={'checks':check,'max_abs_cached_logit_difference':float((a-b).abs().max()),'sample_token_ids':one['token_ids'],
        'sample_messages':messages,'sample_settings':cfg,'samples':[one,two],
        'sanity_generated_tokens':one['output_tokens']+two['output_tokens']}
    if not all(check.values()):raise ValueError('Qwen preflight failed: '+str(check))
    return result


def main():
    import psutil
    import torch
    from viewer import gpu_job_running
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume');args=parser.parse_args()
    selected=runtime.require_cuda_protocol('QWN-002')
    own=[os.getpid(),*[p.pid for p in psutil.Process().parents()]]
    if gpu_job_running(ignore_pids=own):raise RuntimeError('Another inference process is active; do not run concurrently')
    cfg=read(lab.ROOT/'configs/qwen-pilot.json');manifest_path=lab.ROOT/cfg['manifest'];manifest=read(manifest_path)
    if sha(manifest_path)!=cfg['manifest_sha256']:raise ValueError('Frozen problem manifest changed')
    audit_path=lab.ROOT/'runs/qwen-full-audit.json';audit=read(audit_path)
    if sha(audit_path)!=cfg['audit_sha256'] or not all(audit['checks'].values()):raise ValueError('Full checkpoint audit mismatch')
    models=read(lab.ROOT/'configs/models.json');spec=models['qwen08'];variants={'baseline':spec,'published-checkpoint':{**spec,**spec['published_edit']}}
    if [{k:x[k] for k in ['model_id','revision']} for x in variants.values()]!=cfg['models']:raise ValueError('Pinned models changed')
    schedule=cfg['schedule'];outputs=[];previous_elapsed=0.
    if args.resume:
        folder=(lab.ROOT/'runs'/args.resume).resolve()
        if folder.parent!=(lab.ROOT/'runs').resolve():raise ValueError('Invalid resume run')
        metadata=read(folder/'metadata.json');outputs=read(folder/'outputs.json');validate_prefix(outputs,schedule)
        for name,digest in metadata['source_hashes'].items():
            if sha(folder/'source'/name)!=digest or sha(lab.ROOT/name)!=digest:raise ValueError('Resume source/config/runtime differs: '+name)
        state=read(folder/'progress.json');previous_elapsed=state['elapsed_seconds']
        if state.get('active_trial'):raise ValueError('Unfinished trajectory preserved; refusing to reroll it. Review interruption instead.')
        if (folder/'summary.json').exists():raise ValueError('Run is already finalized; do not rerun its samples')
    else:
        folder=lab.ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-QWN-002-anti-loop-pilot');folder.mkdir()
        sources=lab.archive_sources(folder,SOURCES)
        lab.write_json(folder/'audit.json',audit)
        metadata={'experiment':'QWN-002','model_key':'qwen08','model':{**spec,'attention':'sdpa'},'experiment_config':cfg,
            'environment':lab.environment(),'source_hashes':sources,'math_verify_version':importlib.metadata.version('math-verify'),
            'scope':'Four-problem sampled Windows pilot; not an exact upstream benchmark reproduction. All trajectories retained.'}
        lab.write_json(folder/'metadata.json',metadata);lab.write_json(folder/'outputs.json',outputs)
    print('RUN '+folder.name,flush=True)
    start=time.perf_counter();deadline=start+cfg['wall_seconds']-previous_elapsed;manager=Models();checks=read(folder/'preflight.json') if (folder/'preflight.json').exists() else {}
    elapsed=lambda:previous_elapsed+time.perf_counter()-start
    active=None
    try:
        for trial in schedule[len(outputs):]:
            if time.perf_counter()>=deadline:raise TimeoutError('Cumulative wall budget exhausted before next trajectory')
            consumed=sum(x['output_tokens'] for x in outputs)
            if consumed+cfg['max_new_tokens']>cfg['total_generated_tokens']:raise RuntimeError('Insufficient remaining frozen token budget')
            model,tokenizer,load_seconds=manager.load(variants[trial['condition']],'sdpa')
            if trial['condition'] not in checks:
                checks[trial['condition']]=preflight(model,tokenizer,cfg,deadline);lab.write_json(folder/'preflight.json',checks)
            version={name:p._version for name,p in model.named_parameters()}
            problem=next(p for p in manifest['problems'] if p['unique_id']==trial['prompt_id'])
            prompt=problem['problem']+cfg['prompt_suffix'];messages=[{'role':'user','content':prompt}]
            active=trial
            def progress(tokens,text):
                lab.write_json(folder/'active.json',{'trial':trial,'token_ids':tokens,'text':text,'elapsed_seconds':elapsed()})
                lab.write_json(folder/'progress.json',{'status':'running','completed':len(outputs),'planned':len(schedule),'active_trial':trial,'active_tokens':len(tokens),'elapsed_seconds':elapsed()})
            progress([],'')
            print('START '+json.dumps(trial),flush=True)
            result=sampled(model,tokenizer,messages,cfg,cfg['max_new_tokens'],deadline,progress)
            if version!={name:p._version for name,p in model.named_parameters()}:raise RuntimeError('Parameters changed during inference')
            progress(result['token_ids'],result['text'])
            lab.write_json(folder/'pending-result.json',{**trial,**result})
            scoring=grade(result['text'],problem['answer'],timeout=cfg['grader_timeout_seconds'])
            result.update(trial,phase='probe',cohort='pilot-math',prompt=prompt,system=None,
                expected_answer=problem['answer'],model_id=variants[trial['condition']]['model_id'],revision=variants[trial['condition']]['revision'],
                grading=scoring,measurements={'exact_math_match':scoring['correct'],'repeated_fourgram_fraction':repeated_fourgrams(result['token_ids'])},
                load_seconds=load_seconds,parameter_version_unchanged=True,status='complete')
            outputs.append(result);lab.write_json(folder/'outputs.json',outputs);active=None
            lab.write_json(folder/'progress.json',{'status':'running','completed':len(outputs),'planned':len(schedule),'active_trial':None,'elapsed_seconds':elapsed()})
            print('SAVED '+json.dumps({k:result[k] for k in ['condition','prompt_id','output_tokens','termination','grading']}),flush=True)
            del model,tokenizer
            if result['termination']=='wall-limit':raise TimeoutError('Cumulative wall budget exhausted during trajectory')
        validate_prefix(outputs,schedule)
        lab.write_json(folder/'summary.json',{'completed':True,'generations':len(outputs),'planned':len(schedule),'generated_tokens':sum(x['output_tokens'] for x in outputs),
            'elapsed_seconds':elapsed(),'preflight_checks':checks,'parameter_version_checks':all(x['parameter_version_unchanged'] for x in outputs)})
        lab.write_json(folder/'progress.json',{'status':'complete','completed':len(outputs),'planned':len(schedule),'active_trial':None,'elapsed_seconds':elapsed()})
        print('COMPLETE '+folder.name,flush=True)
    except BaseException as error:
        lab.write_json(folder/'interruption.json',{'type':type(error).__name__,'reason':str(error),'elapsed_seconds':elapsed(),'active_trial':active,'completed':len(outputs),'planned':len(schedule)})
        raise
    finally:
        manager.model=None;manager.tokenizer=None
        import gc
        gc.collect();selected.empty_cache()


if __name__=='__main__':main()
