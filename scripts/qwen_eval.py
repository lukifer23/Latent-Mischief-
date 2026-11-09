"""Bounded, recorded Qwen base/edit sampling pilot. No changes to greedy chat."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import lab
import runtime
from live_worker import Models
from math_grade import grade
from measurements import repeated_fourgrams
from qwen_sampling import OutputPresencePenalty
from token_stream import TokenRecorder as Recorder


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


def sampled(model,tokenizer,messages,cfg,limit,deadline,callback,check=None):
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
    recorder=Recorder(tokenizer,callback,selected,check)
    torch.manual_seed(cfg['seed']);selected.reset_peak_memory();selected.synchronize()
    started=time.perf_counter();prefill_end=[]
    def forward_done(module,args,output):
        if not prefill_end:selected.synchronize();prefill_end.append(time.perf_counter())
    handle=model.register_forward_hook(forward_done)
    try:
        if check:check()
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


def preflight(model,tokenizer,cfg,deadline,check=None,callback=None):
    import torch
    messages=[{'role':'user','content':'What is 2 + 3?'}]
    on=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    off=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    checks={'thinking_template_differs':on!=off,'thinking_prefix_open':on.rfind('<think>')>on.rfind('</think>')}
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=True,return_tensors='pt',return_dict=True).to(model.device)
    with torch.inference_mode():
        if check:check()
        a=model(**ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        b=model(**ids,use_cache=True,logits_to_keep=1).logits[0,-1].float()
    checks.update(cached_first_argmax_identical=a.argmax().item()==b.argmax().item(),logits_finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
    one=sampled(model,tokenizer,messages,cfg,8,deadline,callback or (lambda tokens,text:None),check)
    two=sampled(model,tokenizer,messages,cfg,8,deadline,callback or (lambda tokens,text:None),check)
    checks['seeded_sample_repeat_identical']=one['token_ids']==two['token_ids']
    result={'checks':checks,'max_abs_cached_logit_difference':float((a-b).abs().max()),'sample_token_ids':one['token_ids'],
        'sample_messages':messages,'sample_settings':cfg,'samples':[one,two],
        'sanity_generated_tokens':one['output_tokens']+two['output_tokens']}
    if not all(checks.values()):raise ValueError('Qwen preflight failed: '+str(checks))
    return result



def validate_design(cfg, manifest, selected):
    if cfg['experiment_id'] == 'QWN-002':
        runtime.require_cuda_protocol('QWN-002')
    elif cfg['experiment_id'] == 'QWN-003':
        import platform
        if platform.system() != 'Darwin' or selected.backend != 'mps' or selected.dtype_name != 'bfloat16':
            raise ValueError('QWN-003 requires the separately frozen macOS MPS/BF16 runtime')
        if selected.identity()['mps_cpu_fallback'] != '0' or selected.identity()['mps_fast_math'] != '0':
            raise ValueError('QWN-003 requires default math and no MPS CPU fallback')
        if cfg['required_backend'] != 'mps' or cfg['max_new_tokens'] != 8192 or cfg['total_generated_tokens'] != 32768:
            raise ValueError('QWN-003 frozen backend/token budget changed')
        if cfg['wall_seconds'] != 1800 or len(manifest['problems']) != 2:
            raise ValueError('QWN-003 requires two problems and a 30-minute ceiling')
    else:
        raise ValueError('Unknown versioned Qwen protocol')
    ids=[p['unique_id'] for p in manifest['problems']]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate problem IDs')
    schedule=cfg['schedule']
    expected={(c,p,cfg['seed']) for c in ['baseline','published-checkpoint'] for p in ids}
    actual=[(x['condition'],x['prompt_id'],x['seed']) for x in schedule]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Schedule must cover each checkpoint/problem exactly once with the frozen seed')
    if cfg['thinking'] is not True or cfg['system'] is not None or cfg['dtype']!='bfloat16':
        raise ValueError('Thinking/system/dtype policy differs from the protocol')
    if cfg['max_new_tokens']*len(schedule)>cfg['total_generated_tokens']:
        raise ValueError('Schedule exceeds the token budget')


def check_resume(folder,cfg,current_environment):
    metadata=read(folder/'metadata.json')
    if metadata['experiment_config']!=cfg or not metadata['source_hashes']:
        raise ValueError('Resume protocol differs or source archive is absent')
    outputs=read(folder/'outputs.json')
    validate_prefix(outputs,cfg['schedule'])
    for name,digest in metadata['source_hashes'].items():
        if sha(folder/'source'/name)!=digest or sha(lab.ROOT/name)!=digest:
            raise ValueError('Resume source/config/runtime differs: '+name)
    for key in ['python','platform','architecture','packages','runtime','system_memory_bytes']:
        if metadata['environment'].get(key)!=current_environment.get(key):
            raise ValueError('Resume environment differs: '+key)
    state=read(folder/'progress.json')
    if state.get('active_trial'):
        raise ValueError('Unfinished trajectory preserved; refusing to reroll it. Review interruption instead.')
    if (folder/'summary.json').exists():
        raise ValueError('Run is already finalized; do not rerun its samples')
    interruption=read(folder/'interruption.json') if (folder/'interruption.json').exists() else {}
    previous_elapsed=max(state['elapsed_seconds'],interruption.get('elapsed_seconds',0))
    return metadata,outputs,previous_elapsed


def main():
    import psutil
    import signal
    import torch
    from viewer import gpu_job_running
    from resource_guard import ResourceGuard,ResourceLimitError,snapshot
    from qwen_preflight import loaded_digests,bounded_prefill
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume')
    parser.add_argument('--config',type=Path,default=lab.ROOT/'configs/qwen-pilot.json')
    args=parser.parse_args()
    config_path=args.config.resolve()
    if not config_path.is_relative_to(lab.ROOT):
        raise ValueError('Protocol config must be inside this repository')
    cfg=read(config_path)
    selected=runtime.select()
    if cfg['experiment_id']=='QWN-002':
        runtime.require_cuda_protocol('QWN-002')
    own=[os.getpid(),*[p.pid for p in psutil.Process().parents()]]
    if gpu_job_running(ignore_pids=own):
        raise RuntimeError('Another inference process is active; do not run concurrently')
    manifest_path=lab.ROOT/cfg['manifest']
    manifest=read(manifest_path)
    if sha(manifest_path)!=cfg['manifest_sha256']:
        raise ValueError('Frozen problem manifest changed')
    validate_design(cfg,manifest,selected)
    audit_path=lab.ROOT/'runs/qwen-full-audit.json'
    audit=read(audit_path)
    if sha(audit_path)!=cfg['audit_sha256'] or not all(audit['checks'].values()):
        raise ValueError('Full checkpoint audit mismatch')
    models=read(lab.ROOT/'configs/models.json')
    spec=models['qwen08']
    variants={'baseline':spec,'published-checkpoint':{**spec,**spec['published_edit']}}
    if [{k:x[k] for k in ['model_id','revision']} for x in variants.values()]!=cfg['models'] or audit['models']!=cfg['models']:
        raise ValueError('Pinned models changed')
    if cfg.get('resources'):
        torch.set_num_threads(cfg['cpu_threads'])
        torch.set_num_interop_threads(1)
        os.nice(10)
        torch.mps.set_per_process_memory_fraction(cfg['resources']['mps_memory_fraction'])
    current_environment=lab.environment()
    schedule=cfg['schedule']
    outputs=[]
    previous_elapsed=0.
    if args.resume:
        folder=(lab.ROOT/'runs'/args.resume).resolve()
        if folder.parent!=(lab.ROOT/'runs').resolve():
            raise ValueError('Invalid resume run')
        metadata,outputs,previous_elapsed=check_resume(folder,cfg,current_environment)
    else:
        folder=lab.ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+cfg['experiment_id']+'-anti-loop-pilot')
        folder.mkdir()
        source_names=[name for name in SOURCES if name not in ['configs/qwen-pilot.json','.cache/qwn002/problems.json']]
        source_names += [config_path.relative_to(lab.ROOT).as_posix(),cfg['manifest'],
                         'scripts/resource_guard.py','scripts/qwen_preflight.py','scripts/calibrated_map.py']
        if cfg.get('protocol_report'):source_names.append(cfg['protocol_report'])
        sources=lab.archive_sources(folder,source_names)
        lab.write_json(folder/'audit.json',audit)
        metadata={'experiment':cfg['experiment_id'],'model_key':'qwen08','model':{**spec,'attention':'sdpa'},
            'experiment_config':cfg,'environment':current_environment,'source_hashes':sources,
            'math_verify_version':importlib.metadata.version('math-verify'),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=lab.ROOT,text=True).strip(),
            'process_nice':psutil.Process().nice(),'torch_cpu_threads':torch.get_num_threads(),
            'scope':cfg.get('scope','Four-problem sampled Windows pilot; not exact upstream reproduction.')}
        if cfg.get('resources'):
            metadata['mps_recommended_bytes']=torch.mps.recommended_max_memory()
            metadata['mps_allocator_limit_bytes']=int(metadata['mps_recommended_bytes']*cfg['resources']['mps_memory_fraction'])
        lab.write_json(folder/'metadata.json',metadata)
        lab.write_json(folder/'outputs.json',outputs)
    print('RUN '+folder.name,flush=True)
    start=time.perf_counter()
    deadline=start+cfg['wall_seconds']-previous_elapsed
    manager=Models()
    checks=read(folder/'preflight.json') if (folder/'preflight.json').exists() else {}
    elapsed=lambda:previous_elapsed+time.perf_counter()-start
    active=None
    guard=None
    exit_code=0
    resource_check=lambda:None
    def save_progress(status='running',tokens=None):
        if tokens is None and active and (folder/'active.json').exists():
            saved=read(folder/'active.json')
            if saved.get('trial')==active:tokens=saved['token_ids']
        lab.write_json(folder/'progress.json',{'status':status,'completed':len(outputs),'planned':len(schedule),
            'active_trial':active,'active_tokens':len(tokens) if tokens is not None else 0,
            'stage':guard.stage if guard else None,'elapsed_seconds':elapsed()})
    def requested_stop(signum,frame):
        raise KeyboardInterrupt('Signal requested stop: '+str(signum))
    signal.signal(signal.SIGTERM,requested_stop)
    save_progress()
    try:
        if cfg.get('resources'):
            initial_swap=metadata.get('initial_swap_used_bytes')
            guard=ResourceGuard(folder,selected,cfg['resources'],deadline,initial_swap=initial_swap).start()
            resource_check=guard.check
            metadata['initial_swap_used_bytes']=guard.initial_swap
            lab.write_json(folder/'metadata.json',metadata)
            resource_check()
        for trial in schedule[len(outputs):]:
            resource_check()
            if time.perf_counter()>=deadline:
                raise TimeoutError('Cumulative wall budget exhausted before next trajectory')
            consumed=sum(x['output_tokens'] for x in outputs)
            if consumed+cfg['max_new_tokens']>cfg['total_generated_tokens']:
                raise RuntimeError('Insufficient remaining frozen token budget')
            if guard:guard.stage='loading '+trial['condition']
            save_progress()
            model,tokenizer,load_seconds=manager.load(variants[trial['condition']],'sdpa')
            resource_check()
            if trial['condition'] not in checks:
                if guard:guard.stage='preflight '+trial['condition']
                digests=loaded_digests(model,audit,trial['condition'])
                def sanity_progress(tokens,text):
                    lab.write_json(folder/'preflight-active.json',{'condition':trial['condition'],'token_ids':tokens,'text':text})
                result=preflight(model,tokenizer,cfg,deadline,resource_check,sanity_progress)
                if cfg.get('resources'):
                    def emit_prefill(rows):
                        lab.write_json(folder/('prefill-'+trial['condition']+'.json'),rows)
                    result['intermediate_prefill']=bounded_prefill(model,tokenizer,cfg['resources'],resource_check,emit_prefill)
                result['loaded_tensor_sha256']=digests
                checks[trial['condition']]=result
                lab.write_json(folder/'preflight.json',checks)
            version={name:p._version for name,p in model.named_parameters()}
            problem=next(p for p in manifest['problems'] if p['unique_id']==trial['prompt_id'])
            prompt=problem['problem']+cfg['prompt_suffix']
            messages=[{'role':'user','content':prompt}]
            active=trial
            def progress(tokens,text):
                lab.write_json(folder/'active.json',{'trial':trial,'prompt':prompt,'messages':messages,
                    'token_ids':tokens,'text':text,'elapsed_seconds':elapsed()})
                save_progress(tokens=tokens)
            if guard:guard.stage='generating '+trial['condition']+' '+trial['prompt_id']
            progress([],'')
            print('START '+json.dumps(trial),flush=True)
            result=sampled(model,tokenizer,messages,cfg,cfg['max_new_tokens'],deadline,progress,resource_check)
            if version!={name:p._version for name,p in model.named_parameters()}:
                raise RuntimeError('Parameters changed during inference')
            progress(result['token_ids'],result['text'])
            lab.write_json(folder/'pending-result.json',{**trial,**result})
            resource_check()
            if guard:guard.stage='grading'
            scoring=grade(result['text'],problem['answer'],timeout=cfg['grader_timeout_seconds'])
            result.update(trial,phase='probe',cohort='inspected-followup' if cfg['experiment_id']=='QWN-003' else 'pilot-math',
                prompt=prompt,system=None,expected_answer=problem['answer'],model_id=variants[trial['condition']]['model_id'],
                revision=variants[trial['condition']]['revision'],grading=scoring,
                measurements={'exact_math_match':scoring['correct'],'repeated_fourgram_fraction':repeated_fourgrams(result['token_ids'])},
                load_seconds=load_seconds,parameter_version_unchanged=True,status='complete')
            outputs.append(result)
            lab.write_json(folder/'outputs.json',outputs)
            active=None
            save_progress()
            print('SAVED '+json.dumps({k:result[k] for k in ['condition','prompt_id','output_tokens','termination','grading']}),flush=True)
            del model,tokenizer
            if result['termination']=='wall-limit':
                raise TimeoutError('Cumulative wall budget exhausted during trajectory')
            resource_check()
        validate_prefix(outputs,schedule)
        lab.write_json(folder/'summary.json',{'completed':True,'generations':len(outputs),'planned':len(schedule),
            'generated_tokens':sum(x['output_tokens'] for x in outputs),'elapsed_seconds':elapsed(),'preflight_checks':checks,
            'parameter_version_checks':all(x['parameter_version_unchanged'] for x in outputs)})
        save_progress('complete')
        print('COMPLETE '+folder.name,flush=True)
    except BaseException as error:
        lab.write_json(folder/'interruption.json',{'type':type(error).__name__,'reason':str(error),
            'elapsed_seconds':elapsed(),'active_trial':active,'stage':guard.stage if guard else None,
            'completed':len(outputs),'planned':len(schedule),'last_resources':guard.last if guard else None})
        save_progress('interrupted')
        if isinstance(error,ResourceLimitError):
            print('RESOURCE STOP '+str(error),flush=True)
            exit_code=2
        else:raise
    finally:
        if guard:guard.close()
        # Local loop references can retain the model even after the manager clears it.
        if 'model' in locals():del model
        if 'tokenizer' in locals():del tokenizer
        manager.release()
        if cfg.get('resources'):
            try:
                cleanup={'resources':snapshot(folder,selected),'model_released':manager.model is None,'elapsed_seconds':elapsed()}
            except Exception as error:
                cleanup={'telemetry_error':str(error),'model_released':manager.model is None,'elapsed_seconds':elapsed()}
            lab.write_json(folder/'cleanup.json',cleanup)
    if exit_code:raise SystemExit(exit_code)


if __name__=='__main__':main()
