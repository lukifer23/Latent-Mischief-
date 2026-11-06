"""Measure actual long-context prefill/decode/memory; no short-prompt stand-in for 128K."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import lab
from calibration import offset_hook
from verify_model import verify_identity


def main():
    import torch
    import psutil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=lab.ROOT/'configs/performance.json')
    args=parser.parse_args(); cfg_bytes=args.config.read_bytes(); cfg=json.loads(cfg_bytes)
    identity=verify_identity()
    evidence=json.loads((lab.ROOT/cfg['source_evidence']).read_text(encoding='utf-8'))
    unit=next(x for x in evidence['selected'] if x['candidate']==cfg['offset_candidate'] and x['sign']==1 and x['target_fraction']==cfg['offset_target'])
    corpus_paths=['README.md',*[p.relative_to(lab.ROOT).as_posix() for p in sorted((lab.ROOT/'docs').glob('*.md'))]]
    corpus='\n\n'.join((lab.ROOT/p).read_text(encoding='utf-8') for p in corpus_paths)
    folder=lab.ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-PERF-001-long-context'); folder.mkdir()
    sources=lab.archive_sources(folder,['scripts/performance.py','scripts/verify_model.py','scripts/lab.py','scripts/calibration.py',
        'scripts/storage.py','configs/model.json','uv.lock',cfg['source_evidence'],*corpus_paths])
    (folder/'source/experiment-config.json').write_bytes(cfg_bytes); lab.write_json(folder/'identity.json',identity)
    print(f'RUN {folder}',flush=True)
    started=time.perf_counter(); model,tokenizer=lab.load(attention=cfg['attention']); torch.cuda.synchronize(); load_seconds=time.perf_counter()-started
    if model.config.model_type!='lfm3' or type(model).__name__!='Lfm3ForCausalLM': raise RuntimeError('Loaded runtime is not LFM3')
    versions={n:p._version for n,p in model.named_parameters()}
    metadata={'experiment':'PERF-001','model':lab.config(),'environment':lab.environment(),'experiment_config':cfg,
        'source_hashes':sources,'experiment_config_sha256':hashlib.sha256(cfg_bytes).hexdigest(),
        'loaded_class':type(model).__name__,'runtime_attention':model.config._attn_implementation,
        'load_seconds':load_seconds,'weights_identity_checks':identity['checks'],
        'kernel_selection':'PyTorch SDPA automatic; actual kernel not profiled',
        'flash_attention_built':torch.backends.cuda.is_flash_attention_available(),
        'workload':'Repository documents tiled with document separators; synthetic marker at midpoint; actual tokens counted.',
        'scope':cfg['scope']}
    lab.write_json(folder/'metadata.json',metadata)
    def prefill(ids,chunk):
        cache=None; out=None
        for start in range(0,ids.shape[1],chunk):
            end=min(start+chunk,ids.shape[1])
            out=model(input_ids=ids[:,start:end],past_key_values=cache,use_cache=True,
                attention_mask=torch.ones((1,end),device=model.device,dtype=torch.long),logits_to_keep=1)
            cache=out.past_key_values
        return out.logits[0,-1].float(),cache
    with torch.inference_mode():
        short=tokenizer.apply_chat_template([{'role':'user','content':'Describe a green apple in one sentence.'}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)['input_ids'].to(model.device)
        direct=model(input_ids=short,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        chunked,cache=prefill(short,7)
        checks={'short_chunked_first_token_identical':direct.argmax().item()==chunked.argmax().item(),
            'short_chunked_logits_finite':bool(torch.isfinite(chunked).all())}
        diagnostics={'short_max_abs_logit_difference':(direct-chunked).abs().max().item()}
        del direct,chunked,cache
    lab.write_json(folder/'checks.json',checks); lab.write_json(folder/'preflight.json',diagnostics)
    if not all(checks.values()): raise RuntimeError('Chunked prefill preflight failed; no long-context tests attempted')
    print('PREFLIGHT passed',flush=True)
    # Construct exact token budgets. The marker/question are included in the counted input.
    rendered=tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant. Use the supplied documents.'},
        {'role':'user','content':'__LM_PAYLOAD__'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    before,after=rendered.split('__LM_PAYLOAD__'); prefix=tokenizer.encode(before,add_special_tokens=False); suffix=tokenizer.encode(after,add_special_tokens=False)
    filler=tokenizer.encode(corpus+'\n\n[Next project document copy]\n',add_special_tokens=False)
    marker='QUARTZ-LANTERN-5824'
    needle=tokenizer.encode('\n[Benchmark record LM-N0001: marker = '+marker+']\n',add_special_tokens=False)
    question=tokenizer.encode('\nQuestion: First give the marker for LM-N0001, then summarize the discussed lab methods in 80 to 120 words.\n',add_special_tokens=False)
    rows=[]
    for point,length in enumerate(cfg['input_tokens']):
        if length+cfg['max_decode_tokens']>model.config.max_position_embeddings: raise ValueError('Context exceeds configured model limit')
        budget=length-len(prefix)-len(suffix)-len(needle)-len(question)
        if budget<=0: raise ValueError('Context too short for workload')
        tiled=(filler*((budget+len(filler)-1)//len(filler)))[:budget]; middle=len(tiled)//2
        token_ids=prefix+tiled[:middle]+needle+tiled[middle:]+question+suffix
        if len(token_ids)!=length: raise RuntimeError('Token budget construction differs')
        lab.write_json(folder/f'input-{length}.json',{'token_ids':token_ids,'marker':marker,'marker_start_token':len(prefix)+middle})
        conditions=['baseline','offset'] if point%2==0 else ['offset','baseline']
        for condition in conditions:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            start_rss=psutil.Process().memory_info().rss; start=time.perf_counter()
            print(f'START {length} {condition}',flush=True)
            try:
                ids=torch.tensor([token_ids],device=model.device)
                context=offset_hook(model,unit['layer'],unit['neuron'],unit['offset']) if condition=='offset' else nullcontext()
                with torch.inference_mode(),context:
                    torch.cuda.synchronize(); start=time.perf_counter()
                    logits,cache=prefill(ids,cfg['prefill_chunk_tokens']); torch.cuda.synchronize()
                    prefill_end=time.perf_counter(); generated=[logits.argmax().item()]; torch.cuda.synchronize(); first=time.perf_counter()
                    for step in range(1,cfg['max_decode_tokens']):
                        if generated[-1]==tokenizer.eos_token_id: break
                        out=model(input_ids=torch.tensor([[generated[-1]]],device=model.device),past_key_values=cache,use_cache=True,
                            attention_mask=torch.ones((1,length+step),device=model.device,dtype=torch.long),logits_to_keep=1)
                        cache=out.past_key_values; generated.append(out.logits[0,-1].argmax().item())
                    torch.cuda.synchronize(); end=time.perf_counter()
                text=tokenizer.decode(generated,skip_special_tokens=False)
                row={'condition':condition,'requested_input_tokens':length,'actual_input_tokens':len(token_ids),'status':'complete',
                    'prefill_seconds':prefill_end-start,'prefill_tokens_per_second':length/(prefill_end-start),
                    'time_to_first_token_seconds':first-start,'decode_tokens_after_first':len(generated)-1,
                    'decode_seconds_after_first':end-first,'decode_tokens_per_second':(len(generated)-1)/(end-first) if len(generated)>1 else None,
                    'total_seconds':end-start,'generated_tokens':len(generated),'eos':generated[-1]==tokenizer.eos_token_id,
                    'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'process_rss_start_bytes':start_rss,'process_rss_end_bytes':psutil.Process().memory_info().rss,
                    'marker_present_in_output':marker in text,'text':text,'token_ids':generated}
                del ids,logits,cache
                if 'out' in locals(): del out
            except torch.cuda.OutOfMemoryError as error:
                row={'condition':condition,'requested_input_tokens':length,'actual_input_tokens':len(token_ids),'status':'out-of-memory',
                    'elapsed_seconds':time.perf_counter()-start,'error':str(error),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
                lab.write_json(folder/'outputs.json',[*rows,row]); raise
            rows.append(row); lab.write_json(folder/'outputs.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k not in ['text','token_ids']}),flush=True)
    checks['parameter_version_counters_unchanged']=versions=={n:p._version for n,p in model.named_parameters()}
    lab.write_json(folder/'checks.json',checks); lab.write_json(folder/'summary.json',{'completed':all(checks.values()),'checks':checks,
        'measured_points':len(rows),'maximum_measured_input_tokens':max(x['actual_input_tokens'] for x in rows),'load_seconds':load_seconds})
    print(f'COMPLETE {folder}',flush=True)


if __name__=='__main__': main()
