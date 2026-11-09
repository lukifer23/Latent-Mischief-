"""One local GPU worker: model-aware, saved, paired default/edited conversations."""
from contextlib import ExitStack
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import lab
import runtime
from calibration import offset_hook
from character_lab import sparse_hook
from personas import direction_hook


def read(path): return json.loads(path.read_text(encoding='utf-8'))


class Models:
    def __init__(self): self.loaded=None; self.model=None; self.tokenizer=None
    def release(self):
        selected=runtime.for_model(self.model) if self.model is not None else None
        self.loaded=None; self.model=None; self.tokenizer=None; gc.collect()
        if selected:selected.empty_cache()
    def load(self,spec,attention):
        import torch
        selected=runtime.select()
        key=(spec['model_id'],spec['revision'],attention,selected.device,selected.dtype_name)
        if key==self.loaded: return self.model,self.tokenizer,0.
        previous=runtime.for_model(self.model) if self.model is not None else selected
        self.model=None; self.tokenizer=None; gc.collect(); previous.empty_cache()
        from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForMultimodalLM
        start=time.perf_counter()
        if spec['family']=='lfm3':
            # LFM3 currently ships pinned custom modeling code already inspected in this lab.
            tokenizer=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['revision'],trust_remote_code=True)
            model=AutoModelForCausalLM.from_pretrained(spec['model_id'],revision=spec['revision'],code_revision=spec['revision'],
                trust_remote_code=True,dtype=selected.dtype,attn_implementation=attention).to(selected.device).eval()
        else:
            tokenizer=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['revision'],trust_remote_code=False)
            model=AutoModelForMultimodalLM.from_pretrained(spec['model_id'],revision=spec['revision'],
                trust_remote_code=False,dtype=selected.dtype,attn_implementation=attention).to(selected.device).eval()
        actual=getattr(model.config,'model_type',None)
        if actual!=spec['family']: raise ValueError(f'Loaded model type {actual} differs from selected family {spec["family"]}')
        selected.synchronize(); self.loaded=key; self.model,self.tokenizer=model,tokenizer
        return model,tokenizer,time.perf_counter()-start


def hook_context(model,recipe,entry):
    import ast
    import inspect
    import torch
    if str(model.dtype).removeprefix('torch.')!=recipe['model']['dtype']:
        raise ValueError('Selected dtype differs from the exact saved recipe; use its recorded dtype')
    function=lab.gain_hook if recipe['family']=='channel-gain' else offset_hook if recipe['family']=='channel-offset' else direction_hook if recipe['family'] in ['contrast-direction','random-direction'] else sparse_hook
    node=ast.parse(inspect.getsource(function)).body[0]
    if hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()!=recipe['hook_function_ast_sha256']:
        raise ValueError('Current hook implementation differs from the saved recipe')
    stack=ExitStack()
    try:
        for op in recipe.get('operations',[]):
            if op['kind']=='gain': stack.enter_context(lab.gain_hook(model,op['layer'],op['channels'][0],op['gain']))
            elif op['kind']=='offset':
                if recipe['family']=='channel-offset': stack.enter_context(offset_hook(model,op['layer'],op['channels'][0],op['offsets'][0]))
                else: stack.enter_context(sparse_hook(model,op['layer'],op['channels'],op['offsets']))
            elif op['kind']=='residual-vector':
                reference=next(m['vector_artifact'] for m in entry['methods'] if m.get('vector_artifact'))
                path=lab.ROOT/'runs'/reference['run']/reference['file']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['file_sha256_at_catalog_build']: raise ValueError('Historical vector artifact differs')
                data=torch.load(path,map_location='cpu',weights_only=True)
                if recipe['family']=='contrast-direction':
                    persona,layer=reference['key'].split(':'); vector=data['directions'][persona][int(layer)]
                else: vector=data[reference['key']]
                if hashlib.sha256(vector.float().contiguous().numpy().tobytes()).hexdigest()!=op['vector_sha256']: raise ValueError('Vector differs from exact recipe')
                stack.enter_context(direction_hook(model,op['layer'],vector,op['amplitude']))
            else: raise ValueError('Unsupported exact operation')
    except BaseException: stack.close(); raise
    return stack


def generate(model,tokenizer,messages,request,progress):
    import torch
    import psutil
    inputs=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=request['thinking'],return_tensors='pt',return_dict=True).to(model.device)
    ids=inputs['input_ids']; length=ids.shape[1]
    config=getattr(model.config,'text_config',model.config)
    maximum=getattr(config,'max_position_embeddings',0)
    if maximum and length+request['max_new_tokens']>maximum: raise ValueError('Actual tokenized context plus output exceeds selected model window')
    selected=runtime.for_model(model)
    torch.manual_seed(request['seed']); selected.reset_peak_memory(); selected.synchronize(); start=time.perf_counter()
    cache=None
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    if tokenizer.eos_token_id is not None: eos.add(tokenizer.eos_token_id)
    with torch.inference_mode():
        for begin in range(0,length,256):
            end=min(begin+256,length)
            out=model(input_ids=ids[:,begin:end],attention_mask=torch.ones((1,end),device=model.device,dtype=torch.long),
                past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values
        selected.synchronize(); prefill=time.perf_counter()-start
        generated=[out.logits[0,-1].argmax().item()]; selected.synchronize(); first=time.perf_counter()
        for step in range(1,request['max_new_tokens']):
            if generated[-1] in eos: break
            out=model(input_ids=torch.tensor([[generated[-1]]],device=model.device),past_key_values=cache,use_cache=True,
                attention_mask=torch.ones((1,length+step),device=model.device,dtype=torch.long),logits_to_keep=1)
            cache=out.past_key_values; generated.append(out.logits[0,-1].argmax().item())
            if step%16==0: progress(tokenizer.decode(generated,skip_special_tokens=False))
        selected.synchronize(); end=time.perf_counter()
    return {'text':tokenizer.decode(generated,skip_special_tokens=False),'token_ids':generated,'input_token_ids':ids[0].tolist(),
        'input_tokens':length,'output_tokens':len(generated),'seconds':end-start,'prefill_seconds':prefill,
        'time_to_first_token_seconds':first-start,'decode_tokens_after_first':len(generated)-1,
        'decode_tokens_per_second':(len(generated)-1)/(end-first) if len(generated)>1 else None,
        'runtime':selected.identity(),**selected.memory(),
        'process_rss_end_bytes':psutil.Process().memory_info().rss,'terminated_with_eos':generated[-1] in eos,
        'hit_token_limit':len(generated)==request['max_new_tokens'],'messages':messages}


def process(folder,manager):
    request=read(folder/'request.json'); models=read(lab.ROOT/'configs/models.json'); spec=models[request['model_key']]
    catalog=read(lab.ROOT/'runs/persona-catalog.json') if (lab.ROOT/'runs/persona-catalog.json').exists() else read(lab.ROOT/'experiments/personas/registry.json')
    recipe_id=request['recipe_id']; entry=next((e for e in catalog['recipes'] if e['id']==recipe_id),None)
    published=entry is not None and entry['recipe']['family']=='published-checkpoint'
    if published and request['model_key']!='qwen08': raise ValueError('Qwen checkpoint edit cannot be applied to LFM3')
    if entry and entry['recipe']['model']['model_id']!=spec['model_id']: raise ValueError('Recipe model does not match selected model')
    if not published and recipe_id!='default' and entry is None: raise ValueError('Unknown recipe')
    if entry and not published and spec['family']!='lfm3': raise ValueError('This hook adapter is LFM3-specific')
    if entry and runtime.select().dtype_name!=entry['recipe']['model']['dtype']:
        raise ValueError('Selected dtype differs from the exact saved recipe; use its recorded dtype')
    attention=entry['recipe']['model']['attention'] if entry else spec['live_attention']
    state={'status':'running','model_key':request['model_key'],'recipe_id':recipe_id,'stage':'loading','branches':{}}
    lab.write_json(folder/'job.json',state)
    outputs=[]; source_hashes=lab.archive_sources(folder,['scripts/live_worker.py','scripts/lab.py','scripts/calibration.py',
        'scripts/character_lab.py','scripts/personas.py','scripts/storage.py','scripts/measurements.py','scripts/catalog.py','configs/models.json','uv.lock'])
    lab.write_json(folder/'metadata.json',{'experiment':'LIVE','model_key':request['model_key'],'model':spec,'runtime_attention':attention,
        'recipe_id':recipe_id,'recipe':entry['recipe'] if entry else spec.get('published_edit') if published else None,
        'request':request,'source_hashes':source_hashes,'environment':lab.environment(),
        'scope':'Interactive paired branch conversations; greedy manual decode; no benchmark equivalence claim.'})
    for branch in ['baseline','edited']:
        branch_spec={**spec,**spec['published_edit']} if branch=='edited' and published else spec
        state.update(stage='loading '+branch); lab.write_json(folder/'job.json',state)
        model,tokenizer,load_seconds=manager.load(branch_spec,attention)
        version={n:p._version for n,p in model.named_parameters()}
        history=request['history_default' if branch=='baseline' else 'history_edited']
        system=request['system']
        if branch=='edited' and entry and entry['recipe']['family']=='prompt-persona': system=entry['recipe']['system']
        messages=([{'role':'system','content':system}] if system else [])+history+[{'role':'user','content':request['prompt']}]
        def progress(text):
            state.update(stage='generating '+branch); state['branches'][branch]={'text':text,'partial':True}; lab.write_json(folder/'job.json',state)
        context=hook_context(model,entry['recipe'],entry) if branch=='edited' and entry and entry['recipe']['family'] not in ['prompt-persona','published-checkpoint'] else ExitStack()
        with context: result=generate(model,tokenizer,messages,request,progress)
        if version!={n:p._version for n,p in model.named_parameters()}: raise RuntimeError('Interactive hook mutated parameters')
        result.update(condition=branch,phase='probe',prompt_id='live',prompt=request['prompt'],system=system,
            model_id=branch_spec['model_id'],revision=branch_spec['revision'],load_seconds=load_seconds)
        from measurements import measure
        result['measurements']=measure(result,{})
        outputs.append(result); lab.write_json(folder/'outputs.json',outputs)
        state['branches'][branch]=result; lab.write_json(folder/'job.json',state)
        del model,tokenizer
    lab.write_json(folder/'summary.json',{'completed':True,'generations':2,'recipe_id':recipe_id,'model_key':request['model_key'],
        'parameter_version_checks':True,'same_prompt':True,'history_note':'Each branch retains its own saved conversation; messages are recorded separately.'})
    try:
        from catalog import build
        lab.write_json(lab.ROOT/'runs/persona-catalog.json',build())
    except Exception as error:
        state['catalog_warning']=str(error)
    state.update(status='complete',stage='complete'); lab.write_json(folder/'job.json',state)


def main():
    queue=lab.ROOT/'runs'
    manager=Models(); last=time.monotonic()
    while time.monotonic()-last<90:
        pending=[p for p in sorted(queue.glob('*-LIVE-*')) if p.is_dir() and (p/'request.json').exists() and read(p/'job.json').get('status')=='queued']
        if not pending: time.sleep(.2); continue
        for folder in pending:
            try: process(folder,manager)
            except Exception as error:
                import traceback
                traceback.print_exc()
                previous=read(folder/'job.json')
                lab.write_json(folder/'job.json',{**previous,'status':'failed','error':str(error),'type':type(error).__name__})
                manager=Models(); gc.collect()
            last=time.monotonic()


if __name__=='__main__': main()
