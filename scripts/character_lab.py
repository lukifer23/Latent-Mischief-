"""Training-free sparse character discovery with measured-strength random controls."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import lab
from calibrated_map import capture_prefill
from calibration import calibrate_offset
from measurements import measure
from personas import indicators


@contextmanager
def sparse_hook(model, layer, channels, offsets):
    import torch
    module = model.model.layers[layer].feed_forward.w2
    if not channels or len(channels) != len(offsets) or len(set(channels)) != len(channels):
        raise ValueError('Distinct channels and matching offsets required')
    if any(type(x) is not int or not 0 <= x < module.in_features for x in channels) or any(not math.isfinite(x) for x in offsets):
        raise ValueError('Invalid sparse offsets')
    indices = torch.tensor(channels, device=model.device)
    delta = torch.tensor(offsets, device=model.device, dtype=model.dtype)
    def push(_module, args):
        changed = args[0].clone()
        changed[..., indices] += delta
        return (changed,) + args[1:]
    handle = module.register_forward_pre_hook(push)
    try:
        yield
    finally:
        handle.remove()


def rank_profile(paired, rms, count, consistency):
    import torch
    mean = paired.mean(0)
    agreement = ((paired.sign() == mean.sign()).float()).mean(0)
    score = mean.abs() / rms.clamp_min(1e-6)
    eligible = (agreement >= consistency) & (rms >= 1e-5) & (mean.abs() > 1e-8)
    order = sorted((i for i in range(len(mean)) if eligible[i]), key=lambda i: (-score[i].item(), i))
    if len(order) < count:
        raise ValueError('Insufficient consistent nonzero channels; do not silently relax extraction')
    channels = order[:count]
    weights = mean[channels] / rms[channels].clamp_min(1e-6)
    weights /= weights.norm()
    native = weights * rms[channels]
    ranked = [{'neuron': i, 'standardized_difference': (mean[i]/rms[i]).item(),
               'pair_consistency': agreement[i].item(), 'neutral_rms': rms[i].item()} for i in order[:32]]
    return channels, native, ranked


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=lab.ROOT / 'configs/characters.json')
    args = parser.parse_args()
    cfg_bytes = args.config.read_bytes()
    cfg = json.loads(cfg_bytes)
    discovery = json.loads((lab.ROOT / cfg['discovery_prompts']).read_text(encoding='utf-8'))
    probes = json.loads((lab.ROOT / cfg['evaluation_prompts']).read_text(encoding='utf-8'))
    if {x['prompt'] for x in discovery} & {x['prompt'] for x in probes}:
        raise ValueError('Discovery/evaluation prompts overlap')
    folder = lab.ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-CHAR-001-sparse-characters')
    folder.mkdir()
    source_hashes = lab.archive_sources(folder, ['scripts/character_lab.py', 'scripts/calibration.py',
        'scripts/calibrated_map.py', 'scripts/lab.py', 'scripts/storage.py', 'scripts/measurements.py',
        'scripts/personas.py', 'uv.lock', 'configs/model.json', cfg['discovery_prompts'], cfg['evaluation_prompts']])
    (folder / 'source/experiment-config.json').write_bytes(cfg_bytes)
    model, tokenizer = lab.load()
    versions = {name: p._version for name, p in model.named_parameters()}
    lab.write_json(folder / 'metadata.json', {'experiment': 'CHAR-001', 'model': lab.config(),
        'environment': lab.environment(), 'experiment_config': cfg, 'source_hashes': source_hashes,
        'experiment_config_sha256': hashlib.sha256(cfg_bytes).hexdigest(),
        'checkpoint_generation_defaults': model.generation_config.to_dict(),
        'hook': 'sparse constant offsets at gated MLP w2 input, all prompt/decode positions',
        'feature_positions': 'non-special teacher-forced reply tokens, equal weight per response',
        'calibration_positions': 'neutral prompt prefills, non-special except position zero; includes system',
        'limits': 'Lexical/content/length confounds; exploratory character indicators are not personality scores.'})
    print(f'RUN {folder}', flush=True)
    special = set(tokenizer.all_special_ids)
    def prepare(item):
        return tokenizer.apply_chat_template([{'role':'system','content':cfg['system']},
            {'role':'user','content':item['prompt']}], tokenize=True, add_generation_prompt=True,
            enable_thinking=lab.config()['thinking'], return_tensors='pt', return_dict=True).to(model.device)
    features = {style: [] for style in ['neutral', *cfg['personas']]}
    squares, feature_inputs = [], []
    with torch.inference_mode():
        for item in discovery:
            prepared = prepare(item)
            for style in features:
                reply = tokenizer.encode(item[style], add_special_tokens=False)
                prefix = prepared['input_ids'][0].tolist()
                positions = torch.tensor([len(prefix)+i for i, token in enumerate(reply) if token not in special], device=model.device)
                values, squared, handles = {}, {}, []
                def hook(layer):
                    def save(_module, args):
                        x = args[0][0, positions].detach().float()
                        values[layer], squared[layer] = x.mean(0).cpu(), x.square().mean(0).cpu()
                    return save
                try:
                    for index, block in enumerate(model.model.layers):
                        handles.append(block.feed_forward.w2.register_forward_pre_hook(hook(index)))
                    ids = torch.tensor([prefix+reply], device=model.device)
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
                finally:
                    for handle in handles: handle.remove()
                features[style].append(values)
                if style == 'neutral': squares.append(squared)
                feature_inputs.append({'prompt_id':item['id'], 'style':style,'input_ids':prefix,'reply_ids':reply})
            print(f'FEATURE {item["id"]}', flush=True)
    lab.write_json(folder / 'feature-inputs.json', feature_inputs)
    torch.save({'features':features,'neutral_response_squares':squares}, folder / 'features.pt')
    base, rankings = [], {}
    for layer in range(len(model.model.layers)):
        rms = torch.stack([x[layer] for x in squares]).mean(0).sqrt()
        for persona in cfg['personas']:
            paired = torch.stack([p[layer]-n[layer] for p,n in zip(features[persona],features['neutral'])])
            channels, native, ranked = rank_profile(paired,rms,cfg['channels_per_profile'],cfg['minimum_pair_consistency'])
            rankings[f'{persona}:{layer}'] = ranked
            if layer in cfg['layers']:
                base.append({'candidate':f'{persona}-L{layer:02d}', 'persona':persona,'role':'style-profile',
                    'layer':layer,'channels':channels,'native':native.tolist(),'sign':1})
    chooser = torch.Generator().manual_seed(cfg['seed'])
    for layer in cfg['layers']:
        excluded = {channel for p in base if p['layer']==layer for channel in p['channels']}
        options = [i for i in range(model.model.layers[layer].feed_forward.w2.in_features) if i not in excluded]
        order = torch.randperm(len(options),generator=chooser)[:cfg['channels_per_profile']].tolist()
        channels = [options[i] for i in order]
        weights = torch.randn(len(channels),generator=chooser); weights /= weights.norm()
        rms = torch.stack([x[layer] for x in squares]).mean(0).sqrt()
        base.append({'candidate':f'random-L{layer:02d}','persona':None,'role':'sparse-random',
            'layer':layer,'channels':channels,'native':(weights*rms[channels]).tolist(),'sign':1})
    reversed_profile = next(x for x in base if x['persona']=='villain' and x['layer']==cfg['negative_villain_layer'])
    base.append({**reversed_profile,'candidate':reversed_profile['candidate']+'-reversed',
                 'role':'reversed-profile','native':[-x for x in reversed_profile['native']],'sign':-1})
    lab.write_json(folder / 'rankings.json', rankings)
    raw, reference, prepared, masks = {}, {}, {}, {}
    with torch.inference_mode():
        for item in discovery:
            key = item['id']; prepared[key] = prepare(item)
            masks[key] = torch.tensor([i>0 and token not in special for i,token in enumerate(prepared[key]['input_ids'][0].tolist())], device=model.device)
            handles, captured = [], {}
            def save(layer, name):
                def capture(_module, _args, value): captured[layer,name] = value[0,masks[key]].detach().float().clone()
                return capture
            try:
                for layer in cfg['layers']:
                    block = model.model.layers[layer]
                    handles.append(block.feed_forward.w2.register_forward_hook(save(layer,'raw')))
                    handles.append(block.post_ffn_norm.register_forward_hook(save(layer,'post')))
                model(**prepared[key],use_cache=False,logits_to_keep=1)
            finally:
                for handle in handles: handle.remove()
            for layer in cfg['layers']:
                raw.setdefault(layer,[]).append(captured[layer,'raw'])
                reference[layer,key] = captured[layer,'post']
    profiles = []
    with torch.inference_mode():
        for profile in base:
            layer = profile['layer']; block = model.model.layers[layer]
            native = torch.tensor(profile['native'],device=model.device)
            direction = block.feed_forward.w2.weight[:,profile['channels']].float() @ native
            targets = [max(cfg['targets'])] if profile['role']=='reversed-profile' else cfg['targets']
            for target in targets:
                calibrated = calibrate_offset(torch.cat(raw[layer]),direction,block.post_ffn_norm.weight.float(),
                    block.post_ffn_norm.variance_epsilon,1,target,1,cfg['max_rms_multiplier'])
                offsets = (native * calibrated['offset']).tolist()
                actual = []
                for item in discovery[:4]:
                    key = item['id']; captured = {}
                    def capture(_module,_args,value): captured['post'] = value[0,masks[key]].detach().float()
                    handle = block.post_ffn_norm.register_forward_hook(capture)
                    try:
                        with sparse_hook(model,layer,profile['channels'],offsets):
                            model(**prepared[key],use_cache=False,logits_to_keep=1)
                    finally: handle.remove()
                    ref = reference[layer,key]
                    actual.append(((captured['post']-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).mean().item())
                profiles.append({**profile,**calibrated,'target_fraction':target,'offsets':offsets,
                    'profile_id':f'{profile["candidate"]}@{target}', 'actual_screen_fractions':actual,
                    'mean_actual_screen_fraction':sum(actual)/len(actual)})
                lab.write_json(folder / 'profiles.json',profiles)
                print(f'CALIBRATE {profile["candidate"]} {target} cap={calibrated["hit_cap"]}',flush=True)
    outputs, baseline_ids, baseline_prefill = [], {}, {}
    def emit(result,item,**labels):
        result.update(phase='probe',prompt_id=item['id'],cohort=item['cohort'],domain=item['domain'],**labels)
        result['measurements'],result['indicators'] = measure(result,item),indicators(result)
        outputs.append(result); lab.write_json(folder / 'outputs.json',outputs)
        print(f'REPLY {len(outputs)}/170 {labels["condition"]} {item["id"]}',flush=True)
    def generate(item,system=None):
        return lab.generate(model,tokenizer,item['prompt'],system=system or cfg['system'],max_new_tokens=cfg['max_new_tokens'])
    for item in probes:
        with capture_prefill(model,cfg['layers']) as captured: result = generate(item)
        baseline_ids[item['id']],baseline_prefill[item['id']] = result['token_ids'][:],captured
        emit(result,item,condition='baseline')
    first = profiles[0]
    with sparse_hook(model,first['layer'],first['channels'],[0.]*len(first['channels'])): zero = generate(probes[0])
    restored = generate(probes[0])
    checks = {'zero_offset_token_identical':zero['token_ids']==baseline_ids[probes[0]['id']],
              'removed_hook_token_identical':restored['token_ids']==baseline_ids[probes[0]['id']]}
    for name,result in [('zero-offset',zero),('removed-hook',restored)]:
        emit(result,probes[0],condition=name); outputs[-1]['phase']='check'
    lab.write_json(folder / 'outputs.json',outputs); lab.write_json(folder / 'checks.json',checks)
    if not all(checks.values()): raise RuntimeError('Sparse hook check failed')
    schedule = [{'condition':'prompt-persona','persona':persona,'prompt_id':item['id']} for persona in cfg['personas'] for item in probes]
    schedule += [{'condition':'sparse-profile','profile_id':p['profile_id'],'prompt_id':item['id']} for p in profiles for item in probes]
    random.Random(cfg['seed']+1).shuffle(schedule)
    lab.write_json(folder / 'trial-order.json',schedule)
    lookup = {p['profile_id']:p for p in profiles}; prompt_lookup = {p['id']:p for p in probes}
    for trial in schedule:
        item = prompt_lookup[trial['prompt_id']]
        if trial['condition']=='prompt-persona':
            persona = trial['persona']
            emit(generate(item,cfg['system']+' '+cfg['personas'][persona]),item,condition='prompt-persona',persona=persona,role='prompt-control')
        else:
            profile = lookup[trial['profile_id']]
            with capture_prefill(model,[profile['layer']]) as captured:
                with sparse_hook(model,profile['layer'],profile['channels'],profile['offsets']): result = generate(item)
            mask = torch.tensor([i>0 and token not in special for i,token in enumerate(result['input_token_ids'])])
            ref = baseline_prefill[item['id']][profile['layer']][mask]
            actual = ((captured[profile['layer']][mask]-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).mean().item()
            emit(result,item,condition='sparse-profile',**{k:profile[k] for k in ['profile_id','candidate','persona','role','layer','sign','target_fraction','hit_cap']},
                actual_prefill_fraction=actual,token_identical_to_baseline=result['token_ids']==baseline_ids[item['id']])
    checks['parameter_version_counters_unchanged'] = versions=={name:p._version for name,p in model.named_parameters()}
    lab.write_json(folder / 'checks.json',checks)
    lab.write_json(folder / 'summary.json',{'completed':all(checks.values()),'checks':checks,'generations':len(outputs),
        'scheduled_trials':len(schedule),'profiles':len(profiles), 'capped_profiles':sum(x['hit_cap'] for x in profiles),
        'profiles_sha256':hashlib.sha256((folder / 'profiles.json').read_bytes()).hexdigest(),
        'note':'Discovery, not personality validation. Compare ordinary prompting and random controls.'})
    if not all(checks.values()): raise RuntimeError('Parameter versions changed')
    print(f'COMPLETE {folder}',flush=True)


if __name__=='__main__': main()
