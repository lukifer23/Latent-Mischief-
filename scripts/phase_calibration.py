"""Recalibrate frozen sparse directions on small neutral MPS prefills."""
from contextlib import contextmanager
import math
import statistics

from calibration import calibrate_offset
from phase_timing import timed_sparse_hook


@contextmanager
def capture_post(model, layer):
    values = []
    def capture(_module, _args, output):
        if not values:
            values.append(output.detach().float().cpu().clone())
    handle = model.model.layers[layer].post_ffn_norm.register_forward_hook(capture)
    try:
        yield values
    finally:
        handle.remove()


def calibrate_profiles(model, tokenizer, plan, check, persist, folder):
    import torch
    cfg = plan['config']
    block = model.model.layers[cfg['layer']]
    prepared, masks, raw, post, inputs = {}, {}, {}, {}, []
    special = set(tokenizer.all_special_ids)
    with torch.inference_mode():
        for item in plan['calibration']:
            check()
            data = tokenizer.apply_chat_template([{'role': 'system', 'content': cfg['system']},
                {'role': 'user', 'content': item['prompt']}], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors='pt', return_dict=True).to(model.device)
            ids = data['input_ids'][0].tolist()
            if len(ids) > cfg['max_input_tokens']:
                raise ValueError('Calibration input exceeds the frozen input cap')
            mask = torch.tensor([i > 0 and token not in special for i, token in enumerate(ids)], device=model.device)
            if not mask.any():
                raise ValueError('No calibration positions')
            captured = {}
            def save_raw(_module, _args, output):
                captured['raw'] = output[0, mask].detach().float().clone()
            def save_post(_module, _args, output):
                captured['post'] = output[0, mask].detach().float().clone()
            handles = [block.feed_forward.w2.register_forward_hook(save_raw),
                       block.post_ffn_norm.register_forward_hook(save_post)]
            try:
                model(**data, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles: handle.remove()
            prepared[item['id']], masks[item['id']] = data, mask
            raw[item['id']], post[item['id']] = captured['raw'], captured['post']
            inputs.append({'id': item['id'], 'input_token_ids': ids, 'selected_positions': mask.nonzero().flatten().tolist()})
        torch.save({'raw': {k: v.cpu() for k, v in raw.items()}, 'post': {k: v.cpu() for k, v in post.items()},
                    'inputs': inputs}, folder / 'calibration.pt')
        pooled = torch.cat(list(raw.values()))
        profiles = []
        for profile in plan['profiles']:
            check()
            native = torch.tensor(profile['native'], device=model.device, dtype=torch.float32)
            direction = block.feed_forward.w2.weight[:, profile['channels']].detach().float() @ native
            fit = calibrate_offset(pooled, direction, block.post_ffn_norm.weight.detach().float(),
                block.post_ffn_norm.variance_epsilon, 1., cfg['target_fraction'], 1,
                cfg['max_amplitude'], cfg['calibration_iterations'])
            offsets = (native * fit['offset']).tolist()
            screen = []
            for item in plan['calibration']:
                check()
                key = item['id']
                changed = {}
                def capture(_module, _args, output):
                    changed['post'] = output[0, masks[key]].detach().float()
                handle = block.post_ffn_norm.register_forward_hook(capture)
                try:
                    with timed_sparse_hook(model, profile['layer'], profile['channels'], offsets, 'all'):
                        model(**prepared[key], use_cache=False, logits_to_keep=1)
                finally:
                    handle.remove()
                actual = ((changed['post'] - post[key]).norm(dim=-1) / post[key].norm(dim=-1).clamp_min(1e-12)).mean().item()
                screen.append({'prompt_id': key, 'actual_fraction': actual})
            row = {**profile, **fit, 'offsets': offsets, 'target_fraction': cfg['target_fraction'],
                   'screen': screen, 'mean_actual_fraction': statistics.mean(x['actual_fraction'] for x in screen)}
            profiles.append(row)
            persist(profiles)
            if fit['hit_cap'] or not all(math.isfinite(x['actual_fraction']) for x in screen) or abs(row['mean_actual_fraction'] / cfg['target_fraction'] - 1) > cfg['calibration_tolerance_fraction']:
                raise ValueError('Calibrated target/control strength failed; no evaluation generated')
        check()
    return profiles
