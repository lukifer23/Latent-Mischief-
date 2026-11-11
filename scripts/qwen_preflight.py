"""Loaded tensor integrity and bounded teacher-forced prefill checks for Qwen."""
import hashlib


def loaded_digests(model, audit, condition):
    import torch
    found = {}
    for entry in audit['changed_tensors']:
        value = model.get_parameter(entry['tensor']).detach().to(device='cpu', dtype=torch.bfloat16)
        actual = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        expected = entry['base_sha256' if condition == 'baseline' else 'edited_sha256']
        if actual != expected:
            raise ValueError('Loaded tensor differs from audited checkpoint: ' + entry['tensor'])
        found[entry['tensor']] = actual
    return found


def bounded_prefill(model, tokenizer, policy, check, emit):
    import torch
    import runtime
    selected = runtime.for_model(model)
    source = tokenizer.encode('A notebook records careful observations about trees, trains and quiet libraries. ', add_special_tokens=False)
    records = []
    for length in policy['preflight_input_tokens']:
        check()
        tokens = (source * ((length + len(source) - 1) // len(source)))[:length]
        ids = torch.tensor([tokens], device=model.device)
        with torch.inference_mode():
            direct = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            cache = None
            for start in range(0, length, policy['preflight_chunk_tokens']):
                check()
                end = min(start + policy['preflight_chunk_tokens'], length)
                # Batch of one without padding: omit attention_mask. After a generate() call, Qwen3.5's multimodal
                # wrapper derives position ids from a full-length mask and mismatches the chunk (verified on real weights).
                out = model(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
            cached = out.logits[0, -1].float()
            record = {'input_tokens': length, 'input_token_ids': tokens,
                      'finite': bool(torch.isfinite(direct).all() and torch.isfinite(cached).all()),
                      'argmax_identical': direct.argmax().item() == cached.argmax().item(),
                      'max_abs_logit_difference': float((direct - cached).abs().max())}
        records.append(record)
        emit(records)
        if not record['finite'] or not record['argmax_identical']:
            raise ValueError('Intermediate-length cached prefill check failed')
        del ids, direct, cached, out, cache
        selected.empty_cache()
        check()
    return records
