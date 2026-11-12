"""Reversible residual-stream addition at every position, capture, and teacher-forced likelihood."""
from contextlib import contextmanager


def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def residual_hook(model, layer, vector, scale):
    """Add scale*vector to the block output (the residual stream) at all positions of every forward.

    During cached generation each position is seen exactly once (prompt positions in
    prefill, each new token in decode), so this equals one full-sequence forward.
    """
    import torch
    if type(layer) is not int or not 0 <= layer < len(model.model.layers):
        raise ValueError('Layer outside checkpoint')
    if vector.ndim != 1 or not torch.isfinite(vector).all() or not torch.isfinite(torch.tensor(float(scale))):
        raise ValueError('Finite one-dimensional vector and scale required')
    delta = (vector.float() * float(scale)).to(device=model.device, dtype=model.dtype)
    trace = {'forwards': 0, 'positions': 0}

    def push(_module, _inputs, output):
        value = _tensor(output)
        if value.shape[-1] != delta.shape[0]:
            raise ValueError('Vector width differs from the residual stream')
        trace['forwards'] += 1
        trace['positions'] += value.shape[0] * value.shape[1]
        changed = value + delta
        return (changed,) + tuple(output[1:]) if isinstance(output, tuple) else changed

    handle = model.model.layers[layer].register_forward_hook(push)
    try:
        yield trace
    finally:
        handle.remove()


def reply_means(model, ids, positions):
    """Mean block output over the chosen positions, float32 on CPU, for every layer."""
    import torch
    means, handles = {}, []
    index = torch.tensor(positions, device=model.device)

    def make(layer):
        def capture(_module, _inputs, output):
            means[layer] = _tensor(output)[0].index_select(0, index).float().mean(dim=0).cpu()
        return capture

    try:
        for i, block in enumerate(model.model.layers):
            handles.append(block.register_forward_hook(make(i)))
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(means) != len(model.model.layers):
        raise RuntimeError('Not every layer was captured')
    return torch.stack([means[i] for i in range(len(model.model.layers))])


def forced_nll(model, context_ids, reply_ids):
    """Summed negative log-likelihood (nats) of reply_ids after context_ids; one uncached forward."""
    import torch
    if not context_ids or not reply_ids:
        raise ValueError('Context and reply tokens required')
    ids = torch.tensor([context_ids + reply_ids], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    start = len(context_ids)
    scores = torch.log_softmax(logits[0, start - 1:-1].float(), dim=-1)
    picked = scores.gather(1, ids[0, start:, None])[:, 0]
    total = -picked.sum().item()
    if total != total or total in (float('inf'), float('-inf')):
        raise RuntimeError('Non-finite likelihood')
    return {'nll_sum': total, 'tokens': len(reply_ids)}
