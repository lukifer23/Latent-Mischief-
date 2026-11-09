"""Reversible sparse offsets gated by explicit model forward-call phase."""
from contextlib import contextmanager
import math


PHASES = ('prefill', 'decode', 'all')


@contextmanager
def timed_sparse_hook(model, layer, channels, offsets, phase):
    """One unchunked prefill call, then cached decode calls; reset per context."""
    import torch
    if phase not in PHASES:
        raise ValueError('Choose prefill, decode or all')
    if type(layer) is not int or not 0 <= layer < len(model.model.layers):
        raise ValueError('Layer outside checkpoint')
    module = model.model.layers[layer].feed_forward.w2
    if not channels or len(channels) != len(offsets) or len(set(channels)) != len(channels):
        raise ValueError('Distinct channels and matching offsets required')
    if any(type(x) is not int or not 0 <= x < module.in_features for x in channels):
        raise ValueError('Channel outside checkpoint')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in offsets):
        raise ValueError('Finite offsets required')
    indices = torch.tensor(channels, device=model.device)
    delta = torch.tensor(offsets, device=model.device, dtype=model.dtype)
    trace = {'prefill_forwards': 0, 'decode_forwards': 0,
             'prefill_interventions': 0, 'decode_interventions': 0, 'intervened_positions': 0}
    state = {'forward': 0, 'phase': None, 'w2_calls': 0}

    def begin(_model, args, kwargs):
        state['forward'] += 1
        state['phase'] = 'prefill' if state['forward'] == 1 else 'decode'
        state['w2_calls'] = 0
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Timing protocol requires one sequence of token IDs')
        if state['phase'] == 'decode' and ids.shape[1] != 1:
            raise ValueError('Decode phase requires one new cached token; chunked/full-prefix decode is unsupported')
        trace[state['phase'] + '_forwards'] += 1

    def push(_module, args):
        state['w2_calls'] += 1
        if state['w2_calls'] != 1:
            raise ValueError('Expected one target MLP invocation per model forward')
        if phase != 'all' and phase != state['phase']:
            return None
        value = args[0].clone()
        value[..., indices] += delta
        trace[state['phase'] + '_interventions'] += 1
        trace['intervened_positions'] += value.shape[0] * value.shape[1]
        return (value,) + args[1:]

    handles = []
    try:
        handles.append(model.register_forward_pre_hook(begin, with_kwargs=True))
        handles.append(module.register_forward_pre_hook(push))
        yield trace
    finally:
        for handle in reversed(handles):
            handle.remove()
