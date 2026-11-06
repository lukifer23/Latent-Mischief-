"""Local output-effect calibration for reversible single-channel offsets."""
from contextlib import contextmanager
import math


@contextmanager
def offset_hook(model, layer, neuron, offset):
    module = model.model.layers[layer].feed_forward.w2
    if not 0 <= neuron < module.in_features or not math.isfinite(offset):
        raise ValueError('Invalid channel or offset')
    def push(_module, args):
        value = args[0].clone()
        value[..., neuron] += offset
        return (value,) + args[1:]
    handle = module.register_forward_pre_hook(push)
    try:
        yield
    finally:
        handle.remove()


def rmsnorm_float(y, weight, eps):
    return y * (y.square().mean(dim=-1, keepdim=True) + eps).rsqrt() * weight


def local_fraction(y, direction, offset, weight, eps, reference=None):
    reference = rmsnorm_float(y, weight, eps) if reference is None else reference
    changed = rmsnorm_float(y + offset * direction, weight, eps)
    return ((changed - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)).mean().item()


def calibrate_offset(y, direction, weight, eps, rms, target, sign, max_rms=32.0, iterations=16):
    if not math.isfinite(target) or not 0 < target < 1 or sign not in (-1, 1) or rms <= 0:
        raise ValueError('Invalid calibration target, sign or activation RMS')
    reference = rmsnorm_float(y, weight, eps)
    high, low = max_rms * rms, 0.0
    maximum = local_fraction(y, direction, sign * high, weight, eps, reference)
    if maximum < target:
        return {'offset': sign * high, 'estimated_fraction': maximum, 'hit_cap': True}
    for _ in range(iterations):
        middle = (low + high) / 2
        if local_fraction(y, direction, sign * middle, weight, eps, reference) < target:
            low = middle
        else:
            high = middle
    offset = sign * (low + high) / 2
    return {'offset': offset, 'estimated_fraction': local_fraction(y, direction, offset, weight, eps, reference), 'hit_cap': False}


def select_candidates(rows, layers, targets, tolerance, max_kl):
    """Outcome-independent: use local perturbation and next-token distributions, not generated personas."""
    chosen = []
    for layer in layers:
        for target in sorted(targets, reverse=True):
            pairs = {}
            for row in rows:
                if row['layer'] == layer and row['target_fraction'] == target:
                    pairs.setdefault(row['neuron'], []).append(row)
            eligible = []
            for neuron, pair in pairs.items():
                if {x['sign'] for x in pair} != {-1, 1} or len(pair) != 2:
                    continue
                if all(not x['hit_cap'] and abs(x['mean_actual_fraction'] / target - 1) <= tolerance and x['mean_kl'] <= max_kl for x in pair):
                    eligible.append((sum(x['mean_kl'] for x in pair) / 2, neuron, pair))
            if eligible:
                eligible.sort(key=lambda item: (item[0], item[1]))
                _kl, _neuron, pair = eligible[(len(eligible) - 1) // 2]
                chosen.extend(pair)
                break
    return chosen
