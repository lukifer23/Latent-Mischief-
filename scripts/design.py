"""Deterministic candidate selection, independent of model inference."""
import math
import random


def build_candidates(cfg, layer_types, widths):
    rng = random.Random(cfg['seed'])
    candidates, seen = [], set()

    def add(layer, neuron, role, gains):
        if type(layer) is not int or not 0 <= layer < len(layer_types):
            raise ValueError('Layer outside checkpoint')
        if type(neuron) is not int or not 0 <= neuron < widths[layer]:
            raise ValueError('Channel outside checkpoint')
        if (layer, neuron) in seen:
            raise ValueError('Duplicate candidate')
        if not gains or any(type(g) not in (int, float) or not math.isfinite(g) or not -4 <= g <= 4 for g in gains):
            raise ValueError('Gains must be finite and between -4 and 4')
        if len(set(gains)) != len(gains):
            raise ValueError('Duplicate gains')
        seen.add((layer, neuron))
        candidates.append({'layer': layer, 'neuron': neuron, 'operator': layer_types[layer],
                           'candidate': f'L{layer:02d}N{neuron:04d}', 'role': role, 'gains': gains})

    for item in cfg.get('candidates', []):
        add(item['layer'], item['neuron'], item.get('role', 'specified'), item.get('gains', cfg.get('gains')))
    exclusions = {(x['layer'], x['neuron']) for x in cfg.get('exclude_candidates', [])}
    for layer in cfg.get('layers', []):
        if type(layer) is not int or not 0 <= layer < len(widths):
            raise ValueError('Layer outside checkpoint')
        pool = [n for n in range(widths[layer]) if (layer, n) not in exclusions | seen]
        for neuron in rng.sample(pool, cfg['channels_per_layer']):
            add(layer, neuron, 'uniform-sample', cfg['gains'])
    control = cfg.get('random_controls')
    if control:
        layer = control['layer']
        if type(layer) is not int or not 0 <= layer < len(widths):
            raise ValueError('Control layer outside checkpoint')
        pool = [n for n in range(widths[layer]) if (layer, n) not in exclusions | seen]
        for neuron in rng.sample(pool, control['count']):
            add(layer, neuron, 'random-control', control['gains'])
    if not candidates:
        raise ValueError('No candidates selected')
    return candidates
