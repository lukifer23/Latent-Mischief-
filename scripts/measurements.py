"""Inspectable surface measurements; not neuron-function or persona-quality judgments."""
import json
import re


def repeated_fourgrams(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def measure(result, prompt):
    text = result['text']
    phrases = [r'let me check', r'let me count', r'that[\u2019\x27]s three sentences',
               r'exactly three sentences', r'sentence\s+1\s+(?:ends|starts)']
    hits = [pattern for pattern in phrases if re.search(pattern, text, re.I)]
    metrics = {'repeated_fourgram_fraction': repeated_fourgrams(result['token_ids']),
               'reasoning_boundary_present': '<think>' in text or '</think>' in text,
               'self_check_phrase_present': bool(hits), 'self_check_patterns_matched': hits}
    if 'check' in prompt:
        check = prompt['check']
        if check['kind'] != 'exact-json':
            raise ValueError('Unknown deterministic prompt check')
        plain = text.replace('<|im_end|>', '').strip()
        try:
            metrics['exact_json_match'] = json.loads(plain) == check['expected']
        except json.JSONDecodeError:
            metrics['exact_json_match'] = False
    return metrics
