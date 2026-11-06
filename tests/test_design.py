import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from design import build_candidates
from measurements import measure


class DesignTests(unittest.TestCase):
    def test_controls_exclude_target_and_are_reproducible(self):
        cfg = {'seed': 2718, 'candidates': [{'layer': 1, 'neuron': 3, 'gains': [0, 2]}],
               'random_controls': {'layer': 1, 'count': 4, 'gains': [2]}}
        result = build_candidates(cfg, ['conv', 'full_attention'], [10, 10])
        self.assertEqual(result, build_candidates(copy.deepcopy(cfg), ['conv', 'full_attention'], [10, 10]))
        self.assertEqual(len({(x['layer'], x['neuron']) for x in result}), 5)
        self.assertTrue(all(x['layer'] == 1 and x['neuron'] != 3 for x in result[1:]))

    def test_sampling_excludes_prior_candidates(self):
        cfg = {'seed': 1, 'layers': [0], 'channels_per_layer': 4, 'gains': [0, 2],
               'exclude_candidates': [{'layer': 0, 'neuron': 2}]}
        self.assertEqual({x['neuron'] for x in build_candidates(cfg, ['conv'], [5])}, {0, 1, 3, 4})

    def test_invalid_interventions_rejected(self):
        for candidate in [{'layer': -1, 'neuron': 0, 'gains': [2]},
                          {'layer': 0, 'neuron': 10, 'gains': [2]},
                          {'layer': 0, 'neuron': 1, 'gains': [float('nan')]}]:
            with self.assertRaises(ValueError):
                build_candidates({'seed': 1, 'candidates': [candidate]}, ['conv'], [10])

    def test_format_check_does_not_accept_extra_reasoning_text(self):
        prompt = {'check': {'kind': 'exact-json', 'expected': {'colors': ['red', 'blue']}}}
        good = {'text': '{"colors":["red","blue"]}<|im_end|>', 'token_ids': [1, 2, 3]}
        self.assertTrue(measure(good, prompt)['exact_json_match'])
        good['text'] = 'Let me check. </think>{"colors":["red","blue"]}<|im_end|>'
        metrics = measure(good, prompt)
        self.assertFalse(metrics['exact_json_match'])
        self.assertTrue(metrics['reasoning_boundary_present'])
        self.assertTrue(metrics['self_check_phrase_present'])
