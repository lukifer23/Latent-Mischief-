import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from calibration import calibrate_offset, local_fraction, offset_hook, select_candidates


class CalibrationMathTests(unittest.TestCase):
    def test_signed_targets_and_unreachable_cap(self):
        import torch
        y = torch.tensor([[1., 2., 3.], [2., 1., 3.]])
        direction = torch.tensor([0., 1., 0.])
        weight = torch.tensor([1., .8, 1.2])
        for sign in (-1, 1):
            row = calibrate_offset(y, direction, weight, 1e-5, 1, .1, sign)
            self.assertFalse(row['hit_cap'])
            self.assertEqual(math.copysign(1, row['offset']), sign)
            self.assertAlmostEqual(local_fraction(y, direction, row['offset'], weight, 1e-5), .1, delta=2e-4)
        capped = calibrate_offset(y, direction, weight, 1e-5, 1e-6, .1, 1, max_rms=1)
        self.assertTrue(capped['hit_cap'])
        self.assertLess(capped['estimated_fraction'], .1)

    def test_offset_cleanup_preserves_input_and_parameters(self):
        import torch
        from types import SimpleNamespace
        layer = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad(): layer.weight.copy_(torch.tensor([[1., 2., 0.], [0., 1., 3.]]))
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(feed_forward=SimpleNamespace(w2=layer))]))
        x = torch.tensor([[1., 2., 3.]])
        original, weights = layer(x).detach().clone(), layer.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with offset_hook(model, 0, 1, 2):
                self.assertTrue(torch.equal(layer(x), original + torch.tensor([[4., 2.]])))
                raise RuntimeError('simulated generation failure')
        self.assertTrue(torch.equal(layer(x), original))
        self.assertTrue(torch.equal(x, torch.tensor([[1., 2., 3.]])))
        self.assertTrue(torch.equal(layer.weight, weights))

    def test_selection_requires_both_signs_and_rejects_bad_effects(self):
        rows = []
        for neuron, kl in [(1, .01), (2, .02), (3, .03), (4, .04)]:
            for sign in (-1, 1):
                rows.append({'layer': 0, 'neuron': neuron, 'target_fraction': .2, 'sign': sign,
                             'hit_cap': False, 'mean_actual_fraction': .2, 'mean_kl': kl})
        chosen = select_candidates(rows, [0], [.05, .2], .25, .5)
        self.assertEqual({x['neuron'] for x in chosen}, {2})
        broken = [{**x, 'hit_cap': x['sign'] == -1} for x in rows]
        self.assertEqual(select_candidates(broken, [0], [.2], .25, .5), [])
