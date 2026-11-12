from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from transfer_design import (PERSONAS, gap_closed, generation_verdict, grid_table, load_plan, nominate, select_cell,
                             teacher_forced_verdict)

CONFIG = ROOT / 'configs/lfm-prompt-transfer.json'


def tiny_model(width=4, layers=3):
    import torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(width, width, bias=False)

        def forward(self, x):
            return x + self.linear(x)

    class Output:
        def __init__(self, logits):
            self.logits = logits

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.embed = torch.nn.Embedding(11, width)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block() for _ in range(layers)])
            self.head = torch.nn.Linear(width, 11, bias=False)

        @property
        def device(self):
            return self.embed.weight.device

        @property
        def dtype(self):
            return self.embed.weight.dtype

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            x = self.embed(input_ids)
            for block in self.model.layers:
                x = block(x)
            return Output(self.head(x))

    return Tiny().eval()


class SteeringTests(unittest.TestCase):
    def test_zero_scale_is_identity_and_hook_is_removed_even_on_error(self):
        import torch
        from transfer_steering import forced_nll, residual_hook
        model = tiny_model()
        weights = [p.detach().clone() for p in model.parameters()]
        base = forced_nll(model, [1, 2, 3], [4, 5])
        with residual_hook(model, 1, torch.ones(4), 0.0) as trace:
            zero = forced_nll(model, [1, 2, 3], [4, 5])
        self.assertEqual(base, zero)
        self.assertEqual(trace, {'forwards': 1, 'positions': 5})
        with self.assertRaises(RuntimeError):
            with residual_hook(model, 1, torch.ones(4), 3.0):
                raise RuntimeError('boom')
        self.assertEqual(forced_nll(model, [1, 2, 3], [4, 5]), base)
        self.assertFalse(model.model.layers[1]._forward_hooks)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(weights, model.parameters())))

    def test_addition_changes_every_position_by_the_scaled_vector(self):
        import torch
        from transfer_steering import reply_means, residual_hook
        model = tiny_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        vector = torch.tensor([1., 0., -1., 2.])
        before = reply_means(model, ids, [0, 1, 2, 3])
        captured = {}
        handle = model.model.layers[2].register_forward_pre_hook(lambda m, a: captured.setdefault('x', a[0].detach().clone()))
        with residual_hook(model, 1, vector, 0.5):
            model(ids)
        handle.remove()
        captured_base = {}
        handle = model.model.layers[2].register_forward_pre_hook(lambda m, a: captured_base.setdefault('x', a[0].detach().clone()))
        model(ids)
        handle.remove()
        self.assertTrue(torch.allclose(captured['x'] - captured_base['x'], (vector * .5).expand(1, 4, 4), atol=1e-6))
        self.assertEqual(before.shape, (3, 4))
        self.assertTrue(torch.allclose(before[1], captured_base['x'][0].mean(dim=0), atol=1e-6))

    def test_forced_nll_matches_manual_cross_entropy_and_rejects_bad_input(self):
        import torch
        from transfer_steering import forced_nll, residual_hook
        model = tiny_model()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits = model(ids).logits[0]
        manual = torch.nn.functional.cross_entropy(logits[2:4], ids[0, 3:5], reduction='sum').item()
        value = forced_nll(model, [1, 2, 3], [4, 5])
        self.assertAlmostEqual(value['nll_sum'], manual, places=5)
        self.assertEqual(value['tokens'], 2)
        with self.assertRaises(ValueError):
            forced_nll(model, [1], [])
        with self.assertRaises(ValueError):
            with residual_hook(model, 7, torch.ones(4), 1.0):
                pass
        with self.assertRaises(ValueError):
            with residual_hook(model, 0, torch.ones(3), 1.0):
                model(ids)

    def test_directions_are_unit_length_share_one_scale_and_use_only_extract_prompts(self):
        import torch
        from transfer_experiment import build_vectors
        torch.manual_seed(1)
        ids = ['p' + str(i) for i in range(4)]
        shift = {'villain': torch.tensor([3., 0., 0., 0.]), 'hamster': torch.tensor([0., 1., 0., 0.])}
        features = {}
        for i in ids:
            features[(i, 'neutral')] = torch.randn(2, 4)
            for persona in PERSONAS:
                features[(i, persona)] = features[(i, 'neutral')] + shift[persona]
        unit, scale, diagnostics = build_vectors(features, ids, {'random_seed': 5, 'random_vectors': 3}, 4, 2)
        self.assertEqual(sorted(unit), ['content', 'hamster', 'random-1', 'random-2', 'random-3', 'villain'])
        for value in unit.values():
            self.assertTrue(torch.allclose(value.norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(unit['villain'][0], torch.tensor([1., 0., 0., 0.]), atol=1e-5))
        self.assertTrue(torch.allclose(scale, torch.full((2,), 2.), atol=1e-5))
        self.assertAlmostEqual(diagnostics['cosine']['villain~hamster'][0], 0., places=5)
        again, _, _ = build_vectors(features, ids, {'random_seed': 5, 'random_vectors': 3}, 4, 2)
        self.assertTrue(torch.equal(unit['random-2'], again['random-2']))


def rows_for(cell, values, prompts=('a', 'b')):
    """values[text][vector] -> steered nll per prompt (baseline 10, prompted 2 for persona texts)."""
    layer, alpha = cell
    return [{'layer': layer, 'alpha': alpha, 'vector': v, 'prompt_id': p, 'text': t, 'nll_sum': n, 'tokens': 4}
            for t, by in values.items() for v, n in by.items() for p in prompts]


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.base = {(p, t): {'nll_sum': 10., 'tokens': 4} for p in 'ab' for t in ('villain', 'hamster', 'neutral')}
        self.best = {(p, t): {'nll_sum': 2., 'tokens': 4} for p in 'ab' for t in ('villain', 'hamster')}
        self.vectors = ['villain', 'hamster', 'content']

    def test_gap_closed_scale_and_errors(self):
        rows = rows_for((4, 1), {'villain': {'villain': 6., 'hamster': 10., 'content': 12.}})
        self.assertAlmostEqual(gap_closed(rows, self.base, self.best, 'villain', 'villain')['gap_closed'], .5)
        self.assertEqual(gap_closed(rows, self.base, self.best, 'villain', 'hamster')['gap_closed'], 0.)
        harmed = gap_closed(rows, self.base, self.best, 'villain', 'content')
        self.assertAlmostEqual(harmed['gap_closed'], -.25)
        self.assertEqual(harmed['prompts_improved'], 0)
        with self.assertRaises(ValueError):
            gap_closed(rows + rows, self.base, self.best, 'villain', 'villain')
        with self.assertRaises(ValueError):
            gap_closed(rows, self.base, self.base, 'villain', 'villain')
        rows[0]['tokens'] = 5
        with self.assertRaises(ValueError):
            gap_closed(rows, self.base, self.best, 'villain', 'villain')

    def test_selection_prefers_largest_gap_then_weaker_dose(self):
        make = lambda cell, own: rows_for(cell, {t: {'villain': own, 'hamster': own, 'content': 10.} for t in PERSONAS})
        table = grid_table(make((4, 1), 6.) + make((8, 2), 4.) + make((4, 2), 4.) + make((8, 8), 30.),
                           self.base, self.best, self.vectors)
        chosen = select_cell(table, 'villain')
        self.assertEqual((chosen['layer'], chosen['alpha']), (4, 2))

    def test_verdict_requires_margin_consistency_style_and_no_lost_check(self):
        cfg = {'criteria': {'min_gap_closed': .25, 'min_margin_over_controls': .10, 'min_prompts_improved': 2}}
        strong = grid_table(rows_for((4, 1), {t: {'villain': 4., 'hamster': 9., 'content': 10.} for t in PERSONAS}),
                            self.base, self.best, self.vectors)[0]
        forced = teacher_forced_verdict(strong, 'villain', cfg)
        self.assertTrue(forced['C1_transfer'] and forced['C2_specific'] and forced['C3_consistent'])
        self.assertEqual(forced['strongest_control'], 'hamster')
        unspecific = grid_table(rows_for((4, 1), {t: {'villain': 4., 'hamster': 4.5, 'content': 10.} for t in PERSONAS}),
                                self.base, self.best, self.vectors)[0]
        self.assertFalse(teacher_forced_verdict(unspecific, 'villain', cfg)['C2_specific'])

        def generation(steered_style, steered_answer):
            rows = []
            for condition, style, answer in (('reference-neutral', -1., True), ('reference-villain', 2., True),
                                             ('steer-villain', steered_style, steered_answer), ('steer-villain-random', -.9, True)):
                for i in range(8):
                    rows.append({'condition': condition, 'prompt_id': 't' + str(i), 'split': 'test',
                                 'style_llr_per_token': {'villain': style}, 'task_score': {'answer_match': None}})
                rows.append({'condition': condition, 'prompt_id': 'c', 'split': 'check',
                             'style_llr_per_token': {'villain': style}, 'task_score': {'answer_match': answer}})
            return generation_verdict(rows, 'villain', {'criteria': {'min_prompts_improved': 7}})

        good = generation(1., True)
        self.assertTrue(good['C4_style'] and good['C5_no_lost_check'])
        self.assertTrue(nominate(forced, good))
        lost = generation(1., None)
        self.assertEqual(lost['lost_baseline_passed_checks'], ['c'])
        self.assertFalse(nominate(forced, lost))
        self.assertFalse(generation(-2., True)['C4_style'])

    def test_frozen_plan_matches_budget_and_planner_imports_no_torch(self):
        plan = load_plan(ROOT, CONFIG)
        self.assertEqual((plan['generations'], plan['grid_forwards'], plan['test_forwards']), (158, 2880, 288))
        self.assertEqual(len(plan['vectors']), 6)
        self.assertFalse({p['prompt'] for p in plan['prompts'] if p['split'] == 'test'} &
                         {p['prompt'] for p in plan['prompts'] if p['split'] != 'test'})
        code = ("import sys; sys.path.insert(0, 'scripts'); sys.argv=['x','plan']; import transfer_experiment as t; "
                "t.main(); assert 'torch' not in sys.modules")
        done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == '__main__':
    unittest.main()
