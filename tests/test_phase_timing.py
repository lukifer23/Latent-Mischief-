from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from phase_timing import timed_sparse_hook
from phase_experiment import validate_trace


def tiny_model():
    import torch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            block = torch.nn.Module()
            block.feed_forward = torch.nn.Module()
            block.feed_forward.w2 = torch.nn.Linear(3, 3, bias=False)
            with torch.no_grad(): block.feed_forward.w2.weight.copy_(torch.eye(3))
            self.model.layers = torch.nn.ModuleList([block])
        @property
        def device(self): return self.model.layers[0].feed_forward.w2.weight.device
        @property
        def dtype(self): return self.model.layers[0].feed_forward.w2.weight.dtype
        def forward(self, input_ids):
            x = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
            return self.model.layers[0].feed_forward.w2(x)
    return Tiny()


class PhaseHookTests(unittest.TestCase):
    def test_phase_boundaries_are_call_based_even_for_one_token_prefill(self):
        import torch
        for phase in ('prefill', 'decode', 'all'):
            model = tiny_model()
            weights = model.model.layers[0].feed_forward.w2.weight.detach().clone()
            ids = torch.tensor([[3]])
            with timed_sparse_hook(model, 0, [1], [2.], phase) as trace:
                first = model(input_ids=ids)
                second = model(input_ids=torch.tensor([[4]]))
                third = model(input_ids=torch.tensor([[5]]))
            self.assertEqual(first[0, 0, 1].item(), 5. if phase in ('prefill', 'all') else 3.)
            self.assertEqual(second[0, 0, 1].item(), 6. if phase in ('decode', 'all') else 4.)
            self.assertEqual(third[0, 0, 1].item(), 7. if phase in ('decode', 'all') else 5.)
            validate_trace(trace, phase, 3, 1)
            with self.assertRaisesRegex(ValueError, 'token positions'):
                validate_trace({**trace, 'intervened_positions': 999}, phase, 3, 1)
            self.assertTrue(torch.equal(weights, model.model.layers[0].feed_forward.w2.weight))
            self.assertEqual(ids.tolist(), [[3]])
            self.assertFalse(model._forward_pre_hooks)
            self.assertFalse(model.model.layers[0].feed_forward.w2._forward_pre_hooks)

    def test_full_prompt_positions_change_and_decode_shape_errors_clean_up(self):
        import torch
        model = tiny_model()
        with timed_sparse_hook(model, 0, [0], [1.], 'prefill') as trace:
            output = model(torch.tensor([[2, 3, 4]]))
        self.assertEqual(output[0, :, 0].tolist(), [3., 4., 5.])
        self.assertEqual(trace['intervened_positions'], 3)
        with self.assertRaisesRegex(ValueError, 'cached token'):
            with timed_sparse_hook(model, 0, [0], [1.], 'decode'):
                model(input_ids=torch.tensor([[1, 2]]))
                model(input_ids=torch.tensor([[1, 2, 3]]))
        self.assertFalse(model._forward_pre_hooks)
        self.assertEqual(model(torch.tensor([[2]]))[0, 0, 0].item(), 2.)

    def test_zero_offset_and_failure_leave_no_hook_or_parameter_change(self):
        import torch
        model = tiny_model(); ids = torch.tensor([[1, 2]])
        reference = model(ids)
        with self.assertRaises(RuntimeError):
            with timed_sparse_hook(model, 0, [0], [0.], 'all'):
                self.assertTrue(torch.equal(reference, model(ids)))
                raise RuntimeError('generation interrupted')
        self.assertTrue(torch.equal(reference, model(ids)))
        self.assertFalse(model._forward_pre_hooks)
        for phase, channels, offsets in [('invalid', [0], [1.]), ('all', [0, 0], [1., 2.]), ('all', [0], [float('nan')])]:
            with self.assertRaises(ValueError):
                with timed_sparse_hook(model, 0, channels, offsets, phase):
                    model(ids)

    def test_first_token_eos_can_have_zero_decode_interventions(self):
        import torch
        model = tiny_model()
        with timed_sparse_hook(model, 0, [1], [2.], 'decode') as trace:
            model(torch.tensor([[3, 4]]))
        validate_trace(trace, 'decode', 1, 2)
        self.assertEqual(trace['decode_interventions'], 0)
