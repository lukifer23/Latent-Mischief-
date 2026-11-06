from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from character_lab import rank_profile, sparse_hook


class CharacterTests(unittest.TestCase):
    def test_consistent_channels_rank_over_larger_inconsistent_difference(self):
        import torch
        pairs = torch.tensor([[2.,1.,100.], [2.,1.,100.], [2.,1.,100.], [2.,1.,-100.],
                              [2.,1.,-100.], [2.,1.,-100.], [2.,1.,-100.], [2.,1.,-100.]])
        channels,native,ranking = rank_profile(pairs,torch.ones(3),2,.75)
        self.assertEqual(channels,[0,1])
        self.assertAlmostEqual(native.norm().item(),1,places=6)
        self.assertTrue(all(x['pair_consistency']==1 for x in ranking))
        with self.assertRaises(ValueError): rank_profile(pairs,torch.ones(3),3,.75)

    def test_sparse_offsets_cleanup_and_reject_duplicates(self):
        import torch
        module = torch.nn.Linear(3,2,bias=False)
        with torch.no_grad(): module.weight.copy_(torch.tensor([[1.,2.,3.],[2.,1.,1.]]))
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(feed_forward=SimpleNamespace(w2=module))]),device='cpu',dtype=torch.float32)
        x = torch.tensor([[1.,2.,3.]])
        baseline,weights = module(x).detach().clone(),module.weight.detach().clone()
        expected = module(x+torch.tensor([[1.,0.,-2.]])).detach().clone()
        with self.assertRaises(RuntimeError):
            with sparse_hook(model,0,[0,2],[1.,-2.]):
                self.assertTrue(torch.allclose(module(x),expected))
                self.assertFalse(torch.equal(module(x),baseline))
                raise RuntimeError('simulated failure')
        self.assertTrue(torch.equal(module(x),baseline))
        self.assertTrue(torch.equal(module.weight,weights))
        self.assertTrue(torch.equal(x,torch.tensor([[1.,2.,3.]])))
        with self.assertRaises(ValueError):
            with sparse_hook(model,0,[0,0],[1.,2.]): pass
