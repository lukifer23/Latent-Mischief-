"""Analytic distribution checks and real Math-Verify parsing; no mocked models."""
import math
from pathlib import Path
import sys
import unittest
import importlib.util

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from math_grade import grade,last_box


class GradingTests(unittest.TestCase):
    def test_last_box_handles_nested_fraction_and_rejects_unclosed_last_box(self):
        self.assertEqual(last_box(r'\boxed{1} then \boxed{\frac{1}{2}}')[0],r'\frac{1}{2}')
        self.assertEqual(last_box(r'\boxed{1} then \boxed{2')[1],'unclosed-last-box')
        self.assertEqual(last_box('No answer')[1],'missing-box')

    @unittest.skipUnless(importlib.util.find_spec('math_verify'),'Install the pinned eval extra for real Math-Verify checks')
    def test_real_grader_equivalence_wrong_last_answer_and_text(self):
        self.assertTrue(grade(r'Final \boxed{0.25}',r'\frac14')['correct'])
        self.assertFalse(grade(r'\boxed{36} correction \boxed{35}','36')['correct'])
        self.assertTrue(grade(r'\boxed{\text{Evelyn}}',r'\text{Evelyn}')['correct'])


class SamplingTests(unittest.TestCase):
    def test_saved_schedule_prefix_rejects_duplicate_or_changed_trials(self):
        from qwen_eval import validate_prefix
        schedule=[{'condition':c,'prompt_id':'p','seed':0} for c in ['baseline','published-checkpoint']]
        validate_prefix(schedule[:1],schedule)
        with self.assertRaises(ValueError):validate_prefix([schedule[0],schedule[0]],schedule)
        with self.assertRaises(ValueError):validate_prefix([{**schedule[0],'seed':1}],schedule)
    def test_presence_ignores_prompt_and_does_not_count_repetition_frequency(self):
        import torch
        from qwen_sampling import OutputPresencePenalty
        scores=torch.tensor([[0.,1.,2.,3.]])
        ids=torch.tensor([[0,1,2,2]])
        result=OutputPresencePenalty(2,1.5)(ids,scores)
        self.assertEqual(result.tolist(),[[0.,1.,.5,3.]])
        self.assertEqual(scores.tolist(),[[0.,1.,2.,3.]])
        with self.assertRaises(FloatingPointError):OutputPresencePenalty(0,1)(ids,torch.tensor([[float('nan')]*4]))

    def test_hf_warpers_match_independent_small_distribution(self):
        import torch
        from transformers.generation.logits_process import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
        values=[.2,1.4,-.1,2.1,.7];temperature=.6;k=3;p=.8
        order=sorted(range(len(values)),key=lambda i:values[i],reverse=True)[:k]
        weights={i:math.exp(values[i]/temperature) for i in order};normalizer=sum(weights.values())
        cumulative=0.;keep=[]
        for i in order:
            keep.append(i);cumulative+=weights[i]/normalizer
            if cumulative>=p:break
        expected=[weights[i]/sum(weights[j] for j in keep) if i in keep else 0. for i in range(len(values))]
        logits=torch.tensor([values],dtype=torch.float64);ids=torch.tensor([[0]])
        for warper in [TemperatureLogitsWarper(temperature),TopKLogitsWarper(k),TopPLogitsWarper(p)]:logits=warper(ids,logits)
        torch.testing.assert_close(logits.softmax(-1),torch.tensor([expected],dtype=torch.float64))
