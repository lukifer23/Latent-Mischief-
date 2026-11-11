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
    def test_followup_schedule_requires_complete_pairs_and_frozen_mps_policy(self):
        import copy,json
        from qwen_eval import validate_design
        from runtime import Runtime
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/qwen-mac-followup.json').read_text())
        manifest={'problems':[{'unique_id':x} for x in dict.fromkeys(t['prompt_id'] for t in cfg['schedule'])]}
        from unittest.mock import patch
        with patch('platform.system',return_value='Darwin'):
            validate_design(cfg,manifest,Runtime('mps','bfloat16'))
            broken=copy.deepcopy(cfg);broken['schedule'][-1]=broken['schedule'][0]
            with self.assertRaisesRegex(ValueError,'exactly once'):validate_design(broken,manifest,Runtime('mps','bfloat16'))
            with self.assertRaisesRegex(ValueError,'MPS/BF16'):validate_design(cfg,manifest,Runtime('cpu','bfloat16'))

    def test_streamer_preserves_the_current_token_before_resource_interruption(self):
        import torch
        from qwen_eval import Recorder
        from resource_guard import ResourceLimitError
        from unittest.mock import Mock
        def stop():raise ResourceLimitError('pressure')
        selected=Mock()
        recorder=Recorder(Mock(),lambda tokens,text:None,selected,stop)
        recorder.put(torch.tensor([[1,2]]))
        with self.assertRaises(ResourceLimitError):recorder.put(torch.tensor([7]))
        self.assertEqual(recorder.tokens,[7])
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


class CudaFollowupTests(unittest.TestCase):
    def test_qwn004_is_cuda_only_and_keeps_the_frozen_budget(self):
        import copy,json
        from qwen_eval import validate_design
        from runtime import Runtime
        from unittest.mock import patch
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/qwen-cuda-followup.json').read_text())
        manifest={'problems':[{'unique_id':x} for x in dict.fromkeys(t['prompt_id'] for t in cfg['schedule'])]}
        with patch('sys.platform','win32'):
            validate_design(cfg,manifest,Runtime('cuda','bfloat16'))
            with self.assertRaisesRegex(ValueError,'CUDA/BF16'):validate_design(cfg,manifest,Runtime('mps','bfloat16'))
            with self.assertRaisesRegex(ValueError,'CUDA/BF16'):validate_design(cfg,manifest,Runtime('cuda','float16'))
            for key,value in [('max_new_tokens',4096),('wall_seconds',1800)]:
                broken=copy.deepcopy(cfg);broken[key]=value
                with self.assertRaises(ValueError):validate_design(broken,manifest,Runtime('cuda','bfloat16'))
        with patch('sys.platform','darwin'):
            with self.assertRaisesRegex(ValueError,'CUDA/BF16'):validate_design(cfg,manifest,Runtime('cuda','bfloat16'))
