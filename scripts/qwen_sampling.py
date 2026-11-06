"""Output-only presence penalty before HF temperature/top-k/top-p warpers."""
import torch


class OutputPresencePenalty:
    def __init__(self,prompt_length,penalty):
        self.prompt_length=prompt_length;self.penalty=penalty

    def __call__(self,input_ids,scores):
        if not torch.isfinite(scores).all():raise FloatingPointError('Non-finite model logits before sampling')
        seen=torch.zeros_like(scores,dtype=torch.bool)
        generated=input_ids[:,self.prompt_length:]
        if generated.numel():seen.scatter_(1,generated,True)
        return scores-self.penalty*seen
