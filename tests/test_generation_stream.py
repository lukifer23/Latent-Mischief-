from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import lab
from resource_guard import ResourceLimitError


class GenerationStreamTests(unittest.TestCase):
    def setup_model(self):
        import torch
        from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=32, n_embd=8, n_layer=1,
            n_head=1, eos_token_id=2, pad_token_id=0, bos_token_id=1)).eval()
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def apply_chat_template(self, *args, **kwargs):
                return BatchEncoding({'input_ids': torch.tensor([[3, 4]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long)})
            def decode(self, tokens, **kwargs):
                return ' '.join(str(token) for token in tokens)
        return model, Tokenizer()

    def test_real_hf_cpu_stream_matches_nonstreamed_path(self):
        model, tokenizer = self.setup_model()
        baseline = lab.generate(model, tokenizer, 'test', max_new_tokens=3)
        inputs, checkpoints = [], []
        streamed = lab.generate(model, tokenizer, 'test', max_new_tokens=3,
            on_inputs=lambda ids: inputs.append(ids), on_tokens=lambda tokens, text: checkpoints.append(list(tokens)), check=lambda: None)
        self.assertEqual(streamed['token_ids'], baseline['token_ids'])
        self.assertEqual(inputs, [[3, 4]])
        self.assertEqual(streamed['input_tokens'], 2)

    def test_resource_abort_checkpoints_the_actual_generated_token(self):
        model, tokenizer = self.setup_model()
        checks, saved = [], []
        def stop():
            checks.append(True)
            if len(checks) >= 2:
                raise ResourceLimitError('test pressure')
        with self.assertRaisesRegex(ResourceLimitError, 'pressure'):
            lab.generate(model, tokenizer, 'test', max_new_tokens=3, check=stop,
                         on_tokens=lambda tokens, text: saved.append(list(tokens)))
        self.assertEqual(len(saved[-1]), 1)
        with self.assertRaisesRegex(ValueError, 'input exceeds'):
            lab.generate(model, tokenizer, 'test', max_new_tokens=3, max_input_tokens=1)
