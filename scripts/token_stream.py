"""Shared raw-token checkpoints and cooperative checks for local HF generation."""
import time


class TokenRecorder:
    def __init__(self, tokenizer, callback, selected, check=None, interval=64):
        if type(interval) is not int or interval < 1:
            raise ValueError('Positive checkpoint interval required')
        self.tokenizer, self.callback, self.selected = tokenizer, callback, selected
        self.check, self.interval = check, interval
        self.tokens = []
        self.prompt_seen = False
        self.first = None
        self.ended = False

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.first is None:
            self.selected.synchronize()
            self.first = time.perf_counter()
        self.tokens.extend(value.reshape(-1).tolist())
        if len(self.tokens) % self.interval == 0:
            self.callback(self.tokens, self.tokenizer.decode(self.tokens, skip_special_tokens=False))
        if self.check:
            self.check()

    def end(self):
        self.ended = True
