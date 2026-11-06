# Local setup without administrator access

Use existing user-installed uv and Git. `uv sync --locked` creates `.venv` using the existing user-managed Python 3.11. The lock uses prebuilt PyTorch CUDA 13.0 wheels from the official PyTorch index. The original experiments used CUDA 12.8 wheels; the change addresses dependency advisories and requires revalidation. No CUDA toolkit, compiler, WSL, driver installation, or shell activation is required for this path on the documented machine.

Project dependencies remain in `.venv`. uv may use its existing user cache; Hugging Face files used by the lab live in project `.cache/huggingface`. Windows may copy cache files rather than symlink them; administrator access / Developer Mode is not needed.

## Verify

```powershell
uv run python scripts/lab.py doctor
```

Expect CUDA available and RTX 2000 Ada Generation. If unavailable, inspect the error before changing anything. Do not update system drivers or install elevated tools.

## Model code

The pinned checkpoint uses custom Python code. `prepare` downloads source files without executing them and writes their hashes to `runs/source-manifest.json`. Loading uses `trust_remote_code=True` at the pinned revision. Inspect that source when changing model revisions. Custom code relies on Transformers APIs; retain uv.lock with results.

The native PyTorch attention and convolution fallback is the initial path. Optimized kernels are optional future work, not a setup prerequisite.

## Qwen track

`uv run --locked python scripts/prepare_models.py` downloads the pinned official Qwen3.5-0.8B base and separate published anti-loop edit (roughly 1.7GB each), then verifies the two edited tensor digests. Pins live in `configs/models.json`; this command does not follow the latest model revision. Qwen uses native Transformers modeling with `trust_remote_code=False`; LFM3 uses its inspected pinned custom code. Both use the project environment.

On this Windows machine, Qwen uses reference PyTorch convolution and gated-delta kernels because optional `causal_conv1d` and `flash-linear-attention` packages are absent. Local smoke inference passed with those fallbacks; speed is not comparable to optimized-kernel benchmarks. Do not enable `expandable_segments:True`: the separate local performance attempt failed with a GPU mapping error. Neither model requires admin privileges for the verified path.

## Reproducibility

Use `uv sync --locked`. Each run records package versions, model revision, lock/script hashes, prompts, token IDs, output text, generation settings and timing. New runs also archive exact local source, config, prompt and lock files. JSON replacement is atomic so a failed update preserves the previous complete record. Keep raw runs locally; summarize reviewed evidence in tracked experiment reports. Do not commit model files or credentials.
