# Local setup without administrator access

Use existing user-installed uv and Git. `uv sync --locked` creates `.venv` using the existing user-managed Python 3.11. The lock includes prebuilt PyTorch CUDA 12.8 wheels from the official PyTorch index. No CUDA toolkit, compiler, WSL, driver installation, or shell activation is required for this path.

Project dependencies remain in `.venv`. uv may use its existing user cache; Hugging Face files used by the lab live in project `.cache/huggingface`. Windows may copy cache files rather than symlink them; administrator access / Developer Mode is not needed.

## Verify

```powershell
uv run python scripts/lab.py doctor
```

Expect CUDA available and RTX 2000 Ada Generation. If unavailable, inspect the error before changing anything. Do not update system drivers or install elevated tools.

## Model code

The pinned checkpoint uses custom Python code. `prepare` downloads source files without executing them and writes their hashes to `runs/source-manifest.json`. Loading uses `trust_remote_code=True` at the pinned revision. Inspect that source when changing model revisions. Custom code relies on Transformers APIs; retain uv.lock with results.

The native PyTorch attention and convolution fallback is the initial path. Optimized kernels are optional future work, not a setup prerequisite.

## Reproducibility

Use `uv sync --locked`. Each run records package versions, model revision, lock/script hashes, prompts, token IDs, output text, generation settings and timing. New runs also archive exact local source, config, prompt and lock files. JSON replacement is atomic so a failed update preserves the previous complete record. Keep raw runs locally; summarize reviewed evidence in tracked experiment reports. Do not commit model files or credentials.
