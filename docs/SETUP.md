# Local setup without administrator access

Use existing user-installed Git and uv. Python dependencies belong in the project .venv; no activation, global Python change, system PATH change, compiler, driver installation or service is needed. Python 3.11 is selected by .python-version; uv.lock records the exact dependency resolution.

## Install and check

From the repository root, in macOS Terminal or Windows PowerShell:

~~~sh
uv sync --locked --all-extras
uv run --locked python scripts/lab.py doctor
uv run --locked python -m unittest discover -s tests -v
node tests/test_ui.js
node tests/test_ui_startup.js
~~~

Extras install matplotlib for figures and Math-Verify for real grading. Node is needed only for UI checks; the workbench needs no frontend build. The smaller default environment is installed with uv sync --locked.

The platform-aware lock retains the official PyTorch CUDA 13 index on Windows/Linux and uses PyPI's native wheel on macOS. The Mac dependency routing was already an uncommitted change at the start of the 2026-11-09 review; it was preserved and validated. Do not use a CUDA wheel on macOS.

## Backend policy

scripts/runtime.py selects available CUDA first, then Apple MPS. Both default to BF16. CPU inference must be explicitly selected and defaults to float32. LAB_DEVICE accepts auto, cuda, cuda:N, mps or cpu; LAB_DTYPE accepts bfloat16, float16 or float32. An unavailable requested backend or failed dtype probe stops before model loading. The lab never enables MPS-to-CPU operator fallback or relaxes memory limits.

On this Mac, doctor reports arm64, MPS available, backend mps, dtype bfloat16, and 18 GiB system memory. CUDA fields are null/false. The M3 Pro shares that memory with macOS and other applications; it is not dedicated VRAM. On the original Windows workstation expect CUDA and the RTX 2000 Ada. Do not change system drivers to resolve a doctor failure.

Explicit Mac choices:

~~~sh
LAB_DEVICE=mps LAB_DTYPE=bfloat16 uv run --locked python scripts/lab.py smoke
# Optional slow CPU path; full checkpoint CPU inference is not verified here:
LAB_DEVICE=cpu LAB_DTYPE=float32 uv run --locked python scripts/lab.py doctor
~~~

In PowerShell set $env:LAB_DEVICE = "cuda" before the command. Actual runtime identity is saved in metadata and generation records. A backend/dtype change creates new evidence; it does not continue a Windows experiment or validate its calibration. Historical recipe hooks require their recorded dtype.

CUDA records retain peak allocated/reserved memory. The pinned MPS API exposes current tensor and driver allocation, not equivalent peaks; CUDA peak fields are null on MPS. CPU GPU-memory fields are null. See [PyTorch's MPS API](https://docs.pytorch.org/docs/2.14/mps.html).

## Pinned models and bounded checks

~~~sh
uv run --locked python scripts/lab.py prepare
uv run --locked python scripts/lab.py smoke
uv run --locked python scripts/verify_model.py
uv run --locked python scripts/prepare_models.py
uv run --locked python scripts/qwen_audit.py
uv run --locked python scripts/cache_diagnostic.py --model lfm3
uv run --locked python scripts/cache_diagnostic.py --model qwen08
~~~

prepare downloads/hashes pinned LFM3 source/config without executing it. smoke downloads about 3.5 GB of weights and generates five short baselines plus a neutral-hook check. LFM3 uses inspected custom code at the pinned revision with trust_remote_code=True; Qwen uses native Transformers code with trust_remote_code=False. Neither modifies weights.

prepare_models.py reads configs/models.json, downloads the pinned Qwen base and separate published edit (about 1.7 GB each), and checks both edited-tensor digests. It does not rewrite the registry. qwen_audit.py compares every cached tensor on CPU. The cache diagnostic saves a new timestamped local directory with actual input IDs, source archives and numerical differences; Qwen also checks loaded tensor digests and repeatability of short thinking-on samples. These are setup checks, not an anti-loop benchmark.

Both model families ran with native PyTorch reference fallbacks on the M3 Pro. No Triton, FlashAttention, causal-convolution package or MLX conversion was installed. Windows CUDA behavior is retained in code and dispatch tests; actual Windows inference was not rerun from this Mac. See [the review](REVIEW-2026-11-09.md) and [BACKEND-001](../experiments/BACKEND-001-mac-setup.md).

## Workbench and evidence

~~~sh
uv run --locked python scripts/viewer.py
~~~

Open the printed loopback URL yourself. Start Workbench.command is the Mac launcher; Start Workbench.cmd is the Windows launcher. Neither opens or focuses a browser. Add --read-only to reject inference and local writes. A dependency-light review alternative:

~~~sh
uv run --no-project --python 3.11 --with psutil==7.2.2 python scripts/viewer.py --read-only
~~~

Models/cache files use project .cache/huggingface by default; an explicit caller HF_HOME override is honored. Raw outputs, token IDs, source/config/lock snapshots and private conversations stay in ignored runs/. Git provides reports, numbers and 135 recipe cards, not Windows raw runs or direction tensors. Catalog rebuilding retains the reviewed library while adding local evidence. Absent raw runs are marked in the UI; historical vector recipes still need their original artifacts.

Frozen QWN-002 and PERF-001/002 commands reject non-CUDA/BF16 runtimes before inference. Windows-derived calibration reuse also has backend/runtime gates. Follow [NEXT-STEPS.md](NEXT-STEPS.md) for a separate Mac study. Do not enable the failed expandable CUDA allocator or attempt 128K as an installation check.
