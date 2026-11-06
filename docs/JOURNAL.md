# Research journal

## 2026-11-06 — Lab kickoff

Decision: LFM3 first. Initial targets are hamster characterization and theatrical villain behavior, using reversible activation interventions before permanent weight edits.

Constraints: no administrator access, native Windows first, no paid compute. GPU: RTX 2000 Ada Generation, 16,380 MiB total; system RAM about 64 GiB. Existing user tools: uv, Git, Python 3.13 and uv-managed Python 3.11.

Verified Hub checkpoint: LiquidAI/LFM3-1.7B, revision 0dc4268b561499c648c7ba7c9466bf3d8d01517a. Config: 24 layers, hidden size 2048, MLP width 7680, sandwich normalization, 13 convolution + 5 sliding-attention + 6 full-attention layers. Repository ships custom Python code; inspect and pin it before execution.

Novelty caveat: public gguf-neuron-explorer documentation already mentions LFM3. We are not claiming the first LFM steering experiment. No successful behavior intervention has been observed in this lab yet.

### Environment and baseline milestone

Installed isolated Python 3.11.15 environment with torch 2.11.0+cu128, Transformers 5.22.2, huggingface-hub 1.33.0, accelerate 1.15.0 and safetensors 0.8.0. CUDA available; native Windows loading and inference succeeded without admin access, WSL or custom CUDA compilation.

Downloaded pinned 3,457,635,448-byte safetensors checkpoint. Inspected configuration, MLP/decoder path and convolution fallback; recorded source hashes. This was an architecture/compatibility inspection, not a full security audit.

Initial run: `runs/20261106T181203314902Z-LFM-001-baseline/`. Five baseline prompts at greedy decoding and 96 new tokens, plus one neutral gain hook. The gain=1 hook at layer 12, channel 0 (input of feed_forward.w2) produced token-identical identity output. Identity, capital and JSON prompts reached EOS. Arithmetic and story hit the 96-token cap; no claim of looping. Arithmetic ignored integer-only formatting and contained a wrong intermediate result (587; expected final product is 391). Small baseline set only, not a capability benchmark.

Implementation note: metadata capture was improved while the initial process was running, so the initial on-disk script hash is not an exact archive of that process's loaded source. A finalized rerun is required for the clean reference artifact. Do not treat the initial run as the reproducibility reference.

Next: paired persona discovery responses and reversible direction sweeps under LFM-002 / LFM-003. Public repository name pending user choice; 'Latent Mischief' proposed. Nothing has been published.

Finalized rerun: `runs/20261106T181325639283Z-LFM-001-baseline/`. Neutral hook passed again. Full token inputs, checkpoint generation defaults and the unchanged executed script hash are captured. This is the reference baseline artifact; both baseline runs produced the same token sequences on all five prompts.

### Project name and public repository

User selected **Latent Mischief** and authorized the initial commit and GitHub publication at https://github.com/lukifer23/Latent-Mischief-. Local directory stays at its existing path so the working environment and raw evidence remain intact. Public source excludes caches, weights, environments and raw run output. The project package was renamed; the earlier baseline retains its original lock hash as historical provenance.
