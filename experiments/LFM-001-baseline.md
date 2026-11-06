# LFM-001: native Windows baseline and neutral hook

Status: complete (baseline and neutral-hook milestone only).
Date: 2026-11-06
Related roadmap IDs: LAB-001, LFM-001

## Question

Can the pinned LFM3 text model load and generate on this GPU without admin privileges? Does the proposed MLP hook preserve output at neutral gain?

## Setup

LiquidAI/LFM3-1.7B @ 0dc4268b561499c648c7ba7c9466bf3d8d01517a. Native Windows, RTX 2000 Ada 16 GB, BF16, eager attention, PyTorch convolution fallback. Python 3.11.15, torch 2.11.0+cu128, Transformers 5.22.2. Greedy generation, seed 42, 96 new-token cap, enable_thinking=False (requested; actual behavior may vary), checkpoint repetition_penalty=1.1. Prompts: prompts/baseline.json.

## Observation

- Model load and GPU generation succeeded.
- gain=1 at layer 12, neuron 0, feed_forward.w2 input: token-identical to the unsteered identity response.
- Identity, capital and JSON responses terminated with EOS.
- Arithmetic and story responses hit the 96-token budget. This does not establish a loop.
- Arithmetic response did not follow integer-only instruction and included a wrong intermediate value.

Initial raw run: runs/20261106T181203314902Z-LFM-001-baseline/. See journal for the initial script-hash limitation and the finalized rerun.

Clean reference run: `runs/20261106T181325639283Z-LFM-001-baseline/`. Includes complete input/output token IDs, environment, checkpoint generation defaults, script and lock hashes, module inventory and neutral-hook check. Neutral hook passed again; the five baseline outputs also matched the initial run token-for-token. Observed generation time per response was approximately 0.26–2.93 seconds, excluding model load; these are unoptimized smoke-test timings, not throughput benchmarks.

## Limits and next step

Only one neutral-hook prompt and five baselines. No persona, useful-edit, intelligence or performance improvement claim. Next: persona discovery and matched control experiments. No model weights were edited.
