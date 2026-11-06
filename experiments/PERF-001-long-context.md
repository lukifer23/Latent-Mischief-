# PERF-001: measured long-context performance

Status: predeclared. No 128K result claimed before completion.

Verify LiquidAI/LFM3-1.7B against pinned Hub revision, config architecture/model type and cached safetensors SHA-256. Official model card advertises a 256K total window; local config declares 262,144 positions. Target actual input lengths 2,048 / 8,192 / 32,768 / 131,072 tokens, each with up to 64 greedy output tokens. Input plus output remains below the declared limit.

Use BF16 and PyTorch SDPA with 256-token cached prefill chunks to bound attention workspace on the local 16GB GPU. This is a different attention implementation from prior eager experiments; do not pool their timing/behavior claims. Verify finite short-input chunked logits and identical first argmax before long tests. Record the observed numerical difference; this is a limited preflight, not a proof of all cache behavior.

Compare baseline and the frozen positive L04N5848 20%-target offset at each length, alternating condition order. One measurement per point: descriptive only, not a timing distribution. The deterministic project-document workload tiles repository documents with separators and a marker at midpoint, then requests the marker plus a short method summary. It exercises real tokenized inputs; it is not representative production traffic or a validated long-context reasoning benchmark.

Record loaded identity, environment, backend selection, model-load time, prefill time/rate, time to first token, decode time/rate with actual denominator, peak allocated/reserved GPU memory, process RSS, actual input/output counts, EOS and marker-presence check. Raw inputs/outputs remain local. Do not silently truncate a failed point; preserve OOM/error evidence and report unmeasured sizes. No admin changes or paid compute.
