# PERF-001: measured long-context performance

Status: six timing points completed through 32K input tokens; the actual 128K attempt failed with CUDA out of memory. No completed 128K timing or validated 128K usability is claimed. The design below was predeclared.

Verify LiquidAI/LFM3-1.7B against pinned Hub revision, config architecture/model type and cached safetensors SHA-256. Official model card advertises a 256K total window; local config declares 262,144 positions. Target actual input lengths 2,048 / 8,192 / 32,768 / 131,072 tokens, each with up to 64 greedy output tokens. Input plus output remains below the declared limit.

Use BF16 and PyTorch SDPA with 256-token cached prefill chunks to bound attention workspace on the local 16GB GPU. This is a different attention implementation from prior eager experiments; do not pool their timing/behavior claims. Verify finite short-input chunked logits and identical first argmax before long tests. Record the observed numerical difference; this is a limited preflight, not a proof of all cache behavior.

Compare baseline and the frozen positive L04N5848 20%-target offset at each length, alternating condition order. One measurement per point: descriptive only, not a timing distribution. The deterministic project-document workload tiles repository documents with separators and a marker at midpoint, then requests the marker plus a short method summary. It exercises real tokenized inputs; it is not representative production traffic or a validated long-context reasoning benchmark.

Record loaded identity, environment, backend selection, model-load time, prefill time/rate, time to first token, decode time/rate with actual denominator, peak allocated/reserved GPU memory, process RSS, actual input/output counts, EOS and marker-presence check. Raw inputs/outputs remain local. Do not silently truncate a failed point; preserve OOM/error evidence and report unmeasured sizes. No admin changes or paid compute.

## Results and failures

The actual model is **LiquidAI/LFM3-1.7B**, pinned at `0dc4268b561499c648c7ba7c9466bf3d8d01517a`. Config type `lfm3`, runtime class `Lfm3ForCausalLM`, and cached checkpoint SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` matched the pinned Hub LFS digest. This is LFM3, not an LFM2 substitute. Its declared window is a model limit, not proof this machine can serve it.

Run `20261106T202324479004Z-PERF-001-long-context` used native Windows, RTX 2000 Ada 16GB, BF16 SDPA and 256-token cached prefill chunks. [Verified numeric export](evidence/PERF-001/metrics.json) includes exact counts, memory, throughput denominators and source hashes.

| Actual input | Condition | Prefill seconds | Decode tokens/s | Peak GPU allocated / reserved GiB |
|---:|---|---:|---:|---:|
| 2,048 | Default | 0.321 | 34.13 | 3.28 / 3.32 |
| 2,048 | Offset | 0.318 | 35.15 | 3.31 / 3.35 |
| 8,192 | Default | 1.422 | 35.19 | 3.50 / 3.85 |
| 8,192 | Offset | 1.416 | 35.00 | 3.43 / 3.78 |
| 32,768 | Default | 7.970 | 28.55 | 3.99 / 11.45 |
| 32,768 | Offset | 7.990 | 28.58 | 4.27 / 11.81 |
| 131,072 | Offset attempted first | **Out of memory** | Unmeasured | No completed point |

Each completed point produced 64 tokens; decode throughput counts the 63 tokens after the first. First-token time and prefill are separately recorded, and model loading is excluded. The midpoint marker appeared through 32K; that narrow retrieval check does not establish general long-context reasoning.

Short-input chunked versus direct prefill had the same first argmax and finite logits, with maximum absolute logit difference **0.3125**. This is not numerical equivalence. Because the process failed at 128K, a final full-run parameter-version check is unavailable; earlier completed measurements remain preserved.

The first attempt (`20261106T202146443379Z-PERF-001-long-context`) stopped before timing because the tokenizer returned a BatchEncoding rather than the assumed tensor. That adapter was corrected before the measured run. A separate frozen-input allocator follow-up, PERF-002 (`20261106T203254347115Z-PERF-002-long-context`), set process-local `expandable_segments:True` and failed at model loading with CUDA error 999 / GPU mapping failure. It supplied no measurements and is not enabled for chat. No driver, global allocator or admin changes were made.

These single timings do not establish an optimization gain. Next performance work should reduce allocator/cache pressure and repeat measurements, preserving the same tokenized workload. Qwen long-context performance is still unmeasured.
