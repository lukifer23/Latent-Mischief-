# QWN-004: Qwen3.5-0.8B base vs published edit, 8K budget, native Windows CUDA

Status: **implemented; admission/run pending at the time of writing** (see the results section if present). This is a new CUDA experiment. It is not a continuation of the interrupted Windows QWN-002 and is not pooled with the Mac-targeted, resource-refused [QWN-003](QWN-003-mac-followup.md).

## Question and scope

Same question as QWN-003: with an 8,192-token output budget, does the pinned published Qwen3.5-0.8B edit improve paired final-answer/completion behavior on the two lowest-level problems already inspected in QWN-002 (`test/counting_and_probability/14.json`, `test/algebra/1349.json`)? These are inspected follow-up prompts, not fresh validation; no problem replacement after outcomes. One seed and two problems cannot establish a general capability or anti-loop effect. A nomination needs a paired correctness/completion gain and reports regressions; less repetition alone is insufficient.

## What is identical to QWN-003, and what differs

Identical: both model revisions (base `2fc06364...`, edit `d1671a17...`), sampling (T 0.6, top-p 0.95, top-k 20, output presence penalty 1.5, repetition penalty 1.0, seed 0 reset per trajectory), thinking on, no system prompt, boxed-answer suffix, last-balanced-box Math-Verify grading in an isolated process, SDPA attention, schedule order (base L1, edit L1, edit L2, base L2), four trajectories at most 8,192 tokens (32,768 total), bounded preflight (tensor digests, thinking template, seeded-sample repeat, direct vs 256-chunked prefill at 256/1,024/2,048).

Differs: native Windows CUDA/BF16 instead of macOS MPS/BF16; cumulative wall ceiling **60 minutes** (CUDA throughput is unmeasured here; QWN-003's 30 minutes was an MPS budget); resource policy has no OS pressure level (12 GiB available RAM at start / 6 running, 8 GiB free GPU at start / 3 running, 7 GiB CUDA reserved ceiling, allocator fraction 0.5, RSS 8 GiB, swap growth 256 MiB, disk 5 GiB, nvidia-smi thermal/throttle stop, unknown telemetry stops); below-normal process priority. Manifest regenerated from the same pinned MATH-500 revision (`.cache/qwn004/problems.json`, SHA-256 `317e543e...` pinned in config; local, ignored). Full-checkpoint audit digest is the historical Windows one (`0f3e772a...`).

## Commands

```sh
uv run --locked python scripts/prepare_qwen_pilot.py --config configs/qwen-cuda-followup.json
uv run --locked python scripts/qwen_eval.py --config configs/qwen-cuda-followup.json
uv run --locked python scripts/qwen_report.py RUN-ID
```

No automatic retry, dose or budget escalation, rerolling of an unfinished trajectory, or exact KV/RNG continuation. A resource/wall stop preserves partial tokens as interrupted, never as a complete pair.
