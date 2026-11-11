# QWN-004: Qwen3.5-0.8B base vs published edit, 8K budget, native Windows CUDA

Status: **complete (4/4 at 8K, verified). No completion or correctness benefit observed; edit shows lower repetition only.** This is a new CUDA experiment. It is not a continuation of the interrupted Windows QWN-002 and is not pooled with the Mac-targeted, resource-refused [QWN-003](QWN-003-mac-followup.md).

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

## 2026-11-11: results (run `20261111T163309240974Z-QWN-004-anti-loop-pilot`)

Predeclared at `b01f997`. Runtime-only fixes before the completed run (`eeef1ff` audit digest pin, then the preflight fix below); protocol, prompts, sampling and limits unchanged. **Complete: 4/4 trajectories, 32,768 generated tokens, 1,280 s total** (about 316 s each, 25.9 decode tokens/s on the RTX 2000 Ada with the PyTorch reference fallbacks for the linear-attention kernels). Reporter verified 25 archived source files, regraded all 4 rows, matched actual tokenized inputs for both pairs, no missing pairs. Preflight checks (tensor digests, thinking template, seeded repeat, cached prefill at 256/1,024/2,048) passed.

| Problem | Variant | Tokens | Termination | Boxed answer | Repeated 4-gram fraction |
|---|---|---:|---|---|---:|
| L1 counting/14 | base | 8,192 | output cap | none | 0.80 |
| L1 counting/14 | edit | 8,192 | output cap | none | 0.62 |
| L2 algebra/1349 | base | 8,192 | output cap | none | 0.68 |
| L2 algebra/1349 | edit | 8,192 | output cap | none | 0.58 |

**Observation.** All four trajectories reached the 8,192 cap with no final box and no EOS, so correctness and completion are 0/2 for both variants; the two top panels of the figure are blank because every value is zero. The edit's repeated-4-gram fraction is lower in both pairs (0.18 and 0.10 absolute), but the protocol states reduced repetition alone is insufficient and one seed on two inspected problems gives no significance estimate. The edit L2 trajectory contains one think-close marker yet still produced no box. **No completion or correctness benefit was observed at 8K**, extending the Windows 4K partial pilot (QWN-002) to a complete 8K pair; budgets, backend and seeds still differ across experiments, so nothing is pooled. Hypothesis, not observation: much larger budgets or different sampling might change completion; that would be a separately frozen experiment.

## Preflight bug found (affects QWN-003 too)

The first two attempts stopped before any trajectory: a wrong audit digest copied from the Mac config (no run directory, fails closed), then a crash in the chunked-prefill check. After `generate()`, Qwen3.5's multimodal wrapper builds position ids from a full-length attention mask, so passing `attention_mask` with a cache and a 256-token chunk raised a 256 vs 512 shape error. Reproduced on real weights and verified that omitting the mask (batch of one, no padding) matches direct prefill (argmax identical; max logit difference 0.1875). The QWN-003 Mac attempt never reached this check, so QWN-003 would have failed the same way; the fix is in `scripts/qwen_preflight.py` and applies to it once retried. Failed-attempt run directory: `20261111T162559293523Z-QWN-004-anti-loop-pilot` (0 trajectories).
