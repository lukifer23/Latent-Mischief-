# QWN-003: resource-bounded Mac budget follow-up

Status: **admission attempted and refused by system pressure; 0/4 research trajectories, no model preflight or checkpoint loading**. Protocol/source were committed and pushed as **18cf314** before the attempt. This is a new MPS experiment, not continuation of interrupted Windows QWN-002.

## Admission result — 2026-11-09

Run `20261109T200027330848Z-QWN-003-anti-loop-pilot` exited with the intended resource-stop code 2 before loading either checkpoint. macOS reported pressure value **2**, versus the required normal value 1. Available memory was 4.085 GiB, existing swap 4.029 GiB, process RSS 313.3 MiB and MPS driver allocation 8.47 MiB; no thermal/performance warning was reported. Swap growth in this brief admission check was zero. The available-memory floor was met; the independent OS-pressure gate refused the run. Apple's [sysctl conversion](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c) reports dispatch pressure levels; value 2 covers warning/urgent pressure, rather than normal. No pressure-setting command was used.

The initial resource sample, interruption, empty outputs and cleanup are preserved locally. Cleanup reports zero MPS tensor allocation and no retained model; the process exited. There is no preflight/active-trial file and no completed-study summary. Both planned pairs are missing. **No model-answer, repetition, anti-loop, capability or timing result exists from this attempt.**

[Numeric admission evidence](evidence/QWN-003/metrics.json) verifies all 24 archived source/input files and records the actual resource sample, zero saved rows and two missing pairs. Public export contains no generated traces. Fifty-eight Python tests and Node comparison/startup/model-isolation checks pass; guard boundaries, stop latching, partial-token preservation, paired schedules and empty-result exports are tested. Intermediate prefill/sampling under the new capped runner remains unexecuted because admission stopped first.

No automatic retry was performed and limits were not loosened. The next action is a new timestamped attempt of this unchanged config when macOS returns to normal pressure and at least 4 GiB is available. Preserve this refusal. With no research sample begun, a later admission attempt is not a reroll; an active unfinished trajectory would still be protected against rerolling. Updated report text changes its current hash, so strict archived-source continuation of this directory is intentionally not implied.

## Question and hypothesis

Does the pinned published Qwen3.5-0.8B edit improve paired final-answer/completion behavior with an 8,192-token output budget on this M3 Pro? Hypothesis: the larger budget may allow a difference to emerge that the incomplete 4K Windows pilot could not establish. This is a hypothesis, not an expectation or benchmark claim. Backend, dtype implementation, CPU thread limits and output budget differ from the Windows run; their effects cannot be isolated by a cross-run comparison.

Use the two lowest-level problems from QWN-002's already inspected selection: `test/counting_and_probability/14.json` and `test/algebra/1349.json`. Selection is by level/order, with no replacement after Mac outcomes. These are inspected follow-up prompts, not fresh validation. The second includes Asymptote source as text, without a rendered diagram. No new problem substitution or within-run rerolling.

## Frozen setup and schedule

[qwen-mac-followup.json](../configs/qwen-mac-followup.json) is authoritative. Dataset MATH-500 revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; source JSONL SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`. Regenerate the two-problem manifest with prepare_qwen_pilot.py --config configs/qwen-mac-followup.json. Its LF-byte digest is pinned separately from the Windows CRLF manifest. The Mac full-audit JSON has a separately pinned LF digest; its parsed tensor audit equals the published Windows audit. Do not change historical files to hide line-ending differences.

Base revision `2fc06364715b967f1860aea9cf38778875588b17`; edit revision `d1671a178705bcab7644806d23c535517f35723b`. M3 Pro, 18 GiB unified memory, macOS, MPS BF16, Transformers/PyTorch reference kernels, SDPA, no CPU operator fallback or fast math. Two CPU threads, one interop thread, process niceness increased by 10 to lower priority. This is a process setting, not a system configuration change. Preserve both checkpoints.

No system instruction. Original problem plus the exact boxed-answer suffix from config; thinking on. Temperature 0.6, top_p 0.95, top_k 20, output-only presence penalty 1.5, repetition penalty 1.0, seed 0 reset per trajectory. Tokenizer/checkpoint EOS, output cap and wall ceiling remain distinct. Use the existing last-balanced-box Math-Verify 0.9.0 grader in an isolated process with a 15-second timeout. Parser errors interrupt rather than count as model mistakes.

Order: base L1 → edit L1 → edit L2 → base L2. This finishes one pair before the second and counterbalances which variant runs first, requiring three checkpoint loads. Four planned trajectories, at most 8,192 each / 32,768 total, plus at most 32 short sampling sanity tokens. All loading, preflight, sampling and grading share a 30-minute cumulative ceiling. No model runs concurrently with another detected lab inference process.

## Resource gates and stopping

| Signal | Frozen rule |
|---|---|
| macOS pressure | Only normal sysctl pressure value 1 admitted; elevated/unknown stops |
| Available system memory | At least 4 GiB to start; stop below 2 GiB during execution |
| Existing swap | Recorded, not treated as newly caused by this run |
| Swap growth | Stop above 256 MiB increase from the run's initial value; no rebasing on resume |
| Process RSS | Stop above 4 GiB |
| MPS driver allocation | Stop above 4 GiB; unavailable measurement stops |
| MPS allocator | Set process fraction to 0.25 of the Metal recommended working set, about 3.33 GiB on this host |
| Thermal/performance warnings | Stop on reported warning/throttling or unknown telemetry |
| Disk | At least 5 GiB free |
| Manual cancellation | SIGINT/SIGTERM or a STOP file in the new run directory |

Memory/pressure/RSS/swap sampled every second, thermal status refreshed every five seconds. Stops latch even if pressure subsequently recovers; a background sampler flags the condition and generation checks every token. Telemetry failures stop conservatively. Sampling and cooperative interruption cannot prevent all sub-second excursions or preempt an in-flight kernel, model load or grader. The allocator limit provides a separate allocation ceiling; MPS driver snapshots and process RSS overlap in unified memory and must not be added as independent usage. See [PyTorch's process memory limit](https://docs.pytorch.org/docs/2.14/generated/torch.mps.set_per_process_memory_fraction.html).

Before research prompts, verify loaded base/edit tensor digests, thinking template, finite cached first logits and repeatability of two eight-token samples per variant. Check direct versus 256-token-chunked prefill at 256/1,024/2,048 fixed neutral input tokens. Record input IDs, finite/logit differences and first argmax; failure stops. These are bounded checks, not long-context equivalence. One loaded checkpoint is retained; release the model/cache on exit.

## Evidence and interpretation

Save the config/report/manifest/code/lock snapshots, runtime and process settings, complete tokens/text/input IDs, per-trial parameter-version checks, pending results before grading, 64-token checkpoints, interruption details, resource timeline and cleanup state under a new ignored run directory. Preserve partial outputs on cooperative interruption; exact KV/RNG continuation is not implemented. Refuse rerolling an active unfinished research trajectory, finalized runs, source/backend/environment drift or invalid schedule prefixes. No automatic retry after a resource stop.

Report every full row and matched pair, missing counterparts, final-box correctness, EOS/cap, repetition, actual token counts and timings. Interrupted trajectories are not full-budget counterparts. The edited model must improve a paired correctness/completion outcome to nominate a pilot observation; report regressions too. Less exact repetition alone is insufficient. Two inspected problems and one seed do not establish a general capability or anti-loop benefit. A broad claim requires a separately frozen fresh set and multiple seeds.

Public export contains numeric evidence, checks and source/raw hashes; raw traces and token sequences remain local. If the admission gate refuses to start, report zero trajectories and the observed resource reason without inventing a model result.
