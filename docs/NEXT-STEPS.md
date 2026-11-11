# Next milestones and decision gates

Updated 2026-11-09 after the [project review](REVIEW-2026-11-09.md) and [Mac setup](../experiments/BACKEND-001-mac-setup.md). The end goal is a local causal-behavior lab: discover reproducible effects, retain damage/no-effect cases, and build useful reversible controls. Hamster/villain characterization remains an exploratory application. Capability improvement has not been established.

**Update 2026-11-11:** LFM-011 (the LFM-010 design on native Windows CUDA) completed 54/54 with a likely negative result; see [LFM-011](../experiments/LFM-011-phase-timing-cuda.md). The Mac LFM-010 run is no longer needed for the primary question.

**Earlier direction:** build and run [LFM-010 phase timing](../experiments/LFM-010-phase-timing.md) when resources permit. The offline implementation and tests are complete; the first admission attempt refused before loading weights (0/54 replies). [QWN-003](../experiments/QWN-003-mac-followup.md) remains a separate zero-trajectory resource refusal. No automatic retry or service is scheduled.

## Current state

The environment, platform-aware lock, CUDA/MPS/explicit-CPU adapter, pinned downloads, short LFM3/Qwen MPS inference, cache checks, Qwen seeded sampling and paired workbench API are implemented and checked on the M3 Pro. Long-context MPS behavior, a Mac anti-loop pilot, full CPU checkpoint inference and visual browser QA remain unverified. Python/Node checks pass; Windows dispatch is tested without claiming a new Windows hardware run.

QWN-002's evaluator, protocol, grader and full audit already exist. Its Windows pilot stopped at six of eight saved samples, with only two complete matched pairs. Both pairs capped without EOS/final boxes. The 832-token partial trajectory cannot be exactly resumed because KV/RNG state was not saved. Preserve it; never reroll under its old run ID or pool it with Mac data. See [HANDOFF.md](HANDOFF.md).

## First: LFM-010, a small phase-timing experiment

The user chose this direction after the Qwen resource refusal. [LFM-010](../experiments/LFM-010-phase-timing.md) is implemented and frozen for a first guarded run. It compares one villain-derived sixteen-channel profile with its existing same-layer random control, recalibrating both to a small 5% local prefill effect on MPS. No historical Windows amplitude is treated as calibrated Mac strength.

~~~mermaid
flowchart LR
    A[Offline plan and tests] --> B{Normal pressure and 7 GiB available?}
    B -->|No| C[Keep research stopped; retain the plan]
    B -->|Yes| D[Verify weights and recalibrate on neutral prompts]
    D --> E[Check zero offsets and removed hooks]
    E --> F[Run bounded phase comparisons]
    F --> G[Verify all evidence and review blinded replies]
    G --> H[Freeze fresh replication or retain a negative result]
~~~

1. **Build/verify — complete:** explicit prefill/decode/both hooks, deterministic full schedule, input hashes, guarded recording, separate answer/format scores and blinded-review reporting. Preview with uv run --locked python scripts/phase_experiment.py plan; this imports no torch or model.
2. **Admit and calibrate — pending:** require normal macOS pressure and at least 7 GiB available. The 3.5-GB weight payload needs more headroom than Qwen. Verify checkpoint identity, then recalibrate the two frozen directions on four neutral inputs; actual BF16 strengths must meet ±25% tolerance of 5%. Stop rather than loosen strength/resource limits.
3. **Run the frozen pilot — pending:** three new absurd situations plus addition/JSON/factual checks; baseline, ordinary prompted villain and target/random profiles at three placements. Six neutral/removal checks included: **54 generations**, each at most 192 input/48 output tokens, **2,592 generated tokens total / eight minutes**. Start only after committing source/config/protocol. No automatic retry, exact KV/RNG continuation or dose escalation.
4. **Review honestly — pending:** recompute scores, verify exact full-product schedule and actual phase traces, then rate the 24 shuffled behavioral replies for style/coherence/task relevance. The first reply token comes from prefill; decode-only begins with the next forward. Preserve ambiguous grades, individual regressions, caps and no-effect cases.
5. **Decide before expanding:** primary timing comparison is decode-only versus both, requiring clearer coherent villain characterization over both and a stronger villain-minus-random count without task/EOS loss. Prefill comparison is exploratory. A promising result nominates a separately frozen fresh replication; it does not establish a personality neuron or capability improvement. A negative result is a valid completed milestone.

The first actual admission attempt is preserved as `20261109T214535812054Z-LFM-010-phase-timing`: normal pressure, but 6.395 GiB estimated available after runtime initialization against the 7-GiB floor; 0/54 replies, no weights loaded. An earlier lightweight snapshot showed 7.28 GiB, illustrating why admission must run inside the actual process. The runner used about 313 MiB RSS, so its initialization does not explain the whole snapshot change. Available memory includes inactive pages; it is not empty RAM. Release unused application memory before another explicitly initiated attempt; keep the same limits and use a new directory. No automatic retry or UI startup. MPS calibration/phase checks remain unverified.

## Then: expand only from the result

If timing helps, test new prompts before adding hamster, more layers or stronger doses. Keep the same random/prompting controls and separate tone, content, structure and damage. If the base model fails the small checks, build a larger versioned task baseline with explicit answer/format separation before making any preservation claim. Do not keep tuning against the first six evaluation prompts and label them held out.

## Separate Qwen work

QWN-003's evaluator and protocol are implemented; its first admission attempt stopped at OS pressure 2 with zero loaded models/trajectories. Its frozen four-sample, 8K-output/30-minute settings remain in configs/qwen-mac-followup.json. It can be attempted in a new directory when its own admission conditions permit, without pooling with interrupted Windows QWN-002. This is a separate track, not a prerequisite for LFM-010. No automatic retry is scheduled.

## Workbench, provenance and performance

| Priority | Action | Completion evidence |
|---|---|---|
| Near term | Privately transfer desired Windows runs/direction tensors for historical raw browsing | Verify archive hashes; preserve original directories. Public review works without transfer. |
| Before longer campaigns | Extend interruption coverage and specify continuation semantics | QWN-003/LFM-010 cooperative stop and raw-token retention are implemented. Add hard-kill/write-boundary coverage before long campaigns; saved tokens are not restored KV/RNG state. Exact trajectory continuation does not exist. |
| When an isolated browser is available | Visual/keyboard/responsive QA | Inspected screenshots and interaction outcomes on an isolated surface. No native desktop control. |
| After small memory trials | New Mac context-performance protocol | Frozen inputs/budgets and repeated timings; shared memory is not dedicated VRAM. Do not rerun CUDA 128K/allocator designs on MPS. |
| Later | Qwen batch launch UI, stronger evidence validation, reusable runner orchestration | Model-aware allowlisted API, complete schedule/provenance checks and meaningful regressions; keep greedy chat separate. |
| Backlog | Qwenputer/exact computation, LFM family transfer, permanent weight experiments | Specific protocols and separate authorization/documentation where required. |

Work directly on main, push reviewable milestones, and update README/ROADMAP/JOURNAL. Weights, environments, raw outputs and private conversations stay local. Original code/docs are MIT; upstream model terms remain separate.
