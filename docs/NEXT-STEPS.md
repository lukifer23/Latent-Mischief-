# Next milestones and decision gates

Updated 2026-11-09 after the [project review](REVIEW-2026-11-09.md) and [Mac setup](../experiments/BACKEND-001-mac-setup.md). The end goal is a local causal-behavior lab: discover reproducible effects, retain damage/no-effect cases, and build useful reversible controls. Hamster/villain characterization remains an exploratory application. Capability improvement has not been established.

**Execution update:** [QWN-003](../experiments/QWN-003-mac-followup.md) is now predeclared and implemented. Its first admission attempt stopped before loading weights because macOS pressure was 2; 0/4 trajectories exist. The next action is to run the same config under normal pressure with at least 4 GiB available. No automatic retry, limit relaxation or Windows sample continuation. This resource refusal is not a model result.

## Current state

The environment, platform-aware lock, CUDA/MPS/explicit-CPU adapter, pinned downloads, short LFM3/Qwen MPS inference, cache checks, Qwen seeded sampling and paired workbench API are implemented and checked on the M3 Pro. Long-context MPS behavior, a Mac anti-loop pilot, full CPU checkpoint inference and visual browser QA remain unverified. Python/Node checks pass; Windows dispatch is tested without claiming a new Windows hardware run.

QWN-002's evaluator, protocol, grader and full audit already exist. Its Windows pilot stopped at six of eight saved samples, with only two complete matched pairs. Both pairs capped without EOS/final boxes. The 832-token partial trajectory cannot be exactly resumed because KV/RNG state was not saved. Preserve it; never reroll under its old run ID or pool it with Mac data. See [HANDOFF.md](HANDOFF.md).

## First research milestone: separately freeze a Mac Qwen follow-up

Question: under one MPS/BF16/reference-kernel setup, does the published edit improve paired completion/final-answer outcomes at a larger output budget? Short arithmetic success and reduced exact repetition do not answer this.

1. Write a new experiment ID, report and config before generation. Keep both existing checkpoint revisions. Record hardware/runtime, attention, tokenizer/template, seed, temperature 0.6, top_p 0.95, top_k 20, output-only presence penalty 1.5, EOS and Math-Verify 0.9.0 rules. Reusing Windows problems is a labeled follow-up on inspected prompts, not fresh validation.
2. Implemented: the evaluator accepts --config configs/qwen-mac-followup.json. QWN-002's frozen gate and unfinished-sample rejection remain; continuation checks backend/dtype/source/environment. The new bounded preflight and research paths still await safe admission on this machine.
3. Proposed first budget: **two problems × two checkpoints × one seed**, 8,192 output tokens each, 32,768 total, 30 minutes cumulative wall time. Freeze the two inspected problem IDs and order before generation. This is a proposal, not an executed/predeclared study or a promise of completion on shared memory.
4. Recheck loaded tensor digests, thinking template, short seed repeatability and cache predictions at start. Check intermediate input lengths before relying on a large output cache. Stop on OOM/non-finite logits, identity/source mismatch, scoring failure or budget exhaustion. Keep partial trajectories and missing pairs visible.
5. Export paired correctness, EOS/cap, repetition, input/output counts, raw/source hashes, first-token/decode timings and honest MPS current-memory snapshots. Manually distinguish repetitive reasoning from legitimate structure. Publish negative results with the same care.

The milestone succeeds when complete or explicitly interrupted evidence is reviewable. A broader claim needs a later fresh set/multiple seeds, matched settings and all regressions. Transformers/MPS and the publisher's vLLM/H200/32K setup remain different experiments. Do not tune on these samples and then call them held out.

## LFM3: evaluate a narrower effect before another broad search

Persona-direction sweeps, sparse profiles and joint escalation already failed to establish transferable characters. L04 shortening failed its 10% fresh criterion (6.81%); L20's fresh completion repair failed. Stronger pushes are not the default next step.

First create a modest versioned arithmetic/format/completion baseline with separate correctness and format grades. Then predeclare one mechanism question, such as prefill-only versus decode-only versus all-position offsets. Recalibrate on MPS using a small neutral set; Windows amplitudes are not calibrated Mac strengths. Include unsteered, prompted personas, same-layer random controls matched for achieved strength, neutral/removal checks and fresh prompts. Show tone/content/structure/damage separately. Keep an old cookie example only as a labeled anchor. Phase-placement options are planned, not implemented in the current all-position hooks.

## Workbench, provenance and performance

| Priority | Action | Completion evidence |
|---|---|---|
| Near term | Privately transfer desired Windows runs/direction tensors for historical raw browsing | Verify archive hashes; preserve original directories. Public review works without transfer. |
| Near term | Cooperative stop and deliberately specified continuation before long campaigns | Interruption tests during generation/scoring/writes; distinguish saved tokens from actual KV/RNG restoration. Process-tree termination exists; exact trajectory continuation does not. |
| When an isolated browser is available | Visual/keyboard/responsive QA | Inspected screenshots and interaction outcomes on an isolated surface. No native desktop control. |
| After small memory trials | New Mac context-performance protocol | Frozen inputs/budgets and repeated timings; shared memory is not dedicated VRAM. Do not rerun CUDA 128K/allocator designs on MPS. |
| Later | Qwen batch launch UI, stronger evidence validation, reusable runner orchestration | Model-aware allowlisted API, complete schedule/provenance checks and meaningful regressions; keep greedy chat separate. |
| Backlog | Qwenputer/exact computation, LFM family transfer, permanent weight experiments | Specific protocols and separate authorization/documentation where required. |

Work directly on main, push reviewable milestones, and update README/ROADMAP/JOURNAL. Weights, environments, raw outputs and private conversations stay local. Original code/docs are MIT; upstream model terms remain separate.
