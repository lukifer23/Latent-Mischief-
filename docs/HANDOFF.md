# Device handoff — 2026-11-06

**2026-11-09 update:** Mac setup and short MPS inference are verified. Current priority is [LFM-010 phase timing](../experiments/LFM-010-phase-timing.md), implemented/offline tested; its first admission refused before loading weights at 6.395 GiB estimated available (0/54 replies). MPS research execution still requires normal pressure and 7 GiB available inside the initialized runner. QWN-003's first admission refused before loading weights; it stays separate. All project services remain stopped. Read [the review](REVIEW-2026-11-09.md), [current setup](SETUP.md), [BACKEND-001](../experiments/BACKEND-001-mac-setup.md) and [next steps](NEXT-STEPS.md). The interrupted Windows state below remains unchanged. Raw Windows runs/transfer ZIP were absent from this clone; no historical raw regrading was claimed on Mac.

## Read this first

The user asked to stop because they were leaving. **No inference should still be running.** Native desktop/browser control is prohibited: it disrupted their concurrent work. Use project files, CLI/API checks or a genuinely isolated browser surface. No admin privileges, paid compute, new branches or force pushes. Work on `main`; push reviewable milestones. See AGENTS.md.

Start with this file, README.md, QWN-002's report and docs/NEXT-STEPS.md. Do not treat the old unchecked roadmap as evidence that the evaluator is still missing.

## Actual stopping point

- A real thinking-on sampled Qwen batch evaluator, output-only presence penalty and Windows-safe Math-Verify 0.9.0 grader are implemented. Existing greedy chat remains separate. Protocol/source were pushed as **aedb63b before generation**.
- Full CPU checkpoint audit: 488 tensors checked; only two expected MLP tensors differ, 64 columns each. 131,072 positions addressed, **130,188 bitwise changed**. Expected edited digests match. Evidence: `experiments/evidence/QWN-002/weights.json`.
- Run **20261106T213644088335Z-QWN-002-anti-loop-pilot** was stopped by user request after **6/8 complete trajectories**. A seventh, baseline level 3, has **832 partial tokens** preserved; the eighth was never started. Approximate last checkpoint elapsed time: 904 seconds. No completed-study summary exists.
- There are only **two complete matched pairs**: levels 1 and 2. Both base and edit capped at 4,096 without EOS or a final boxed answer. Edited levels 3/4 also capped, but their baseline counterparts are unavailable. Do not compare unmatched 0/4 versus 0/2 as evidence of a capability difference.
- Repeated token 4-gram fractions decreased on the two paired samples (59.17%→39.16%, 44.76%→39.97%). This is a descriptive surface measure; no completion/final-answer benefit or elimination of reasoning loops was established. Upstream used up to 32K tokens and different kernels/sampling implementation.
- All six saved rows were independently regraded; 19 archived source files and both complete pairs' actual input-token equality verified. Report: `experiments/QWN-002-anti-loop-pilot.md`; numeric evidence/figure in `experiments/evidence/QWN-002/`.

The interrupted trajectory **cannot be resumed exactly**: KV state and RNG state were not checkpointed. The runner deliberately rejects rerolling an unfinished sample. A new device/backend/dtype also changes the experiment. Preserve this interrupted run; use a separately identified follow-up rather than overwriting it or pooling incompatible results.

## What transfers through Git

Code, configs, dataset/model revisions, selection rules, audit digests, numerical results, figures and reports are committed. **Raw runs, model weights, local caches and private chats are ignored.** Cloning alone does not provide full raw replies/history.

A local transfer bundle is prepared at `.cache/handoff/Latent-Mischief-QWN-002-handoff.zip`. It includes only this benchmark run, its archived inputs/code, the Qwen audit and selected problem manifest—no model weights, credentials or private chats. Copy it privately to the new device and extract into the cloned repo if raw inspection is needed. It is not uploaded or committed. Without it, public numeric evidence is enough to understand the status and build the next implementation; full raw-output reproduction requires the pinned dataset/models and a new labeled run.

## Mac: original limitations, superseded by BACKEND-001

At the original handoff inference was Windows/CUDA-specific. BACKEND-001 adds platform-aware dependencies and a shared adapter, then verifies short MPS/BF16 inference. Long MPS campaigns and full CPU checkpoint inference remain unverified. The dependency-light review command below remains supported.

With existing Git and user-installed uv, clone the repository and enter it:

```sh
git clone https://github.com/lukifer23/Latent-Mischief-.git
cd Latent-Mischief-
# Skips the CUDA project environment; installs only the small review dependency:
uv run --no-project --python 3.11 --with psutil==7.2.2 python scripts/viewer.py --read-only
```

Open the printed loopback address yourself. This mode uses standard-library HTTP plus psutil, not torch. All POST routes reject inference and writes. Public recipe cards are available; raw comparisons require transferred runs. Stop with Ctrl+C. HTTP/API/DOM checks now also pass on Mac; **visual browser rendering remains unverified**. Reports and figures are readable without Python.

## Original Mac implementation checklist

Steps 1–3 below are complete for BACKEND-001's short tested scope. Step 4 remains planned. The adapter retains CUDA metrics, records actual backend/dtype and marks unavailable MPS peaks as null. Frozen Windows protocols reject MPS rather than silently changing experiment identity.

1. Detect architecture/RAM and existing tools (`uname -m`, `sysctl -n hw.memsize`, `uv --version`). Keep installs user-local. Do not assume Apple Silicon, MPS availability or BF16 support.
2. Make torch sources/lock platform-aware while retaining the verified Windows CUDA path. Introduce an explicit device/dtype adapter, including synchronization and memory metrics. Record unsupported metrics as unavailable, never invented CUDA equivalents. Retain current CUDA behavior and tests.
3. Validate loading, EOS, thinking template, seeded sampling and cached versus full-prefix logits on the chosen Mac backend. At handoff the cache diagnostic was unexecuted; BACKEND-001 now runs it on MPS for both families and verifies loaded Qwen tensor digests. Short numerical differences remain recorded.
4. Only then predeclare a separately named Mac pilot/budget follow-up. The open question is whether a larger output budget/backend changes these incomplete 4K outcomes. Keep paired prompts/settings matched and report missing samples. Do not claim an anti-loop win from the short arithmetic smoke tests.

Other outstanding work: Qwen batch launching from the UI is **not implemented** (CLI works); visual QA remains incomplete; exact KV+RNG continuation needs deliberate implementation. QWN-003/LFM-010 now implement cooperative resource/manual stop with raw-token retention. The historical Windows STOP was targeted worker termination preserving the last 64-token checkpoint, not a verified graceful resume system.

## Regression work and verification

Adding the optional grader changed the full lock hash without changing inference dependencies. `runtime_identity.py` now compares the archived and current locked default/transitive inference graph plus installed versions before calibration reuse. Actual CAL-001 matched all 56 inference dependency entries despite the optional extras; real core changes still reject. Full lock hashes remain recorded, and strict experiment-resume rules are unchanged. No fresh 210-generation LFM re-evaluation was claimed.

Checks at the original handoff: **43 Python tests**, Node comparison/startup/model-isolation/math-label checks and dependency-light HTTP rejection checks passed. No final cache diagnostic or LFM GPU replay was run after that stop request. On 2026-11-09 the separate BACKEND-001 milestone adds Mac compatibility evidence and 51 passing tests; historical LFM findings remain unchanged: no validated evil/hamster persona, Windows 32K performance completed, Windows 128K OOM.

Suggested next-agent prompt: “Read AGENTS.md, docs/REVIEW-2026-11-09.md, docs/NEXT-STEPS.md and experiments/LFM-010-phase-timing.md. LFM-010 is implemented/offline verified; real MPS execution requires its unchanged admission policy. Keep services stopped unless needed/requested. Preserve the zero-trajectory QWN-003 refusal and interrupted Windows evidence separately; no rerolls, pooling, desktop control, admin actions or paid compute. Keep raw/private data local and use main only.”
