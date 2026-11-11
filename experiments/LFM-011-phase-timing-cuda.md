# LFM-011: LFM-010 phase timing on native Windows CUDA

Status: **implemented and verified offline; research run pending**. No LFM-011 model run exists yet. LFM-010 remains a separate macOS/MPS protocol whose only attempt was a zero-output resource refusal; nothing here relabels or pools it.

## Why this exists

LFM-010 was frozen for MPS and refused at admission on the Mac (6.395 GiB available vs a 7 GiB floor). The Windows machine has an NVIDIA RTX 2000 Ada (16 GB), so the same scientific question can be tested here without waiting for Mac memory. Hypotheses, frozen direction, prompts, conditions, schedule seed, token budgets, scoring and review rubric are **identical** to [LFM-010](LFM-010-phase-timing.md); read that report for the full design, phase definitions and decision rules. This is a backend replication of the design, not a new hypothesis.

## What differs (and why it is not a silent change)

| Item | LFM-010 | LFM-011 |
|---|---|---|
| Backend | macOS MPS/BF16 | native Windows CUDA/BF16, eager attention |
| Calibration | recalibrated to 5% on MPS | recalibrated to 5% on CUDA; same ±25% tolerance, same uncapped requirement |
| Admission | macOS pressure level, 7 GiB available, MPS driver ceiling | no OS pressure level exists: 12 GiB available RAM at start (6 running), 8 GiB free GPU memory at start (3 running), 7 GiB CUDA reserved ceiling, allocator fraction 0.5, RSS ≤ 8 GiB, swap growth ≤ 256 MiB, disk ≥ 5 GiB |
| Thermal | `pmset` warnings | `nvidia-smi` temperature ≥ 85 C or any thermal/power throttle reason active; unknown telemetry fails closed |
| Process priority | nice +10 | below-normal priority class (no admin) |

The frozen direction's channel indices and native components come from the published CHAR-001 artifact (originally measured on Windows/CUDA), but amplitudes are **not** carried over: both vectors are recalibrated here before generation. The pinned model revision and the same wall/token ceilings apply. 54 generations, at most 48 new tokens each, 2,592 generated tokens total, eight-minute cumulative wall ceiling, no automatic retry, no dose escalation, no exact KV/RNG continuation. A greedy CUDA trajectory may differ from an MPS one for the same inputs; the two backends are never pooled.

## Commands

```sh
uv run --locked python scripts/phase_experiment.py plan --config configs/lfm-phase-timing-cuda.json
# Only after committing source/config/protocol:
uv run --locked python scripts/phase_experiment.py run --config configs/lfm-phase-timing-cuda.json
uv run --locked python scripts/phase_report.py RUN-ID
uv run --locked python scripts/phase_report.py RUN-ID --blind-review
```

Reviewed metrics publish to `experiments/evidence/LFM-011/` only after human ratings exist. Until then the report states **manual review required**; no persona verdict is inferred from vocabulary hits.

## Evidence status

- Implemented: backend-aware runner/report/guard (default MPS behavior unchanged), CUDA resource policy, CUDA thermal parser.
- Verified offline: unit tests for the CUDA policy and thermal parsing (see JOURNAL).
- Not yet verified: CUDA calibration within tolerance, phase neutrality on real LFM3 weights, the 54-reply campaign, any behavioral result.
