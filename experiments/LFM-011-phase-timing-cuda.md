# LFM-011: LFM-010 phase timing on native Windows CUDA

Status: **run complete (54/54), numeric evidence published, blinded human review pending; direct reading suggests a negative result**. LFM-010 remains a separate macOS/MPS protocol whose only attempt was a zero-output resource refusal; nothing here relabels or pools it.

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

## 2026-11-11: results (run `20261111T162206863981Z-LFM-011-phase-timing`)

Predeclared at `b1eac9f`; start-up fixes `b1eac9f`..HEAD below (runtime code only; protocol, prompts and limits unchanged). **Complete: 54/54 replies, 1,532 generated tokens, exit 0.** Verified: archived sources, full frozen schedule, per-phase forward traces, calibration tolerance (villain 5.02%, random 5.01% mean actual prefill change, uncapped; amplitudes 6.01 and 5.93), all six zero/restored checks token-identical to baseline, parameter versions unchanged. Mean generation 0.88 s; peak CUDA reserved 3.26 GiB.

| Condition (6 replies each) | Changed vs baseline | EOS | Answer / strict (3 checks) |
|---|---:|---:|---:|
| baseline | - | 4 | 2 / 2 |
| prompted villain | 4 | 3 | 3 / 3 |
| villain prefill / decode / both | 1 / 2 / 2 | 4 / 3 / 3 | 2/2, 2/2, 2/2 |
| random prefill / decode / both | 2 / 2 / 1 | 4 / 4 / 4 | 2/1, 2/2, 2/2 |

**Observations (not hypotheses).** At this 5% local strength the internal profiles changed only a few replies, mostly small wording shifts ("let's" to "let us", a different continuation of the same sentence). Direct reading of the changed replies found no sustained villain voice; the matched random vector changed about as many replies. Decode-only and both-phase villain produced identical text on the changed prompts, so decode placement carried the effect here. The robot-sandwich reply lost its baseline EOS under villain decode/both and random prefill (26 to 42-48 tokens, plus a continuation about returning the sandwich). One random-prefill reply gained a stray `</think>` marker and lost strict format on the gold-symbol check. The baseline already fails the addition check (23+18), so preservation evidence there is empty; no steered condition fixed it. The ordinary prompted villain visibly characterizes and passed all three checks.

**Decision status.** The formal blinded 0-2 review has **not** been done: a pack and key exist locally (`blind-review.json`, `review-key.json`) and numeric evidence is published at `evidence/LFM-011/metrics.json` with review marked required. On direct reading, no phase appears to meet the nomination rule (two of three villain replies sustaining character at 2/2/2), so the expected outcome is a **negative/no-effect result**, pending human ratings. This is one greedy trajectory per cell on three scenarios; it provides no significance estimate and says nothing about stronger doses.

## Start-up failures before the run (preserved, 0 replies each)

Three earlier attempts in this session never generated text: `...162034855054Z` (Windows `psutil.disk_usage` rejected a `Path`; fixed `str(root)`), `...162056191561Z` (CUDA flag missing from the frozen resource policy so the macOS branch ran; flag moved into the config and the test now uses the real config), `...162121148383Z` (weights loaded, but the pinned generation config leaves `num_beams` unset, which Hugging Face treats as 1; check relaxed to None-or-1). None involved resource refusal or model output, and no limit was loosened. The completed run used the same frozen config, prompts and token budgets throughout.
