# LFM-002 / LFM-003: stronger-dose follow-up

Status: complete; substantial degradation, no usable hamster/villain recipe.
Date: 2026-11-06
Raw run: runs/20261106T182953749085Z-LFM-002-persona-discovery/
Managed job: runs/jobs/20261106T182951911844Z-personas/

## Setup

Same checkpoint, BF16, greedy decoding and last-position decoder-output hook. Layers 13/19, doses 0.7/1.0, 192 new-token cap. One norm-matched random direction per selected layer at dose 0.7. Candidate directions were re-derived from new 192-token discovery responses, so this is not a clean dose-only comparison with the first run. No final held-out test has been used.

53 generations: 12 discovery, 3 unsteered probes, 2 hook checks, 6 persona-prompt probes, 24 contrast-direction probes, 6 random-direction probes. The managed job completed with exit code 0. Zero-dose and hook-removal token checks passed.

## Observations

Increasing strength produced garbled invented words and repetition in many contrast-direction outputs. For example, the layer-19 hamster direction at dose 1.0 devolved into repeated 'wigg' / 'sque' fragments. This is damage, not successful hamster characterization.

| Condition | Layer | Dose | Replies | Mean repeated token 4-gram fraction |
|---|---:|---:|---:|---:|
| Unsteered baseline | — | — | 3 | 0.35% |
| Hamster direction | 13 | 0.7 | 3 | 17.99% |
| Villain direction | 13 | 0.7 | 3 | 24.34% |
| One random direction | 13 | 0.7 | 3 | 0.35% |
| Hamster direction | 19 | 0.7 | 3 | 42.50% |
| Villain direction | 19 | 0.7 | 3 | 28.75% |
| One random direction | 19 | 0.7 | 3 | 0.88% |
| Hamster direction | 19 | 1.0 | 3 | 91.71% |

There is **no matched random control at dose 1.0**. At dose 0.7 only one random vector per layer was tested; the three replies reuse that vector and are not three independent vector samples. No statistical significance or generality claim is justified.

All 24 contrast-direction probes hit the budget without EOS. All six persona-prompt probes reached EOS; the three hamster-prompt probes contained hamster terms. No internal hamster probe contained the predefined hamster terms. These counts alone do not measure persona quality.

## Metric and limits

Repeated four-gram fraction is `1 - distinct consecutive 4-token sequences / total consecutive 4-token sequences`. Legitimate repeated structure can increase it; it is not a validated loop classifier. Here the full outputs also show obvious repetition and degradation. All unsteered probes also reached the 192-token cap, so length is censored. One random-control reply reached EOS exactly at the cap and is recorded as both an EOS completion and a cap hit.

The original no-system-prompt baseline differs in instruction content from the explicitly prompted personas. An ordinary neutral system prompt belongs in the next probe suite so we can distinguish persona effects from generic instruction/length effects.

## Interpretation and next test

These candidate directions can strongly perturb generation, but a useful characterization has not transferred. Increasing strength is not the next default strategy. Next isolate cleaner final-answer features, use a neutral-system-prompt baseline, and compare hook placement and selected MLP-channel contributions at smaller doses. Increase random-vector count before interpreting specificity.

Recompute the table with `uv run --locked python scripts/summarize.py 20261106T182953749085Z-LFM-002-persona-discovery`. Exact source/input snapshots matching the recorded hashes have been archived beside both persona runs. The local workbench can compare all their complete outputs and record human notes.
