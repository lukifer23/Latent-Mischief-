# CAL-001: calibrated offset mapping

Status: predeclared; no results yet.

## Milestones and scope

1. Measure gated MLP activation RMS for all 184,320 channels on 12 neutral calibration inputs.
2. Uniformly sample 16 channels in each of six layers (96 total). Calibrate positive and negative constant offsets toward 5% and 20% mean relative change of that block's post-normalized MLP output, capped at 32 times the channel's calibration RMS.
3. Screen the 384 interventions on four calibration inputs, recording actual BF16 local output change and next-token KL(P_baseline || P_changed). This is not a generation/personality score.
4. Freeze one candidate per layer using only screen measurements. Prefer the larger target; require both signs to meet ±25% target tolerance, no cap hit, and each sign's mean KL <=0.5 nats. Choose the lower median candidate ranked by paired mean KL. Fall back to 5% or skip a layer if no candidate qualifies. Publish all screen outcomes, including ineligible cases, in the local evidence.
5. Evaluate both signed offsets for the frozen candidates on 16 different prompts covering six exact checks and ten behavioral domains. At six selected units, this is 16 baselines + 2 neutral/cleanup checks + 192 interventions = 210 generations. Fewer selected units reduce the count explicitly.

## Definitions and controls

Calibration uses non-special prompt tokens except position zero, including the neutral system instruction. RMS pools observed token positions; it is not a population estimate over arbitrary language. Native channel offsets are applied before w2 and post_ffn_norm, to all prompt and generated-token positions. Float32 local calibration is an approximation; the screen measures the actual BF16 forward effect. A 20% MLP-output change is not a 20% change in the full residual stream or behavior.

The 96 channels are selected before outcomes using seed 20261106. Signed and strength pairs expose dose sensitivity. Random sampled units form a reference cohort, but choosing a screen-median candidate is a selection procedure; generation results do not become independent random-unit estimates. Candidate selection never uses generated character labels or evaluation outputs. Greedy generation is a single trajectory per condition, not multi-seed validation.

The calibration inputs and the 16 evaluation prompts are disjoint. The latter will be inspected for effect descriptions and are therefore exploratory, not a future held-out validation set. Six exact checks measure narrow task/format behavior only; qualitative correctness for the other ten prompts requires review. Gains and calibrated offsets are separate intervention families.

## Evidence and deliverable

Keep all activation statistics, offsets, achieved effects, per-prompt screen rows, selected candidates, neutral checks, generation schedules, complete outputs and code/config snapshots locally. Create a repository report with reviewed aggregate figures and representative successes/failures. Preserve the current workbench and make calibrated results browsable. No weight training or permanent checkpoint change.
