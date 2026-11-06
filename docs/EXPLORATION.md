# Change first, characterize afterward

## Research direction

The user requested an open exploration phase: make controlled internal changes, record the resulting behavior, and use those observations to design targeted changes later. Hamster and villain remain possible demonstrations, not the discovery objective.

## Sampling plan

Start with a reproducible, modest sample of MLP channels across early, middle and late layers, including convolution and attention blocks. Choose candidates before seeing their output; retain no-effect results. Expand coverage only after validating the pipeline and recording the cost.

For each candidate, compare neutral gain=1 against ablation (gain=0) and amplification (gain=2). These are changes to the channel's actual gated activation before w2 and post-FFN normalization, not a permanent weight edit. Sign inversion and additive residual pushes are separate later experiments; do not pool their dose scales.

Use the same neutral system instruction, prompts, greedy settings and token budget for baseline and altered conditions. Record seed, layer, operator type, channel, gain, hook location, token positions, raw token IDs, outputs and timing. Initially change one channel at a time. No persona instruction is supplied during generation.

## Describe before naming

Review output pairs in shuffled condition order where practical. Record dimensions separately:

- Tone: warmth, formality, assertiveness, theatricality, apologizing.
- Content: topic intrusions, identity claims, recurring imagery.
- Structure: length, formatting, reasoning traces, dialogue patterns.
- Damage: factual errors, incoherence, language mixing, repetition, failure to finish.
- No detectable effect: keep it as an outcome, not a discarded trial.

Avoid naming a 'personality neuron' from one example. Labels describe observed effects under a specific intervention and prompt set. A unit may have several effects or no stable interpretation. Automated word counts and repetition measures help browse evidence; they are not labels or quality judgments.

## From observation to targeted editing

After a candidate shows a repeatable effect, freeze a tentative description and test new prompts, multiple seeds, turn histories and smaller doses. Include task checks and random-channel comparisons. Only then try combining candidates or designing a specific persona. Document failed replications and collateral changes.

The first two persona-direction runs remain historical evidence, including their failures. This change of strategy does not reinterpret damaged outputs as successful discoveries.
