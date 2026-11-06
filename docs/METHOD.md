# Research method

Current priority is [open exploration](EXPLORATION.md): systematically sample changes and characterize their effects before pursuing a named persona. The initial targeted experiments below remain documented history and possible later applications.

## First questions

Can an internal intervention cause hamster characterization or theatrical villain behavior on prompts without persona instructions? Can we keep task performance usable? Does a direction work across seeds, unfamiliar prompts, dialogue turns and doses?

## Mechanism

Pinned LFM3 MLP: `w2(silu(w1(x)) * w3(x))`. The decoder applies `post_ffn_norm` before adding the MLP output to the residual. A gain hook at `feed_forward.w2` input changes the actual channel activation before both w2 and that norm. An additive residual hook after the norm is a different intervention. Record location, token positions and dose for every comparison.

## Stages

1. GPU / source / model load, baseline and gain=1 identity check.
2. Generate persona-conditioned discovery responses; derive contrast directions with explicitly documented token selection.
3. Sweep small doses and layers with matched random-direction controls.
4. Narrow to candidate channels and compare additive pushes, gain and ablation.
5. Freeze recipe; use separately authored held-out prompts and capability checks.
6. Only then consider checkpoint patches and cross-family-member transfer.

## Scoring

Score topic intrusion, self-identification and persistent characterization separately. For villain behavior, separate melodrama, arrogance, hostility and refusal. Track exact-answer correctness, format validity, repetition, language mixing, output length, EOS completion and latency. A token-cap hit is not automatically a reasoning loop.

Compare with an unsteered baseline, ordinary persona system prompt and matched random interventions. Store complete outputs, including failures. Persona discovery prompts are not a held-out evaluation set. Labels such as 'hamster neuron' remain shorthand hypotheses until validated across contexts.
