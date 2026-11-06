# LFM-002 / LFM-003: first persona-direction sweep

Status: complete; no convincing internal persona transfer observed in the tested range.
Date: 2026-11-06
Raw run: runs/20261106T182140177847Z-LFM-002-persona-discovery/
Specification: [persona discovery](LFM-002-persona-discovery.md)

## What ran

Pinned LFM3-1.7B BF16 on native Windows, unchanged weights. Greedy generation, requested non-thinking mode, checkpoint repetition penalty 1.1, 96 new-token cap. Four paired discovery prompts produced neutral, hamster and theatrical-villain responses. Response-token mean differences defined candidate residual directions. Pushes were applied at decoder outputs on the last position of every forward.

95 total generations: 12 discovery, 3 unsteered probes, 2 hook checks, 6 persona-prompt probes, 54 contrast-direction probes and 18 random-direction probes. Selected layers 7/13/19 and doses 0.05/0.15/0.35. Two random directions per selected layer at dose 0.15 only.

## Observations

- Zero-dose and hook-removal checks both matched the baseline token sequence on the desk prompt.
- Explicit hamster system prompts produced recognizable hamster characterization on all three exploratory probes; all three reached EOS.
- None of the 27 hamster-direction probe replies contained the small predeclared hamster keyword set. Inspection did not establish persistent hamster characterization.
- Villain direction pushes changed wording and imagery, but did not establish convincing persona transfer. The villain keyword set also missed obvious theatrical character in some explicitly prompted replies, so a zero keyword count is not a valid villain-quality score.
- Every unsteered probe, contrast-direction probe and random-direction probe hit the 96-token budget without EOS. This budget censors completion and length comparisons; it does not establish loops.
- The unsteered Moon explanation was factually wrong. Some changed conditions also gave wrong explanations. These three prompts are not a capability benchmark, and no accuracy improvement is claimed.

## Interpretation

The pipeline changes model outputs and its neutral hooks behave as expected. The first low/moderate-dose directions did not provide a useful persona recipe. Explicit persona prompting is currently the stronger demonstrated method.

This does not show that LFM3 cannot be steered. Candidate extraction, hook placement, layer choice, dose and prompt selection remain open variables. Differences in generated reply content can contaminate a persona direction. Some discovery responses were truncated or included apparent reasoning text despite requested non-thinking mode. These are important confounds.

## Follow-up

Try a bounded stronger-dose sweep (0.7/1.0 at layers 13/19) with a 192-token cap, retaining a random comparison at dose 0.7. This changes both budget and strength, so any improvement requires later controlled comparisons to isolate the cause. Then consider response-position selection and prefill-versus-decode placement before exhaustive neuron searches. Preserve this negative result.
