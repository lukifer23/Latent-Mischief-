# MAP-001: effect-first channel exploration pilot

Status: complete; pilot observations only, no replicated personality labels.
Date: 2026-11-06
Raw run: runs/20261106T184602567236Z-MAP-001-channel-exploration/

## Question

What observable behavior changes arise from ablating or doubling uniformly sampled MLP channels? This is a coverage/pipeline pilot, not a search for hamsters, villains or improved intelligence.

## Design

Six layers (3, 4, 10, 15, 20, 22), covering early/middle/late convolution, sliding-attention and full-attention blocks. One uniformly sampled channel per layer using Python Random seed 42, selected before generation. Gains 0 and 2 applied to the actual gated MLP channel activation before w2 and post_ffn_norm, at all prompt and generated-token positions.

All conditions use the same neutral system instruction, greedy settings and 128-token cap. Three prompts probe description, disagreement and scene writing. Baseline plus gain=1 and hook-removal checks precede 36 intervention replies; 41 total generations planned. Candidate identities and all source/input snapshots are recorded. This is not a final held-out set.

Runtime: patched PyTorch 2.14.1+cu130 and setuptools 84.2.1. Prior persona runs used the earlier runtime. The patched runtime passed a separate baseline/neutral-hook check before this pilot.

## Review rubric

Record tone, content, structure, damage and no-effect separately, with concrete examples. 'Changed token sequence' only means exact tokens differ; it does not establish a personality change. Retain capped or unchanged replies and failed interpretations. No inference across prompts or seeds until tested.

## Evidence

41 generations completed. Gain=1 and hook-removal checks passed. Of 36 intervention replies, 18 changed the exact token sequence and 18 were identical to the corresponding baseline. This is not a personality-change rate.

| Candidate | Operator | Changed / 6 intervention replies |
|---|---|---:|
| L03N5238 | Convolution | 5 |
| L04N0912 | Sliding attention | 4 |
| L10N0204 | Full attention | 4 |
| L15N6074 | Convolution | 3 |
| L20N2253 | Sliding attention | 2 |
| L22N2006 | Full attention | 0 |

Operator labels describe the block containing the MLP; the intervention itself is on its MLP channel, not its attention/convolution operator.

## Tentative observations

- L03N5238 at gain 0 changed the disagreement answer from a general statement about evaluating evidence to an example-based conversational response. That is a structure/tone observation on one prompt, not an established character trait.
- L10N0204 at gain 2 made the evening response finish after three descriptive sentences. Its baseline and ablation condition included explicit sentence-count checking, a `</think>` boundary and a restarted answer that hit the cap. This nominates a completion/self-check behavior for fresh testing; it does not establish a general reasoning control. At the same gain, the disagreement output was exactly unchanged and the door story changed.
- L22N2006 at gains 0 and 2 left all three prompts token-identical. This is a useful no-effect result within this test. It does not prove that the channel is inactive or irrelevant on other inputs.

## Limits and follow-up

One seed, three discovery prompts, one sampled channel per layer, no blinded review and no capability suite. Some replies include reasoning-like text despite requested non-thinking mode; the evening baseline is capped. Exact token changes may be minor lexical divergence and should not be labeled personality changes.

Keep the no-effect candidate and the completion candidate in the catalog. Next test new description, disagreement and scene prompts, compare intermediate gains, and check whether the apparent completion behavior replicates. Increase sampled-channel coverage without selecting only entertaining cases. All raw outputs and a randomized review-order file are preserved; current viewer review is not blinded.
