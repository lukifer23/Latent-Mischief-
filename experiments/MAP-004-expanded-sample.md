# MAP-004: expanded effect-first sample specification

Status: complete; broader output changes recorded, no stable personality labels established.
Raw run: runs/20261106T190341875119Z-MAP-004-channel-exploration/
Managed job: runs/jobs/20261106T190341741000Z-mapping/

Uniformly sample two additional MLP channels in each of layers 3/4/10/15/20/22, excluding all six original pilot channels. Fixed selection seed 1337. Gains 0/2, 128-token budget, same three neutral discovery prompts as MAP-001, and shuffled intervention order. No selection for a desired character or entertaining output.

12 candidates x 2 gains x 3 prompts = 72 intervention replies, plus 3 baselines and 2 neutral/cleanup checks = 77 generations. This expands coverage on already-inspected prompts; it is not a fresh-prompt replication. Retain all sampled candidates, including unchanged outputs. Assign only descriptive provisional labels after inspection, with prompt scope and damage checks.

Do not pool numerical changes from these old prompts with MAP-003's fresh replication cohort. Greedy decoding has no sampling-seed variation; the seed controls candidate/trial selection, not repeated independent response samples.

## Results

77 generations completed, exit code 0. Both hook checks passed. Eight archived source/input files and all 72 scheduled intervention trials verified. Of the 72 intervention replies, 40 changed exact tokens and 32 were identical to baseline. No-effect outputs were retained.

| Candidate | Block operator | Changed / 6 replies | EOS / 6 replies |
|---|---|---:|---:|
| L03N5060 | Convolution | 3 | 4 |
| L03N7582 | Convolution | 4 | 4 |
| L04N4369 | Sliding attention | 5 | 3 |
| L04N5816 | Sliding attention | 5 | 4 |
| L10N2998 | Full attention | 4 | 4 |
| L10N4678 | Full attention | 6 | 4 |
| L15N4798 | Convolution | 3 | 4 |
| L15N5995 | Convolution | 4 | 4 |
| L20N1357 | Sliding attention | 2 | 4 |
| L20N6351 | Sliding attention | 2 | 4 |
| L22N2727 | Full attention | 1 | 3 |
| L22N6754 | Full attention | 1 | 4 |

Operator describes the block containing the edited MLP channel. These are not direct attention/convolution interventions. Only two channels were sampled per layer; no depth-specialization conclusion follows.

## Descriptive observations

- The disagreement prompt changed in 5/24 intervention cases and produced five unique token sequences. Ablating L10N4678 and doubling L10N2998 produced the same example-based alternative reply. This shared output branch is a warning against assigning a unique personality function to either unit.
- The evening prompt changed in 15/24 cases, but none reached EOS within 128 tokens. Doubling L03N7582 removed the think-boundary marker while continuing into a longer description beyond the requested three sentences. Marker removal therefore did not imply better completion or instruction following.
- The door prompt changed in 20/24 cases and produced 16 unique sequences. Most differences concerned characters, imagery and narrative details. Ablating L22N2727 introduced blood/ozone imagery on this one scene and lost EOS within the budget. Its other two prompts stayed identical. Do not label this a sinister-personality control from one fiction sample.
- Total EOS completions were 46/72. The baseline completed disagreement and door but capped on evening; two door conditions additionally capped (L04N4369 gain 0, L22N2727 gain 0). No baseline EOS failure was repaired.

## Limits and next work

These are three already-inspected discovery prompts, one greedy decode per condition, sparse random coverage, no blinded review and no broad capability grading. A changed sequence may be ordinary decoding divergence. Retain lexical/structural observations and completion damage in the catalog; no stable personality trait has been established.

Next diversify neutral prompts and calibrate intervention size against activation/output changes before expanding to much larger sweeps. Compare additive offsets with natural-activation gain as separate intervention families. Replicate any proposed tone/content description on new prompts before naming an effect.
