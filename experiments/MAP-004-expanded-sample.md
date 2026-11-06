# MAP-004: expanded effect-first sample specification

Status: predeclared; outcome pending.

Uniformly sample two additional MLP channels in each of layers 3/4/10/15/20/22, excluding all six original pilot channels. Fixed selection seed 1337. Gains 0/2, 128-token budget, same three neutral discovery prompts as MAP-001, and shuffled intervention order. No selection for a desired character or entertaining output.

12 candidates x 2 gains x 3 prompts = 72 intervention replies, plus 3 baselines and 2 neutral/cleanup checks = 77 generations. This expands coverage on already-inspected prompts; it is not a fresh-prompt replication. Retain all sampled candidates, including unchanged outputs. Assign only descriptive provisional labels after inspection, with prompt scope and damage checks.

Do not pool numerical changes from these old prompts with MAP-003's fresh replication cohort. Greedy decoding has no sampling-seed variation; the seed controls candidate/trial selection, not repeated independent response samples.
