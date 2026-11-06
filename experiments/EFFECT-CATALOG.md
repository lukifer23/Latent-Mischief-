# Effect catalog

Descriptions follow observations. No validated personality neurons yet.

| Candidate / intervention | Observed behavior and prompt scope | Damage / no-effect | Replication status | Evidence |
|---|---|---|---|---|
| Persona-derived residual directions, strong doses | Invented words and repetitive fragments on several probes | Substantial degradation; no usable character recipe | Two exploratory sweeps; very few random controls | LFM-002-stronger-dose.md |
| L03N5238, gain 0 | Disagreement reply became example-based and more conversational on one prompt | Door story hit the cap; other effects vary by prompt | Single pilot observation | MAP-001-effect-first-pilot.md |
| L10N0204, gain 2 | Original evening anchor shortens from 171 to 72 tokens at a 192-token budget | No fresh completion improvement; think-boundary markers increase from 2/6 to 4/6 | Anchor effect reproduced; general completion interpretation unsupported on six fresh prompts | MAP-003-completion-replication.md |
| L22N2006, gains 0 and 2 | All three prompt outputs token-identical to baseline | No detectable token effect in this pilot | One seed and three prompts only | MAP-001-effect-first-pilot.md |

## Adding an observation

Include the exact model revision, operator/layer/channel, intervention kind and gain/dose, affected positions, run ID, prompt IDs and comparison. Separate tone/content/structure/damage; record unchanged outputs too. Use 'single observation', 'repeated within discovery', 'replicated on new prompts' and 'failed replication' explicitly. A replication label requires a new test, not a second reading of the same output.
