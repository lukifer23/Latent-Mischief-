# Effect catalog

Descriptions follow observations. No validated personality neurons yet.

| Candidate / intervention | Observed behavior and prompt scope | Damage / no-effect | Replication status | Evidence |
|---|---|---|---|---|
| Persona-derived residual directions, strong doses | Invented words and repetitive fragments on several probes | Substantial degradation; no usable character recipe | Two exploratory sweeps; very few random controls | LFM-002-stronger-dose.md |
| L03N5238, gain 0 | Disagreement reply became example-based and more conversational on one prompt | Door story hit the cap; other effects vary by prompt | Single pilot observation | MAP-001-effect-first-pilot.md |
| L10N0204, gain 2 | Original evening anchor shortens from 171 to 72 tokens at a 192-token budget | No fresh completion improvement; think-boundary markers increase from 2/6 to 4/6 | Anchor effect reproduced; general completion interpretation unsupported on six fresh prompts | MAP-003-completion-replication.md |
| L22N2006, gains 0 and 2 | All three prompt outputs token-identical to baseline | No detectable token effect in this pilot | One seed and three prompts only | MAP-001-effect-first-pilot.md |
| L10N4678 gain 0; L10N2998 gain 2 | Both yielded the same example-based disagreement response | Other prompts also changed; not a unique unit-specific trait | Shared branch on one discovery prompt; no fresh replication | MAP-004-expanded-sample.md |
| L03N7582 gain 2 | Evening response dropped the think boundary and continued describing the scene | Still capped; exceeded the requested three sentences | Single discovery observation; not completion repair | MAP-004-expanded-sample.md |
| L22N2727 gain 0 | One door scene introduced blood/ozone imagery | Door lost EOS within budget; two other prompts unchanged | Single fiction example; no sinister-personality label warranted | MAP-004-expanded-sample.md |

## Calibrated observations

Latest calibrated observations (all exploratory; see [MAP-005](MAP-005-calibrated-effects.md)):

| Candidate / intervention | Observed effect | Collateral changes / limits | Status |
|---|---|---|---|
| L03N6960 positive offset, local target 20% | Spanish greeting loses repeated answer/think boundary | 19+24 changes from 43 to 49; boundary still appears elsewhere; strict total stays 4/6 | Single-prompt cleanup with regression; no general fix |
| L04N5848 positive offset, local target 20% | Seven of ten behavioral replies shorter; direct uncertainty reply | Three longer; astronaut now capped; total EOS remains 15/16 | Compression candidate, fresh replication pending |
| L20N0472 positive offset, local target 20% | Rainy-afternoon reply completes in 98 tokens; all 16 reach EOS | Only one baseline completion failure; no fresh validation; strict checks unchanged | Narrow completion/length candidate |
| L03/L10/L15/L20 selected negative offsets | Same arithmetic input returns 49 rather than 43 | Shared alternate answer across multiple units; not a unique neuron function | Recorded factual regression |
| L22N0450 either sign | All six exact-check sequences unchanged, despite measured local changes | Behavioral replies vary; negative offset adds a cap failure | No detected exact-check effect; no persona claim |

## Adding an observation

Include the exact model revision, operator/layer/channel, intervention kind and gain/dose, affected positions, run ID, prompt IDs and comparison. Separate tone/content/structure/damage; record unchanged outputs too. Use 'single observation', 'repeated within discovery', 'replicated on new prompts' and 'failed replication' explicitly. A replication label requires a new test, not a second reading of the same output.
