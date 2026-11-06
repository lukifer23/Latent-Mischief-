# LFM-002 / LFM-003: persona discovery

Status: first discovery run complete. See [findings](LFM-002-first-sweep.md); no validated internal persona recipe yet.

## Questions

Can we induce hamster characterization or a theatrical villain persona through reversible internal interventions on LFM3-1.7B? Do the two effects combine? Does ordinary task accuracy survive at useful doses?

## Initial discovery design

- Use the post-trained text checkpoint first. Keep weights unchanged.
- Generate paired ordinary / persona-conditioned responses to the same discovery prompts.
- Candidate direction: mean persona-minus-neutral residual activation, with token selection and averaging recorded. Prefer response tokens over the persona instruction itself, to reduce direct instruction-token artifacts.
- Inspect several layers before scanning all channels. LFM3 has 184,320 MLP channels (24 x 7,680), so exhaustive generation sweeps are not the first step.
- Sweep small normalized doses; compare with norm-matched random directions and a system-prompt persona baseline.
- Test prompts that omit any reference to hamsters or villains.

## Behavior definitions

Hamster: score topic intrusion, self-identification, and persistent characterization separately. A hamster word alone is weak evidence.

Villain: theatrical language, arrogance, melodrama and hostility are separate dimensions. Do not equate a villain role with changes in factual ability or refusal behavior.

## Stop / revise conditions

If the direction only copies persona vocabulary, revise token selection and discovery prompts. If coherence degrades as much as random controls, reduce dose or change location. If effects vanish on unfamiliar prompts, report limited transfer rather than a general persona control.

## Evidence

First run: `runs/20261106T182140177847Z-LFM-002-persona-discovery/`.

Implementation: scripts/personas.py. Four discovery prompts per persona including neutral (12 responses), with per-layer residual means extracted by teacher-forced replay on non-special reply-token positions. Each response receives equal weight. Candidate directions are the mean paired differences, normalized to unit length. Doses are fractions of mean neutral reply residual norm. This direction can capture content differences as well as style; successful transfer must be demonstrated separately.

Initial sweep: layers 7, 13 and 19; doses 0.05, 0.15 and 0.35. Hook location: decoder layer output after the block's residual addition. A push acts on the last position of every forward (the final prompt token during prefill, then each incremental decode token). No persona system instruction is supplied to steered probes. These exploratory probes are prompt-disjoint from discovery, but will be inspected here, so they cannot become the final held-out test.

Controls: unsteered baseline, persona system prompt, zero-dose additive hook, hook-removal check, and two seeded norm-matched random directions at each selected layer at dose 0.15. Two controls are a smoke-level comparison, not sufficient for a strong significance claim. No random-control conclusion applies to other doses yet.

Raw outputs, direction tensors, pairing diagnostics, token IDs, checks and source/config hashes are stored locally. Keyword indicators aid browsing only; they do not measure characterization quality.
