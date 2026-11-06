# LFM-002 / LFM-003: persona discovery

Status: planned; depends on LFM-001 baseline and hook checks.

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

No intervention result yet. Baseline infrastructure is being established; this specification is not an observation.
