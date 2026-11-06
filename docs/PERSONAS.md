# Naming, exact recipes, and observed personas

A desired character and an observed persona are different fields. Extracting a profile from villain text does not make the edited model a villain. Every tested change gets a stable exact-recipe ID and neutral codename, including failed and no-effect changes. A reviewed nickname can then describe observed behavior without changing that ID.

## Identity and naming

- **Recipe ID**: `LM-` plus sixteen SHA-256 hex characters from canonical recipe JSON. Model/revision/dtype/attention, hook implementation fingerprint, location, token positions and exact operations determine identity. Changing a channel, native offset, gain, vector, amplitude or application method produces a different ID. Prompt sets and generation settings are recorded as observations rather than silently replacing a recipe.
- **Codename**: deterministic `Mischief-XXXXXX`, with no behavioral claim. This remains available even when no nickname is justified.
- **Method label / intended character**: how the change was constructed, such as `villain-L10@0.15`. This is a hypothesis or method label.
- **Observed name**: a human-readable nickname such as “Cookie Diplomat,” accompanied by scope, evidence and collateral damage. Status defaults to unreviewed. Provisional means a description worth tracking, not validation.

“Cookie Diplomat” currently refers to one friendly-sounding cookie reply under a combined villain-derived edit; it also misunderstands the event as sharing. “Pocket Editor” describes tentative shortening, including failed full replication. “Laundry Rambler” describes a damaged random-profile condition. None is labeled a validated personality.

## What each card records

Cards in [the library](../experiments/personas/README.md) contain exact intervention JSON, method derivation/source references, and a table of differences from matched default. Each observation retains its run and prompt cohort, raw-output indices, prompt hash, system instructions, generation settings, changed token count, length, EOS, repetition and available exact checks. Missing checks are not zero scores. Source configuration, code/input hashes, model revision and environment are retained in the registry's run records and local run archives.

Different cohorts, instruction contexts, token budgets, runtimes and source experiments are kept separate. A nickname never pools those differences into a personality score. Full raw examples stay in local runs/workbench; reviewed examples and conclusions live in experiment reports. Historical residual-direction cards identify exact vectors by digest and reference local tensor artifacts; those cards require those artifacts or reproduction of the original discovery procedure.

## Workflow after a change

1. Save the exact recipe and complete raw experiment evidence.
2. Compare with unedited output under the same prompt/settings; include controls and guardrails.
3. Build the local catalog with `uv run --locked python scripts/catalog.py`.
4. Assign a neutral codename automatically. Review before giving it a behavioral nickname; note tone/content/structure/damage separately.
5. Record scope and uncertainty. A single unusual answer is provisional. Fresh replication can support a limited description; an equal aggregate score can still hide an individual regression.
6. Publish reviewed cards with `uv run --locked python scripts/catalog.py --publish`, then review the Git diff before a milestone commit.

Local UI nickname edits are saved separately from immutable recipes and raw evidence. The public reviewed names are versioned in `experiments/personas/names.json`; local overrides are not automatically published. Performance observations must state actual context length, backend, hardware, memory and throughput, and be compared within that runtime. An eager recipe's short-prompt timing is not its 128K SDPA performance.

Interactive LIVE observations appear only in the local catalog. `--publish` excludes all LIVE conversations, histories and associated observations from public cards/registry. Deliberately reviewed smoke-test examples can be exported separately; routine chat contents are not automatically published. The Qwen published-checkpoint card remains reference-only even after a loading/arithmetic smoke test: that test does not reproduce its anti-loop claim.
