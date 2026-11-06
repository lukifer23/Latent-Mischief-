# Latent Mischief

*Small models. Strange behaviors. Reproducible experiments.*

Project repository: https://github.com/lukifer23/Latent-Mischief-

## Current evidence

| Available now | Status |
|---|---|
| Native Windows GPU inference without admin | Verified locally |
| Pinned model, dependencies, raw evidence recording | Implemented |
| Neutral and removed-hook checks | Passed on the documented prompts |
| Two persona-direction sweeps | 148 generations; no validated internal persona recipe; strong pushes often damaged output |
| Effect-first channel pilot | 41 generations; 18/36 intervention replies changed tokens, 18 stayed identical; traits remain unvalidated |
| Local comparison / run-control workbench | Implemented; API tests pass, visual browser QA pending |
| Validated hamster or villain neuron recipe | Not yet discovered |
| Capability improvement or state-of-the-art result | Not established |

See [effect-first pilot](experiments/MAP-001-effect-first-pilot.md), [first-sweep findings](experiments/LFM-002-first-sweep.md) and [stronger-dose findings](experiments/LFM-002-stronger-dose.md). Planned work is not a result, and personality changes would not by themselves demonstrate improved capability.

A local lab for exploring LFM3 behavior through reversible activation interventions. Current priority: sample controlled internal changes, document the effects that emerge, and build a replicated effect catalog before designing targeted personas. Hamster and theatrical-villain experiments remain historical evidence and possible later demonstrations. Qwen0.8B capability experiments and compiled computation remain on the idea board.

## Start here

- [Experiment board](docs/ROADMAP.md): checklist, status, notes and future ideas.
- [Research journal](docs/JOURNAL.md): dated decisions, observations and next steps.
- [Setup](docs/SETUP.md): native Windows, no administrator privileges.
- [Local workbench](docs/WORKBENCH.md): comparisons, charts, run controls, notes and logs.
- [Method](docs/METHOD.md): how we distinguish a funny output from a repeatable effect.
- [Open exploration](docs/EXPLORATION.md): change first, characterize afterward.
- [Sources](docs/SOURCES.md): upstream references and provenance.
- [Architecture](docs/ARCHITECTURE.md): components, evidence files and data flow.
- [Contributing](CONTRIBUTING.md): experiment and validation conventions.
- [Experiment template](experiments/TEMPLATE.md): one report per reviewed experiment.

## Commands

Run from this directory in PowerShell. Activation is unnecessary, so no execution-policy changes are needed.

```powershell
uv sync --locked
uv run --locked python scripts/lab.py doctor
uv run --locked python scripts/lab.py prepare
uv run --locked python scripts/lab.py smoke
uv run --locked python scripts/personas.py
uv run --locked python scripts/explore.py
uv run --locked python scripts/viewer.py
```

`prepare` downloads pinned source/config files. `smoke` downloads approximately 3.5 GB of model weights on first use, loads the GPU model, generates five baselines and checks a gain=1 hook for token equality. No upstream weights are modified. Raw run files go in `runs/`; model/cache files in `.cache/`.

`personas.py` generates paired neutral/hamster/villain responses, derives response-token contrast directions, and tests three layers and three doses on prompt-disjoint exploratory probes. It includes ordinary persona-prompt baselines, zero-dose / hook-removal checks, and two random directions per layer at dose 0.15. Random controls are matched only at 0.15; the other doses are exploratory. This run is candidate discovery, not held-out validation. See [the specification](experiments/LFM-002-persona-discovery.md).

`viewer.py` serves the local workbench at http://127.0.0.1:8765. It reads run history, compares responses, charts basic measurements, saves review notes, and launches bounded experiment configurations. Nothing is hosted publicly by this command.

`explore.py` runs the effect-first pilot: uniformly sample one MLP channel in each of six layers, ablate or double its natural activation, and compare three neutral prompts. Candidates are chosen before viewing outputs; all outcomes are retained. Launch this runner from the CLI for now; the viewer displays its history, gain/layer measurements and comparisons.

## Organization

```text
configs/       Pinned model and runtime defaults
docs/          Board, journal, setup, method, references
experiments/   Reviewed reports and experiment specifications
prompts/       Versioned prompt sets (baseline is not held-out evaluation)
scripts/       Reproducible commands
ui/            Local workbench HTML, CSS and JavaScript
tests/         API boundary and recording checks
runs/          Local raw outputs, metadata, checks (ignored by Git)
.cache/        Local model downloads (ignored by Git)
uv.lock        Exact dependency resolution
```

No behavior recipe has been validated yet. Read the journal for actual progress; roadmap items are not results.

## License and workflow

Original project code and documentation: [MIT](LICENSE). Downloaded models and external projects retain their own terms; see [license scope](docs/LICENSING.md).

Development happens directly on `main`, with remote updates at documented milestones. Local caches, model weights and raw runs stay out of Git.
