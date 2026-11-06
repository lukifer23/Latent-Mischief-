# Latent Mischief

*Small models. Strange behaviors. Reproducible experiments.*

Project repository: https://github.com/lukifer23/Latent-Mischief-

A local lab for exploring LFM3 behavior through reversible activation interventions. First targets: hamster characterization and theatrical villain personality. Qwen0.8B capability experiments and compiled computation remain on the idea board.

## Start here

- [Experiment board](docs/ROADMAP.md): checklist, status, notes and future ideas.
- [Research journal](docs/JOURNAL.md): dated decisions, observations and next steps.
- [Setup](docs/SETUP.md): native Windows, no administrator privileges.
- [Method](docs/METHOD.md): how we distinguish a funny output from a repeatable effect.
- [Sources](docs/SOURCES.md): upstream references and provenance.
- [Experiment template](experiments/TEMPLATE.md): one report per reviewed experiment.

## Commands

Run from this directory in PowerShell. Activation is unnecessary, so no execution-policy changes are needed.

```powershell
uv sync --locked
uv run --locked python scripts/lab.py doctor
uv run --locked python scripts/lab.py prepare
uv run --locked python scripts/lab.py smoke
```

`prepare` downloads pinned source/config files. `smoke` downloads approximately 3.5 GB of model weights on first use, loads the GPU model, generates five baselines and checks a gain=1 hook for token equality. No upstream weights are modified. Raw run files go in `runs/`; model/cache files in `.cache/`.

## Organization

```text
configs/       Pinned model and runtime defaults
docs/          Board, journal, setup, method, references
experiments/   Reviewed reports and experiment specifications
prompts/       Versioned prompt sets (baseline is not held-out evaluation)
scripts/       Reproducible commands
runs/          Local raw outputs, metadata, checks (ignored by Git)
.cache/        Local model downloads (ignored by Git)
uv.lock        Exact dependency resolution
```

No behavior recipe has been validated yet. Read the journal for actual progress; roadmap items are not results.
