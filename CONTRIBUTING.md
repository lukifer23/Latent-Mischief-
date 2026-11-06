# Working on Latent Mischief

Start with [README](README.md), the [experiment board](docs/ROADMAP.md), and [working agreements](AGENTS.md).

Work directly on `main`. Commit and push real milestones with a concise description of what changed and what was checked. Do not create feature branches or force-push. Original code and docs use MIT; see [license scope](docs/LICENSING.md).

## Propose or run an experiment

1. Reuse a roadmap ID or add a new one with a precise question.
2. Copy experiments/TEMPLATE.md; define the intervention, controls, scoring and stopping criteria before looking at outcomes.
3. Keep model revision and dependencies pinned. No admin installs or paid compute are assumed.
4. Run through the CLI or local workbench and retain raw outputs, including failures. Keep source files unchanged while their experiment runs.
5. Record a dated journal entry. Write a reviewed report with observed results, confounds and the next test. Discovery probes that have been inspected are no longer a final held-out set.
6. Commit code, prompt/config changes and reviewed reports. Do not commit caches, models, environments, credentials or raw run directories.

## Validate changes

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
node --check ui/app.js
```

Node is needed only for the optional JavaScript syntax check, not to run the workbench. Changes to model hooks need explicit neutrality/cleanup checks and evidence that the intended intervention is applied at the documented location. Tests of the HTTP API do not replace browser visual/interaction checks.

Use precise claims: 'keyword appeared in this reply', 'a prompt-conditioned character transferred to these probes', and 'accuracy improved on this held-out task' require different evidence. Report empty or negative results with the same care as interesting examples.
