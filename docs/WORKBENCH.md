# Local experiment workbench

## Launch

From the project directory:

```powershell
uv sync --locked
uv run --locked python scripts/viewer.py
```

Open http://127.0.0.1:8765 in your browser. No frontend build, administrator privileges, cloud account or paid compute is needed. A different local port can be selected with `--port 8766`.

On Windows, you can also double-click **Start Workbench.cmd**, then open the printed URL. This uses the existing user-installed uv and the project environment.

## What works in this version

- Run history reads local experiment folders, including partial runs. 'Partial' does not distinguish still running from interrupted; check job logs.
- Comparison places an unsteered reference beside each matching probe reply. Filter by prompt, persona, condition, layer and dose.
- Charts show reply length, EOS counts, mean generation time and dose/layer grids. Keyword grids count simple terms, not persona quality or accuracy. Condition aggregates mix prompts; inspect individual outputs before interpreting changes.
- Repeated token 4-gram fraction is `1 - unique consecutive 4-token sequences / total consecutive 4-token sequences` (zero for fewer than four tokens). It is a repetition indicator, not a validated loop detector; legitimate repeated structure can increase it.
- Review labels and notes save separately in each run's annotations.json. They do not overwrite raw outputs. In-page unsaved drafts survive automatic rerendering, but a browser reload discards unsaved drafts.
- Export downloads the selected run's JSON, including metadata, raw output and saved annotations. Review contents before sharing; future prompts may contain private data.
- Configure a run adjusts layers, doses, token cap, random control count and random-control dose. Launch creates an immutable job config and a stdout/stderr log under runs/jobs. Jobs use the project interpreter and allowlisted commands only.
- Job history and saved logs survive a server restart. Stop controls cover jobs started by the current server session. Stopping preserves partial evidence. CLI experiments appear in run history but cannot be stopped from the UI.

## Limits

The server binds only to 127.0.0.1 and is a local development tool, not a hosted multi-user service. Do not expose it through a tunnel or bind it to a public interface. Mutation endpoints require a per-session token and check Host/Origin. Model output is rendered as plain text, never executable HTML.

The job controller checks for other local experiment processes to avoid overlapping GPU jobs. This is a process-level guard, not a cross-machine scheduler. Restarting the server loses its managed-process handles; previous logs and finished statuses remain readable, but jobs are not resumed or reattached automatically. A previously running job is shown as 'unmanaged' rather than presumed failed or complete.

No live chat, individual-neuron map, automatic capability grader, manual prompt editor or arbitrary command execution is implemented. Persona definitions and prompt sets remain versioned files. Charts do not establish improved ability. Visual browser verification is pending: the Computer Use tool stopped because it could not confidently determine the browser URL. API and JavaScript syntax checks are separate from visual QA.

## Reproduce from the CLI

```powershell
uv run --locked python scripts/personas.py
uv run --locked python scripts/personas.py --config runs/jobs/JOB-ID/config.json
uv run --locked python -m unittest discover -s tests -v
```

The persona runner loads the config once and records its content and hash. Keep scripts unchanged while a run executes; commit code versions between runs. Current generation is greedy, with a fixed seed; sampled multi-seed validation remains future work.
