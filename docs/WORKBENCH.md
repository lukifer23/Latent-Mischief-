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
- Comparison places an unsteered reference beside each matching probe reply. Filter by prompt, persona, condition, layer, dose, role and prompt cohort. Select `fresh` to separate replication prompts from the original anchor.
- Charts show reply length, EOS counts, mean generation time and intervention grids. Channel plots separate candidates and gains; target and same-layer controls are not pooled into one cell. Cohort filtering updates charts as well as comparisons. Other filters apply to the comparison cards. Values may still aggregate several prompts; inspect individual outputs.
- Optional grids show think-boundary and exact-JSON-match fractions where recorded. These are surface/format checks, not general reasoning or accuracy scores. Missing measurements show a dash, not an invented zero. Keyword grids remain browsing aids, not persona scores.
- Repeated token 4-gram fraction is `1 - unique consecutive 4-token sequences / total consecutive 4-token sequences` (zero for fewer than four tokens). It is a repetition indicator, not a validated loop detector; legitimate repeated structure can increase it.
- Review labels and notes save separately in each run's annotations.json. They do not overwrite raw outputs. In-page unsaved drafts survive automatic rerendering, but a browser reload discards unsaved drafts.
- Export downloads the selected run's JSON, including metadata, raw output and saved annotations. Review contents before sharing; future prompts may contain private data.
- Configure a run defaults to effect-first mapping and offers the pilot, fresh-prompt replication and expanded-sample designs. Their versioned configurations are copied intact into the job record. Persona mode retains adjustable layers/doses/token/control settings. Launch creates a saved config and stdout/stderr log under runs/jobs. Jobs use the project interpreter and allowlisted scripts/design names only.
- Job history and saved logs survive a server restart. Stop controls cover jobs started by the current server session. Stopping preserves partial evidence. CLI experiments appear in run history but cannot be stopped from the UI.

## Limits

The server binds only to 127.0.0.1 and is a local development tool, not a hosted multi-user service. Do not expose it through a tunnel or bind it to a public interface. Mutation endpoints require a per-session token and check Host/Origin. Model output is rendered as plain text, never executable HTML.

The job controller checks for other local experiment processes to avoid overlapping GPU jobs. This is a process-level guard, not a cross-machine scheduler. Restarting the server loses its managed-process handles; previous logs and finished statuses remain readable, but jobs are not resumed or reattached automatically. A previously running job is shown as 'unmanaged' rather than presumed failed or complete.

No live chat, individual-neuron map, automatic capability grader, manual prompt editor or arbitrary command execution is implemented. Persona definitions and prompt sets remain versioned files. Charts do not establish improved ability. Visual browser verification is pending: the Computer Use tool stopped because it could not confidently determine the browser URL. API and JavaScript syntax checks are separate from visual QA.

Open exploration can now be launched through the workbench's frozen-design selector or the CLI. Gain multiplies natural channel activity; additive direction dose is a different quantity and is not pooled with gain. The copied design can be reproduced independently from the saved job config. Editing a predeclared design requires a new version before the next run, not an invisible UI override.

## Reproduce from the CLI

The two-stage calibrated workflow is also available in Configure a run. **Calibrate and screen 96 channels** uses the frozen versioned design. **Evaluate frozen calibrated offsets** requires a completed qualifying calibration from local history; the runner verifies its hashes and model/runtime lock. Reusing these evaluation prompts is a repeat, not fresh validation. Calibration-only history entries contain measurements rather than generated replies. Offset charts keep channel/sign separate, and cards show native offset, calibration-RMS units and measured prefill change. The strength grid is a relative fraction at post_ffn_norm, not a percentage of behavior change.

```powershell
uv run --locked python scripts/personas.py
uv run --locked python scripts/personas.py --config runs/jobs/JOB-ID/config.json
uv run --locked python scripts/map_experiment.py --config configs/completion-replication.json
uv run --locked python scripts/map_experiment.py --config configs/exploration-expanded.json
uv run --locked python scripts/verify_run.py RUN-ID
uv run --locked python scripts/summarize.py RUN-ID
uv run --locked python scripts/calibrate.py
uv run --locked python scripts/calibrated_map.py --calibration CALIBRATION-RUN-ID
uv run --locked python -m unittest discover -s tests -v
```

Runners load their config once and record its content and hash. Mapping runs also save the shuffled trial schedule. `verify_run.py` checks archived hashes, hook checks and exact scheduled trials; `summarize.py` preserves candidate/gain/role/cohort groups instead of pooling controls. Keep scripts unchanged while a run executes; commit code versions between runs. Current generation is greedy, with a fixed seed; sampled multi-seed validation remains future work.

To export numeric evidence and PNG/SVG figures after both stages complete:

```powershell
uv sync --locked --extra analysis
uv run --locked --extra analysis python scripts/evidence_report.py --calibration CALIBRATION-RUN-ID --evaluation EVALUATION-RUN-ID
```

This verifies source archives and candidate selection, then exports numeric measurements rather than raw reply text. Published figures can be redrawn from their accompanying `metrics.json` using `--evidence PATH/metrics.json --destination LOCAL-FOLDER`. Browser visual QA remains pending; figure rendering is checked separately.
