# Local experiment workbench

## Launch

From the project directory:

```powershell
uv sync --locked
uv run --locked python scripts/viewer.py
```

Open http://127.0.0.1:8765 in your browser. No frontend build, administrator privileges, cloud account or paid compute is needed. A different local port can be selected with `--port 8766`.

On Windows, you can also double-click **Start Workbench.cmd**, then open the printed URL. This uses the existing user-installed uv and the project environment.

If the header stays at **Connecting** and every menu is empty, refresh after restarting the viewer when its Python routes change. A missing JavaScript asset previously stopped initialization before any data loaded. The server now serves all required scripts; startup failures display an error, and pending menus show disabled loading labels. Startup chooses a run with recorded replies, skipping performance-only entries. `node tests/test_ui_startup.js --live` exercises script initialization against the running server without GPU inference; this is a DOM fixture check, not visual browser QA.

## What works in this version

- **Chat & Try** is the first workspace. Select LFM3-1.7B or Qwen3.5-0.8B in the header, choose an exact recipe, enter a prompt and run both replies. The unchanged control runs the same model twice. Each branch maintains its own conversation history; changing the model/recipe or choosing New conversation clears in-page history. Saved run files remain local. Reloading the page clears in-page chat history.
- LFM3 recipes use reversible hooks on the pinned original weights; historical vector recipes require their local discovery artifacts. Hook implementation fingerprints are checked before applying a saved recipe. Qwen uses the pinned official base versus the separately downloaded published edit; LFM3 channel indices are never applied to Qwen. The published anti-loop edit is distinct from the 4B Evil Qwen recipe.
- Live cards report actual input/output tokens, first-token latency, decode rate, peak allocated GPU memory and loading time. Model loading is separate from first-token latency. A worker retains one loaded checkpoint for up to 90 idle seconds; Qwen base/edit comparisons swap checkpoints and can require two loads. Stop terminates this server's worker and preserves saved/partial evidence. Windows virtualenv launcher children are included in worker tracking.
- The header's model selection filters saved runs, recipes and context-performance results. Frozen batch designs remain LFM3-specific and are disabled under Qwen; Qwen interactive comparisons are recorded as runs. The Performance workspace shows completed context measurements, rather than extrapolating from chat speed.
- **Saved Comparisons** provides default plus Edit A and optional Edit B/control, with word-change highlighting. **Persona Library** exposes stable recipe IDs, exact methodology, per-cohort differences and local nickname/evidence editing. Names are annotations, not automatic discoveries or scores. See [naming rules](PERSONAS.md).

- Run history reads local experiment folders, including partial runs. An explicit interruption record appears as interrupted; otherwise 'partial' does not distinguish still running from interrupted, so check logs/provenance.
- Comparison places an unsteered reference beside each matching probe reply. Filter by prompt, persona, condition, layer, dose, role and prompt cohort. Select `fresh` to separate replication prompts from the original anchor.
- Charts show reply length, EOS counts, mean generation time and intervention grids. Channel plots separate candidates and gains; target and same-layer controls are not pooled into one cell. Cohort filtering updates charts as well as comparisons. Other filters apply to the comparison cards. Values may still aggregate several prompts; inspect individual outputs.
- Optional grids show think-boundary and exact-JSON-match fractions where recorded. These are surface/format checks, not general reasoning or accuracy scores. Missing measurements show a dash, not an invented zero. Keyword grids remain browsing aids, not persona scores.
- Repeated token 4-gram fraction is `1 - unique consecutive 4-token sequences / total consecutive 4-token sequences` (zero for fewer than four tokens). It is a repetition indicator, not a validated loop detector; legitimate repeated structure can increase it.
- Review labels and notes save separately in each run's annotations.json. They do not overwrite raw outputs. In-page unsaved drafts survive automatic rerendering, but a browser reload discards unsaved drafts.
- Export downloads the selected run's JSON, including metadata, raw output and saved annotations. Review contents before sharing; future prompts may contain private data.
- Configure a run defaults to effect-first mapping and offers the pilot, fresh-prompt replication and expanded-sample designs. Their versioned configurations are copied intact into the job record. Persona mode retains adjustable layers/doses/token/control settings. Launch creates a saved config and stdout/stderr log under runs/jobs. Jobs use the project interpreter and allowlisted scripts/design names only.
- Job history and saved logs survive a server restart. Stop controls cover jobs started by the current server session. Stopping preserves partial evidence. CLI experiments appear in run history but cannot be stopped from the UI.

## Character maps and direct comparisons

Character experiments add browsable signed channel rankings for both styles across all 24 layers. A bar is paired mean activation difference divided by neutral response RMS; its tooltip includes pair consistency and native RMS. The first sixteen channels are selected only in the four tested layers. These are fixed-text associations, not a complete neuron atlas or validated personality functions.

Saved comparisons offer optional word-change highlighting. Additions/removals use plain DOM text; model content cannot become executable markup. Word edits are separate from tokenizer edits and are not quality scores. Edit A and Edit B/control replace the former pin tray; changing a prompt updates both selectors. Recipe names link to their methodology and scoped evidence.

Frozen launch modes include sparse character discovery, combined profiles at amplitude multipliers 1/2/4, and fresh offset replication. Sparse profile plots separate profile/strength, prompted controls and random profiles. Combined profiles expose actual layer changes individually; multipliers do not stand for calibrated percentages. The measured-strength grid shows unavailable joint scalar values as missing.

Export includes recipes/rankings when present. The campaign reporter verifies source archives, exact schedules and recomputed measurements, then creates numeric exports and PNG/SVG figures. For replication it evaluates the predeclared criteria with fresh cohorts separated from anchors. Its reports track individual lost checks/new cap failures rather than hiding regressions behind equal aggregate scores.

## Limits

The server binds only to 127.0.0.1 and is a local development tool, not a hosted multi-user service. Do not expose it through a tunnel or bind it to a public interface. Mutation endpoints require a per-session token and check Host/Origin. Model output is rendered as plain text, never executable HTML.

The job controller checks for other local experiment processes to avoid overlapping GPU jobs. This is a process-level guard, not a cross-machine scheduler. Restarting the server loses its managed-process handles; previous logs and finished statuses remain readable, but jobs are not resumed or reattached automatically. A previously running job is shown as 'unmanaged' rather than presumed failed or complete.

Live chat and a manual prompt editor are implemented and inference-smoke-tested; visual interaction testing remains incomplete. There is no arbitrary-neuron editor, automatic capability grader or arbitrary command execution. Charts do not establish improved ability. Actual 128K inference failed on this machine; the declared model window does not promise that context is usable here. Qwen uses PyTorch reference fallbacks for optional convolution/linear-attention kernels on this Windows setup. See [fresh-start QA](UI-QA.md) for the browser-tool blocker and precise nonvisual checks.

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

An interrupted calibrated evaluation can be continued with `uv run --locked python scripts/resume_calibrated.py --run PARENT-RUN-ID`. This creates a new run, preserves and hashes parent evidence, and requires exact baseline-token and prefill-tensor revalidation before continuing the saved schedule. Additional replay checks are recorded separately. It cannot supply a missing final parameter-version check for the interrupted segment. The GPU guard includes this runner. JSON replacement retries transient Windows PermissionError conflicts; persistent errors still stop a run while preserving the prior complete record.
