# Latent Mischief

*Small models. Strange behaviors. Reproducible experiments.*

Project repository: https://github.com/lukifer23/Latent-Mischief-

## Current evidence

| Available now | Status |
|---|---|
| Native Windows GPU inference without admin | Verified locally |
| Apple Silicon setup and short inference | M3 Pro / MPS / BF16 verified for LFM3 and both Qwen checkpoints; long MPS campaigns unverified |
| Pinned model, dependencies, raw evidence recording | Implemented |
| Neutral and removed-hook checks | Passed on the documented prompts |
| Two persona-direction sweeps | 148 generations; no validated internal persona recipe; strong pushes often damaged output |
| Effect-first channel pilot | 41 generations; 18/36 intervention replies changed tokens, 18 stayed identical; traits remain unvalidated |
| Completion-candidate replication | 51 generations; anchor shortening repeats, no fresh-prompt completion improvement |
| Expanded effect-first mapping | 77 generations over 12 new channels; 40/72 intervention replies changed, 32 stayed identical; traits unvalidated |
| Calibrated single-channel offsets | 184,320-channel statistics; 96 sampled channels, 384 conditions, 1,536 screen forwards |
| Diverse calibrated evaluation | 210 mapping records + 18 recovery checks; 117/192 intervention replies changed; no strict-check score improvement |
| Sparse character profiles | 170 generations; no coherent internal persona established |
| Fresh calibrated-offset replication | 158 generations; 6.81% shortening for one candidate, below 10% criterion; completion candidate failed |
| Combined character profiles | 86 generations; 56/63 internal replies changed; hamster/random regressions retained |
| Actual LFM3 long-context performance | Paired timings through 32,768 input tokens; 131,072-token attempt ran out of memory |
| Qwen3.5-0.8B base and published edit | All 488 tensors audited; sampled pilot user-stopped at 6/8 saved, two full pairs. Both checkpoints capped without final answers on those pairs; no anti-loop benefit established at 4K |
| Model-aware Chat & Try / saved comparisons | API and DOM-fixture checks pass; fresh visual browser QA blocked |
| Exact recipe / persona library | 135 model-specific recipes; provisional names and evidence kept separate |
| Validated hamster or villain neuron recipe | Not yet discovered |
| Capability improvement or state-of-the-art result | Not established |

See [expanded mapping](experiments/MAP-004-expanded-sample.md), [replication findings](experiments/MAP-003-completion-replication.md), [effect-first pilot](experiments/MAP-001-effect-first-pilot.md), [first-sweep findings](experiments/LFM-002-first-sweep.md) and [stronger-dose findings](experiments/LFM-002-stronger-dose.md). Planned work is not a result, and personality changes would not by themselves demonstrate improved capability.

A local lab for exploring LFM3 behavior through reversible activation interventions, with a separate Qwen3.5-0.8B track for comparing the official base against a pinned published weight-space edit. Current priority: characterize changes with matched controls and honest regression checks before claiming targeted personas. Compiled computation remains on the idea board.

Recent research: [fresh replication](experiments/MAP-006-offset-replication.md), [combined profiles](experiments/CHAR-002-joint-characters.md), [context performance and failures](experiments/PERF-001-long-context.md), and [paired LFM3/Qwen smoke tests](experiments/LIVE-001-model-comparisons.md). The three newer LFM3 campaigns total 414 generations. There is verified behavior change and useful negative evidence; no validated evil/hamster persona or capability improvement. See the [recipe library](experiments/personas/README.md) and [QA record](docs/UI-QA.md) for scope and remaining gaps.

Latest milestone: [full project review](docs/REVIEW-2026-11-09.md) and [Mac setup evidence](experiments/BACKEND-001-mac-setup.md). Short MPS inference, cache/sampling checks and paired HTTP conversations work. The [QWN-002 Windows pilot](experiments/QWN-002-anti-loop-pilot.md) remains interrupted and separate; a larger Mac follow-up is planned, not run.

Next protocol: [QWN-003](experiments/QWN-003-mac-followup.md), a separately frozen two-problem MPS follow-up with resource admission, allocator and cooperative-stop limits. Its implementation is tested; research generation has not started.

## Start here

- **Switching devices?** Read [the handoff](docs/HANDOFF.md): interrupted Windows evidence, private raw-data transfer, Mac setup and verified scope.
- [Next steps](docs/NEXT-STEPS.md): ordered execution plan, first milestone, evidence requirements and stopping rules.
- [Experiment board](docs/ROADMAP.md): checklist, status, notes and future ideas.
- [Research journal](docs/JOURNAL.md): dated decisions, observations and next steps.
- [Setup](docs/SETUP.md): Apple Silicon MPS and native Windows CUDA, no administrator privileges.
- [Local workbench](docs/WORKBENCH.md): comparisons, charts, run controls, notes and logs.
- [Persona naming](docs/PERSONAS.md): exact recipe IDs, provisional names, differences and methodology.
- [Method](docs/METHOD.md): how we distinguish a funny output from a repeatable effect.
- [Open exploration](docs/EXPLORATION.md): change first, characterize afterward.
- [Sources](docs/SOURCES.md): upstream references and provenance.
- [Architecture](docs/ARCHITECTURE.md): components, evidence files and data flow.
- [Contributing](CONTRIBUTING.md): experiment and validation conventions.
- [Experiment template](experiments/TEMPLATE.md): one report per reviewed experiment.

## Commands

Run from this directory in macOS Terminal or PowerShell. Activation is unnecessary.

```powershell
uv sync --locked --all-extras
uv run --locked python scripts/lab.py doctor
uv run --locked python scripts/lab.py prepare
uv run --locked python scripts/lab.py smoke
uv run --locked python scripts/personas.py
uv run --locked python scripts/explore.py
uv run --locked python scripts/map_experiment.py --config configs/completion-replication.json
uv run --locked python scripts/map_experiment.py --config configs/exploration-expanded.json
uv run --locked python scripts/calibrate.py
uv run --locked python scripts/calibrated_map.py --calibration CALIBRATION-RUN-ID
uv run --locked python scripts/viewer.py
```

`prepare` downloads pinned source/config files. `smoke` downloads approximately 3.5 GB of model weights on first use, loads on CUDA or MPS, generates five baselines and checks a gain=1 hook for token equality. CPU must be selected explicitly. No upstream weights are modified. Raw run files go in `runs/`; model/cache files in `.cache/`. Runtime choices and Qwen/cache checks are in [setup](docs/SETUP.md). Frozen Windows calibration, Qwen-pilot and long-context designs have runtime gates; this command list is a workflow reference, not a request to run every campaign on a new device.

`personas.py` generates paired neutral/hamster/villain responses, derives response-token contrast directions, and tests three layers and three doses on prompt-disjoint exploratory probes. It includes ordinary persona-prompt baselines, zero-dose / hook-removal checks, and two random directions per layer at dose 0.15. Random controls are matched only at 0.15; the other doses are exploratory. This run is candidate discovery, not held-out validation. See [the specification](experiments/LFM-002-persona-discovery.md).

`viewer.py` serves the local workbench at http://127.0.0.1:8765. It reads run history, compares responses, charts basic measurements, saves review notes, and launches bounded experiment configurations. Nothing is hosted publicly by this command.

`explore.py` runs the original effect-first pilot: uniformly sample one MLP channel in each of six layers, ablate or double its natural activation, and compare three neutral prompts. Candidates are chosen before viewing outputs; all outcomes are retained. The workbench can launch this pilot or the newer frozen mapping designs.

`map_experiment.py` runs versioned replication or expanded-mapping designs. The completion replication separates fresh prompts from the discovery anchor and includes three same-layer random channel controls. Expanded mapping excludes the original six sampled channels. These designs are available in the workbench's launch selector. Run GPU experiments sequentially; do not launch both CLI commands together.

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
