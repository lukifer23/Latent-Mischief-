# Research journal

## 2026-11-06 — Lab kickoff

Decision: LFM3 first. Initial targets are hamster characterization and theatrical villain behavior, using reversible activation interventions before permanent weight edits.

Constraints: no administrator access, native Windows first, no paid compute. GPU: RTX 2000 Ada Generation, 16,380 MiB total; system RAM about 64 GiB. Existing user tools: uv, Git, Python 3.13 and uv-managed Python 3.11.

Verified Hub checkpoint: LiquidAI/LFM3-1.7B, revision 0dc4268b561499c648c7ba7c9466bf3d8d01517a. Config: 24 layers, hidden size 2048, MLP width 7680, sandwich normalization, 13 convolution + 5 sliding-attention + 6 full-attention layers. Repository ships custom Python code; inspect and pin it before execution.

Novelty caveat: public gguf-neuron-explorer documentation already mentions LFM3. We are not claiming the first LFM steering experiment. No successful behavior intervention has been observed in this lab yet.

### Environment and baseline milestone

Installed isolated Python 3.11.15 environment with torch 2.11.0+cu128, Transformers 5.22.2, huggingface-hub 1.33.0, accelerate 1.15.0 and safetensors 0.8.0. CUDA available; native Windows loading and inference succeeded without admin access, WSL or custom CUDA compilation.

Downloaded pinned 3,457,635,448-byte safetensors checkpoint. Inspected configuration, MLP/decoder path and convolution fallback; recorded source hashes. This was an architecture/compatibility inspection, not a full security audit.

Initial run: `runs/20261106T181203314902Z-LFM-001-baseline/`. Five baseline prompts at greedy decoding and 96 new tokens, plus one neutral gain hook. The gain=1 hook at layer 12, channel 0 (input of feed_forward.w2) produced token-identical identity output. Identity, capital and JSON prompts reached EOS. Arithmetic and story hit the 96-token cap; no claim of looping. Arithmetic ignored integer-only formatting and contained a wrong intermediate result (587; expected final product is 391). Small baseline set only, not a capability benchmark.

Implementation note: metadata capture was improved while the initial process was running, so the initial on-disk script hash is not an exact archive of that process's loaded source. A finalized rerun is required for the clean reference artifact. Do not treat the initial run as the reproducibility reference.

Next: paired persona discovery responses and reversible direction sweeps under LFM-002 / LFM-003. Public repository name pending user choice; 'Latent Mischief' proposed. Nothing has been published.

Finalized rerun: `runs/20261106T181325639283Z-LFM-001-baseline/`. Neutral hook passed again. Full token inputs, checkpoint generation defaults and the unchanged executed script hash are captured. This is the reference baseline artifact; both baseline runs produced the same token sequences on all five prompts.

### Project name and public repository

User selected **Latent Mischief** and authorized the initial commit and GitHub publication at https://github.com/lukifer23/Latent-Mischief-. Local directory stays at its existing path so the working environment and raw evidence remain intact. Public source excludes caches, weights, environments and raw run output. The project package was renamed; the earlier baseline retains its original lock hash as historical provenance.

Initial commit `403fa23` pushed to `origin/main` and verified against the remote. No existing Git author identity was configured, so the commit used the repository owner's GitHub username and username-based GitHub noreply address for this command only; global Git configuration was not changed.

### First persona discovery run

Started `runs/20261106T182140177847Z-LFM-002-persona-discovery/`: paired generated-response features for neutral, hamster and theatrical villain characterizations. Four discovery prompts; three prompt-disjoint exploratory probes; three layers and doses; two random controls per selected layer at dose 0.15. Full details and limitations are in experiments/LFM-002-persona-discovery.md. Source scripts are kept unchanged during execution so recorded hashes identify the executed implementation.

Completed the 95-generation first sweep. Zero-dose and removed-hook checks passed. Explicit hamster prompting worked on the three probes; internal pushes did not establish a convincing hamster/villain recipe at doses 0.05/0.15/0.35. Every unsteered, random and contrast-direction probe hit the 96-token cap. Recorded this limitation and the wrong baseline Moon explanation in experiments/LFM-002-first-sweep.md. Negative results are retained.

### Repository metadata and workbench

Updated GitHub description and 11 topics through the GitHub REST API using the existing Git credential helper; no credential was printed or written to project files. Verified the returned metadata. No hosted demo URL was invented.

Implemented a local workbench: run history, filtered side-by-side comparisons, basic charts, dose/layer grids, saved human annotations, JSON export, bounded experiment launch controls and job logs. Loopback-only server; no admin or frontend build. API boundary/recording tests passed. Browser access host was unavailable; native-browser fallback then stopped because Computer Use could not determine the URL confidently. No further browser automation attempted; visual QA remains pending.

Started a follow-up through the job API: job `20261106T182951911844Z-personas`, layers 13/19, doses 0.7/1.0, 192-token cap and one random direction per layer at dose 0.7. This also exercises saved job configuration and process launch. The low control count is exploratory. Run outcome pending.

Follow-up completed successfully at the process level: 53 generations, both hook checks passed, but high-dose characterization transfer failed and frequently damaged generation. Layer-19 hamster pushes averaged 42.50% repeated 4-grams at dose 0.7 versus 0.88% for the one matched random vector; at dose 1.0 repetition was 91.71% with no matched random control. These are descriptive results over three prompts, not independent-vector significance tests. See experiments/LFM-002-stronger-dose.md for exact comparisons and confounds.

Both persona runs now have exact source/input snapshots whose hashes match their original metadata (the earlier source versions were reconstructed and hash-verified before archiving). Future runners archive source, prompt/config and lock files automatically. Added atomic JSON replacement to preserve the last complete record if a write fails; tests exercise that failure path. This persistence change was made only after the active model run completed.

### Publication and licensing policy

User requested regular milestone commits/pushes, main only, and authorized choosing MIT or Apache 2.0. Selected MIT for original project code and documentation; added LICENSE, package metadata and docs/LICENSING.md distinguishing the model's upstream terms. Recorded the main-only workflow in AGENTS.md and CONTRIBUTING.md. First concrete results are the documented hook checks, prompted-persona behavior, failed internal transfer, and high-dose degradation; no successful neuron recipe or capability gain is claimed.

Final milestone verification: 10 API/persistence tests passed, JavaScript syntax check passed, and the updated local server retained completed job history/logs across restart. Launched a fresh GPU baseline through the job API (`20261106T184000813192Z-smoke`); it exited 0 and the neutral hook passed. Raw baseline: `runs/20261106T184010480410Z-LFM-001-baseline/`, including automatic source/config/lock snapshots. Workbench remains available on http://127.0.0.1:8765; browser visual QA is still pending, not silently marked passed.

Milestone `feed6f7` pushed and verified on origin/main: controlled persona sweeps, honest findings, local workbench, MIT license and tests. GitHub reported three dependency advisories: two setuptools issues fixed in 84.1.0 or earlier, and torch.jit.script memory corruption fixed in PyTorch 2.13.0. Retrieved exact advisories from the repository's Dependabot API. Updating to patched prebuilt packages; this changes the runtime, so previous numerical results retain their original versions and a new GPU baseline is required.

### Strategy update: map effects before pursuing personas

User proposed sampling internal changes and recording the behavior/personality shifts that emerge, then using that catalog for targeted editing. Adopted as the next priority (MAP-001). Tone, content, structure, damage and no-effect are distinct outcomes; labels follow observations and remain provisional. This avoids interpreting a changed or broken reply as evidence for the persona we hoped to create. Previous persona-direction failures remain in the record.

Patched environment installed torch 2.14.1+cu130 and setuptools 84.2.1, both above the reported fixed-version floors. Native Windows GPU baseline succeeded (`runs/20261106T184453563784Z-LFM-001-baseline/`); neutral gain hook passed. No system driver/toolkit change was made. All 10 API/persistence tests passed again.

Started MAP-001 at `runs/20261106T184602567236Z-MAP-001-channel-exploration/`: six uniformly sampled channels across convolution/sliding/full-attention layers, gains 0/2, three neutral probes, planned 41 generations. All conditions share the same system instruction. Viewer supports the resulting comparisons and gain/layer measurements; launching this new runner is CLI-only in this version.

MAP-001 completed: 41 total generations, both hook checks passed. 18/36 intervention replies changed token sequence and 18 were identical. L22N2006 showed no output changes in the six tested conditions. L10N0204 at gain 2 made the evening response terminate without its baseline's overt sentence-count check/restart, while the disagreement response stayed identical. This is a tentative single-prompt completion observation, not a neuron function or general personality label. Findings and confounds are in experiments/MAP-001-effect-first-pilot.md; catalog includes no-effect and damage outcomes.

### Next milestone: replication and broader mapping

Before generating new results, specified MAP-003 (51 generations): L10N0204 at gains 0/1.5/2, three same-layer random channels at gain 2, six fresh prompts plus the original anchor. Predeclared a limited completion criterion and explicit floor-effect handling. MAP-004 separately samples 12 new channels on the existing discovery prompts (77 generations). Seeds fix candidate/trial selection; greedy generations are not treated as independent sampling-seed replicates.

Added a generic mapping runner, deterministic candidate validation, shuffled trial schedules, cohort labels and transparent phrase/format measurements. Four new tests passed alongside the existing ten. Source/config files will remain unchanged during execution. The original pilot runner remains available to preserve its historical workflow.

Pushed pre-generation designs and code as d813ab2. MAP-003 finished at `runs/20261106T185739757150Z-MAP-003-channel-exploration/` with 51 outputs and both hook checks passing. All baseline/target/control conditions completed 5/6 fresh prompts; locked-box remained capped. Target gain 2 reproduced the original anchor shortening (171 to 72 tokens) but increased fresh think-boundary markers (2/6 to 4/6). The general completion criterion was not met, with limited repair opportunity because the fresh baseline had only one EOS failure. Exact JSON passed all seven conditions. Updated the catalog rather than naming a general reasoning/completion neuron.

Workbench updates in progress: allowlisted frozen mapping designs with exact saved configs, role/fresh-cohort comparison filters, per-candidate rather than pooled same-layer grids and chart groups. Added an API test confirming mapping designs cannot be overridden through arbitrary token-cap fields. 15 tests pass; visual browser QA remains pending.

MAP-004 launched through the new mapping job API as `20261106T190341741000Z-mapping`; raw run `runs/20261106T190341875119Z-MAP-004-channel-exploration/`. This exercises a real saved-config launch in addition to API boundary tests. Added run verification and grouped summaries; MAP-003's eight archived input/source files and 42 scheduled intervention trials verified. Verification tests reject changed source bytes and duplicate trial records. The workbench treats missing keyword measurements as unavailable, not zero.

Pushed MAP-003 findings, fixed-design workbench launch, cohort/role comparisons and evidence checks as fb36dd5 while the broader sample completed. MAP-004 then completed with 77 generations, exit 0 and both hook checks passing. Source/trial verification passed for all eight archived files and 72 intervention cases. 40/72 replies changed tokens, 32 were identical. No evening baseline truncation was repaired; two door conditions additionally capped. Several distinct channel changes produced the same alternate disagreement wording, underscoring that token divergence is not a unique personality feature.

Added all 12 sampled candidates to the result table and provisional structural/imagery/damage observations to the catalog. No traits are marked replicated. Current validation: 18 tests pass, JavaScript syntax passes, actual mapping API launch/completion is verified, and browser visual QA remains pending. Next milestone should diversify prompts and calibrate intervention sizes rather than assigning personalities from these sparse examples.

Dependency follow-up: GitHub's repository Dependabot API now returns zero open alerts. Both next-step runs are finished (51 + 77 = 128 new generations), verified and reported; no GPU job is left running. The local workbench remains available for reviewing full outputs and saved job logs.

### CAL-001 / MAP-005: a larger calibrated milestone

User requested going further to real milestones. Predeclared activation-statistic collection for all 184,320 MLP channels, 96 sampled channels across six layers, 5%/20% local post-MLP-normalization targets at both signs, and actual BF16/next-token-KL checks on four calibration inputs. Selection is a median-effect rule using screen measurements only. The 16 evaluation prompts are disjoint from the 12 calibration inputs and include six exact task/format checks. No desired personality is specified.

Added an optional matplotlib analysis extra for reproducible repository figures; no global/admin installs. Existing repository Markdown reports and local UI remain the delivery surfaces. Applied the data-visualization skill's chart/scoping guidance to the evidence figures. Core calibration, cleanup and selection tests passed with the existing suite (21 tests total) before generation. Source/config/prompt files will remain unchanged during execution.
