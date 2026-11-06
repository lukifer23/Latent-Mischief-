# Working agreements

- This is a local, LFM3-first research lab. No administrator privileges are available.
- Use the project `.venv` and `uv`; do not change global Python, drivers, system PATH, execution policy, or install system services.
- Start at README.md and docs/ROADMAP.md. Update docs/JOURNAL.md after meaningful work; keep hypotheses separate from observations.
- Preserve upstream weights. Begin with reversible inference hooks; no checkpoint rewriting without a separately documented experiment.
- Pin model revisions and dependencies. Record prompts, generation settings, seed, intervention placement and strength, environment, and raw outputs for every run.
- Keep model downloads, generated outputs, secrets and environments out of Git. Do not publish or use paid remote compute unless requested.
- Evaluate interesting effects on new prompts and against unsteered, prompt-persona and matched random controls. A persona effect is not a capability improvement.
- Native Windows and prebuilt wheels are the first path. Do not assume WSL, compilers, Triton, or FlashAttention are installed.
- Work directly on `main` only. Commit and push to origin/main at real, reviewable milestones; user has authorized regular remote updates. Do not create feature branches or use force pushes.
- Original project code and documentation use MIT. Do not imply that this relicenses upstream model weights or third-party work.
- Keep implemented / verified / exploratory / planned status explicit. Report concrete negative results as carefully as successes, with matched-control limits.
