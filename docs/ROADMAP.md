# Experiment board

Status: backlog / active / complete / blocked / inconclusive. Completion means an artifact or measurement exists, not that a hypothesis succeeded.

| ID | Status | Idea / deliverable | Notes / next evidence |
|---|---|---|---|
| LAB-001 | complete | Organized local lab, pinned environment, GPU check | Native Windows; no admin. CUDA works; exact dependencies in uv.lock. |
| LFM-001 | complete | Pinned LFM3 source inspection and baseline | Five baselines recorded; gain=1 hook matched tokens on one prompt. See experiment report. |
| LFM-002 | active | Hamster LFM | Two sweeps complete, no usable recipe. Next: cleaner feature extraction and targeted channel search. |
| LFM-003 | active | Theatrical villain LFM | Stronger-dose follow-up degraded output. Next: neutral-system baseline and better feature separation. |
| LFM-004 | active | Contrast-derived activation directions | Extraction and sweep implemented; candidate quality not validated. |
| LFM-005 | backlog | Individual-channel causal search | Compare additive pushes, gains, ablation; use matched controls. |
| LFM-006 | backlog | Hamster + villain mixtures | Test interaction and dose, not just cherry-picked examples. |
| LFM-007 | backlog | Capability and format regression checks | Accuracy, completion, repetition, language mixing and latency. |
| LFM-008 | backlog | Transfer to LFM3 base / VL / Omni | Shared backbone does not guarantee interchangeable directions. |
| LFM-009 | backlog | Useful controls | Concision, uncertainty, JSON reliability, tool selection. |
| QWN-001 | backlog | Push Qwen3.5-0.8B to its limits | Reproduce released edit; study accuracy/latency/memory frontier. |
| CPU-001 | backlog | Exact computation inside a model | Start with arithmetic or memory; later investigate Qwenputer. |
| UI-001 | active | Local experiment workbench | Run history, comparisons, charts, controls, notes and logs implemented; API tests pass. Visual QA blocked by browser-tool limitation. |
| UI-002 | backlog | Live interactive single-run playground | Persistent loaded model, per-prompt tweaks and saved recipes; avoid repeated model load. |
| EVAL-001 | backlog | Explicit scoring and capability suite | Manual rubric first; do not treat keyword charts as persona quality. |
| MAP-001 | complete | Untargeted channel pilot | 41 generations; followed by replication, expanded sampling and calibration. See EXPLORATION.md. |
| MAP-002 | active | Effect catalog with replication status | First tentative observations and explicit no-effect case recorded. No traits replicated yet. |
| MAP-003 | complete | Completion-candidate replication | 51 generations. Anchor shortening repeats; no fresh EOS repair. General completion criterion not met. |
| MAP-004 | complete | Expanded untargeted sample | 77 generations; 40/72 changed, 32 identical. Two additional capped replies; no stable personality label. |
| CAL-001 | complete | Measured intervention calibration | 184,320-channel statistics; 96 sampled channels, 384 conditions, 1,536 screen forwards; selection verified. |
| MAP-005 | complete | Diverse calibrated effect map | 210 mapping records + 18 recovery checks; 117/192 changed; no strict-check gain; figures/numeric export published. |
| MAP-006 | active | Fresh compression/completion replication | 158 generations predeclared; target/control choices and criteria frozen before fresh generation. |
| CHAR-001 | complete | Sparse villain/hamster character discovery | 170 generations; 73/150 sparse replies changed; no coherent internal persona; prompted controls visibly characterize. |
| CHAR-002 | active | Combined character profiles and escalation | Four layers together, ×1/×2/×4 amplitude; 86 generations predeclared after CHAR-001; preserve damage. |

## Updating an idea

Keep the ID stable. Add dated evidence to JOURNAL.md and link run directories. Record unsuccessful attempts and changed hypotheses. Move results into experiments/ with a concise report once reviewed.
