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
| MAP-001 | active | Untargeted channel exploration | First six-channel pilot complete (41 generations). Next: broader sampling and replication. See EXPLORATION.md. |
| MAP-002 | active | Effect catalog with replication status | First tentative observations and explicit no-effect case recorded. No traits replicated yet. |
| MAP-003 | complete | Completion-candidate replication | 51 generations. Anchor shortening repeats; no fresh EOS repair. General completion criterion not met. |
| MAP-004 | active | Expanded untargeted sample | Predeclared 12 additional channels excluding the original pilot; results pending. |

## Updating an idea

Keep the ID stable. Add dated evidence to JOURNAL.md and link run directories. Record unsuccessful attempts and changed hypotheses. Move results into experiments/ with a concise report once reviewed.
