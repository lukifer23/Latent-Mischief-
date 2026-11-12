# Experiment board

Status: backlog / active / complete / blocked / inconclusive. Completion means an artifact or measurement exists, not that a hypothesis succeeded.

**Execution order:** follow [Next steps and decision gates](NEXT-STEPS.md). Current priority is LFM-010: the user chose a small phase-timing/character experiment. Offline implementation is verified; MPS execution waits for resource admission. QWN-003 stays separate after its resource refusal; QWN-002 remains interrupted.

| ID | Status | Idea / deliverable | Notes / next evidence |
|---|---|---|---|
| LAB-001 | complete | Organized local lab, pinned environment, GPU check | Native Windows; no admin. CUDA works; exact dependencies in uv.lock. |
| BACKEND-001 | complete | Mac environment and bounded compatibility | M3 Pro / 18 GiB / MPS BF16: LFM3 baselines, Qwen base/edit, loaded digests, 17-position cache checks and paired HTTP conversations pass. No long MPS study or new Windows hardware replay. See [report](../experiments/BACKEND-001-mac-setup.md). |
| LFM-001 | complete | Pinned LFM3 source inspection and baseline | Five baselines recorded; gain=1 hook matched tokens on one prompt. See experiment report. |
| LFM-002 | active | Hamster LFM | Two sweeps complete, no usable recipe. Next: cleaner feature extraction and targeted channel search. |
| LFM-003 | active | Theatrical villain LFM | Stronger-dose follow-up degraded output. Next: neutral-system baseline and better feature separation. |
| LFM-004 | active | Contrast-derived activation directions | Extraction and sweep implemented; candidate quality not validated. |
| LFM-005 | backlog | Individual-channel causal search | Compare additive pushes, gains, ablation; use matched controls. |
| LFM-006 | backlog | Hamster + villain mixtures | CHAR-002 combined same-character profiles across layers without a stable character. A hamster-plus-villain mixture remains untested. |
| LFM-007 | backlog | Capability and format regression checks | Accuracy, completion, repetition, language mixing and latency. |
| LFM-008 | backlog | Transfer to LFM3 base / VL / Omni | Shared backbone does not guarantee interchangeable directions. |
| LFM-009 | backlog | Useful controls | Concision, uncertainty, JSON reliability, tool selection. |
| LFM-010 | active | Phase-specific sparse steering pilot | Implemented/offline verified: prefill/decode/both gates, fresh MPS calibration, baseline/prompt/random controls, 54 short replies and blinded review. First admission refused before loading weights: 0/54 replies, 6.395 GiB available versus 7-GiB floor. Normal pressure and unchanged admission policy still required. See [protocol](../experiments/LFM-010-phase-timing.md). |
| LFM-010 | blocked | Phase timing on MPS | Frozen; one zero-output resource refusal on the Mac. Superseded for evidence by LFM-011 on CUDA; not pooled. |
| LFM-011 | complete | Phase timing, native Windows CUDA | 54/54 verified. Villain steering at 5% changed 1-2/6 replies per phase, random similar, no villain voice on direct reading; blinded review pending. Likely negative. |
| LFM-012 | active | Prompt-to-activation transfer on MPS | Predeclared: mean residual shift of the villain/hamster system prompts added at all positions; teacher-forced gap-closed metric, content/random/cross-character controls, frozen select/test split and decision rule. See [protocol](../experiments/LFM-012-prompt-transfer.md). |
| QWN-001 | active | Push Qwen3.5-0.8B to its limits | Base/edit pinned and edited tensors digest-verified; local arithmetic + conversation smoke pass. Anti-loop benchmarks and long context unverified. |
| QWN-002 | inconclusive | Bounded thinking-on base/edit pilot | Evaluator implemented; predeclared aedb63b. User-stopped at 6/8 saved, two matched pairs, 832 partial tokens. No final-answer/completion benefit at 4K. Preserve Windows evidence; Mac follow-up needs a new protocol. |
| QWN-003 | active | Resource-bounded MPS budget follow-up | Protocol/source precommitted 18cf314. First admission refused at OS pressure 2: 0/4 trajectories, no weights loaded. Guards/tests and numeric refusal evidence complete; actual study waits for normal pressure. See [report](../experiments/QWN-003-mac-followup.md). |
| QWN-003 | blocked | Mac 8K follow-up | Resource-refused on MPS (0 trajectories). Superseded by QWN-004; its preflight needs the mask fix (applied) before any retry. |
| QWN-004 | complete | 8K paired follow-up on CUDA | 4/4, all capped, no boxes, none correct; edit lower repetition only. No completion benefit observed. |
| CPU-001 | backlog | Exact computation inside a model | Start with arithmetic or memory; later investigate Qwenputer. |
| UI-001 | active | Local experiment workbench | Run history, comparisons, charts, controls, notes and logs implemented; API tests pass. Visual QA blocked by browser-tool limitation. |
| UI-002 | active | Live interactive paired playground | Model selector, recipes, default/edit conversations and timing implemented; GPU/API smoke and DOM-fixture switching pass. Visual QA blocked. |
| EVAL-001 | backlog | Explicit scoring and capability suite | Manual rubric first; do not treat keyword charts as persona quality. |
| MAP-001 | complete | Untargeted channel pilot | 41 generations; followed by replication, expanded sampling and calibration. See EXPLORATION.md. |
| MAP-002 | active | Effect catalog with replication status | First tentative observations and explicit no-effect case recorded. No traits replicated yet. |
| MAP-003 | complete | Completion-candidate replication | 51 generations. Anchor shortening repeats; no fresh EOS repair. General completion criterion not met. |
| MAP-004 | complete | Expanded untargeted sample | 77 generations; 40/72 changed, 32 identical. Two additional capped replies; no stable personality label. |
| CAL-001 | complete | Measured intervention calibration | 184,320-channel statistics; 96 sampled channels, 384 conditions, 1,536 screen forwards; selection verified. |
| MAP-005 | complete | Diverse calibrated effect map | 210 mapping records + 18 recovery checks; 117/192 changed; no strict-check gain; figures/numeric export published. |
| MAP-006 | complete | Fresh compression/completion replication | 158 generations; L04 shortening 6.81%, below 10% criterion; L20 no fresh cap repair. Neither primary criterion met. |
| CHAR-001 | complete | Sparse villain/hamster character discovery | 170 generations; 73/150 sparse replies changed; no coherent internal persona; prompted controls visibly characterize. |
| CHAR-002 | complete | Combined character profiles and escalation | 86 generations; 56/63 internal replies changed; no stable character, hamster/random check regressions. Figures and negative results published. |
| PERF-001 | active | Actual context performance | Six paired timings through 32K input; actual 128K attempt OOM. Reduce allocator/cache pressure before retrying frozen inputs. |
| PERF-002 | complete | Expandable allocator attempt | Failed at model load with GPU mapping error; no timing. Not enabled for chat. |
| LIVE-001 | complete | Paired model inference smoke | LFM unchanged control identical; nonzero hook changes response; Qwen base/edit answer 43 then 44 with saved histories. No anti-loop/quality claim. |
| DOC-001 | complete | Exact recipe / persona registry | 135 model-specific cards, method fingerprints, scoped observations, provisional naming and local-only chat records. |

## Updating an idea

Keep the ID stable. Add dated evidence to JOURNAL.md and link run directories. Record unsuccessful attempts and changed hypotheses. Move results into experiments/ with a concise report once reviewed.
