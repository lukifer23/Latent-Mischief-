# Next milestone and decision gates

Updated 2026-11-06. This is the ordered work plan, not a claim that its experiments have run. The [roadmap](ROADMAP.md) tracks the broader ideas; [current evidence](../README.md#current-evidence) states what is already verified.

**Current override:** the evaluator/protocol/full audit are implemented and the first pilot was stopped by user request at 6/8 saved trajectories. Read [HANDOFF.md](HANDOFF.md) for accurate partial results and Mac prerequisites. The unchecked list below preserves the original plan; it is not a current completion counter. Do not restart the interrupted sample or assume Mac inference works.

## First: QWN-002 — evaluate the published Qwen edit properly

**Question:** does the pinned 0.8B edit reduce unproductive repetition/completion failures on our native Windows setup while retaining correct answers?

Both checkpoints already load and pass tiny arithmetic/conversation smoke checks. Those checks do not test the claimed anti-loop effect. The current live worker is greedy with a maximum 2,048-token API output budget. It cannot reproduce the author's thinking-on, sampled, up-to-32,768-token evaluation.

The pinned [released model card](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/README.md) documents temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, thinking on, seed 0 and last-boxed-answer grading. Its re-run used vLLM and an H200; ours currently uses Transformers/PyTorch reference kernels on a 16GB Windows GPU. Backend and output-budget differences must remain explicit. A smaller local pilot is an exploratory comparison, not an exact benchmark reproduction.

### Work checklist, in order

- [ ] Write `experiments/QWN-002-anti-loop-pilot.md` and a versioned config **before generation**. Freeze exact dataset revision, four problem IDs, prompts/template, expected answers, both checkpoint revisions, sampling, EOS policy, grader, execution order and budgets. Record selection without looking at these checkpoints' generated answers.
- [ ] Implement a separate model-aware batch evaluator; do not silently change the existing greedy chat path. Validate thinking-template handling, sampling, seed recording, presence-penalty semantics, EOS termination and resumed-run behavior. Presence penalty is not interchangeable with repetition penalty. If upstream behavior cannot be matched, document the deviation before running.
- [ ] Verify all model tensor differences against the upstream base, extending the existing two edited-tensor digest checks. Record which tensors/columns differ and preserve the original checkpoints. Distinguish publisher claims from our own verification.
- [ ] Execute the bounded pilot: four problems × two checkpoints × one fixed seed = **eight planned trajectories**, thinking on, identical settings within each pair, cap **4,096 output tokens each**, at most **32,768 generated tokens total** and **30 minutes wall time**. Run serially in the existing project environment; save progress after every trajectory. These are proposed limits until the protocol/config is committed.
- [ ] Export every trajectory, paired correctness, EOS/cap status, repeated-token indicators, manual reasoning-loop review, input/output tokens, timings and allocated/reserved memory. Include raw-output hashes, full settings/source archives, failures and missing pairs. A cap hit alone is not a loop; EOS alone is not correctness.
- [ ] Publish the report and numeric comparison, then make the next decision from the complete pilot. Do not select prompts, change the cap or reroll samples after seeing outcomes within the same experiment ID.

### Acceptance and stopping rules

The milestone succeeds when the evaluator and complete evidence are reviewable, including a negative result. A benefit is only a **pilot observation** when an edited reply improves a paired outcome under the same settings; report any paired regression alongside it. Eight short-budget trajectories cannot validate the published benchmark percentages or a general capability gain.

If neither model exhibits repetition, or both merely exhaust the smaller cap, the anti-loop question remains unresolved. Predeclare a separate higher-budget experiment or fresh sample; do not call absence of looping proof of success. Any promising result needs a separately frozen fresh set and multiple seeds before a broader claim.

Stop and preserve partial evidence at the wall/token limit, OOM, non-finite outputs, model/source mismatch or broken scoring. Do not install drivers, compilers, system services or paid compute to force completion. Give a concise progress update before GPU work; this documentation update starts no inference.

## Then, in priority order

| Track | Concrete next action | Completion evidence |
|---|---|---|
| Workbench usability | Check chat → saved comparison → recipe naming → metrics using a genuinely isolated browser surface, only when available | Step screenshots and interaction/keyboard findings; until then visual QA stays incomplete. No native desktop control or focus changes. |
| LFM3 effect mapping | Propose a new controlled search after reviewing failed character/replication results; separate tone, content, structure and damage, with prompting/random controls and fresh checks | New predeclared protocol and all scoped comparisons. Do not keep escalating the failed profiles or rename them validated personas. |
| Actual 128K context | Investigate allocator/cache pressure using the frozen PERF-001 input IDs; validate any cache change at smaller sizes before another 128K attempt | Repeated matched timings, actual token counts and explicit failures. Keep the failed expandable allocator disabled; no extrapolated 128K throughput. |
| Longer-term ideas | Keep Qwenputer, LFM family transfer and character mixtures on the idea board | Advance only with a specific testable protocol; they are not prerequisites for QWN-002. |

Every completed milestone updates README, ROADMAP and JOURNAL and is pushed to `main`. Original weights, private conversations and raw run directories remain local. Exact recipe identities and provisional persona names keep their existing documentation rules.
