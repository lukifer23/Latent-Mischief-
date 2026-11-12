# LFM-012: does an ordinary persona prompt transfer as a residual direction?

Status: **blocked on host memory; no scientific result**. Protocol, runner, verifier and tests are complete. Four attempts on the Mac stopped on macOS memory pressure. One produced 88 of 108 unsteered reference replies; none built a direction or ran a steered forward. Stopped at the user's request on 2026-11-11 with five of eight segments unused. Related roadmap IDs: LFM-004, LFM-012.

## Why this experiment

Every internal character attempt so far has failed, while an ordinary system prompt characterizes clearly. Two things about those attempts limit what the failures mean:

| Earlier attempt | Intervention | Readout |
|---|---|---|
| LFM-002 / LFM-003 | Dense contrast direction, **last position only**, from four prompts; nothing between "no effect" (0.35) and "garbled" (0.7) | Keyword hits and direct reading |
| CHAR-001/002, LFM-010/011 | **Sixteen MLP channels** at a 5% local strength | Changed-token counts and direct reading |

Neither tested the plainest version of the idea: take the shift that the working system prompt itself produces in the residual stream, add it at **every** position, and measure the result on a continuous scale. That is this experiment. It is a new protocol on a new prompt set, not an escalation of any earlier run.

## Question

With the default system prompt, does adding the mean residual shift between prompted-character replies and default replies make the model behave as if it had been given the character prompt, more than matched controls do, on prompts that played no part in building or choosing the direction?

Hypothesis: a mid-depth layer and a strength of a few natural shifts recovers a clear share of the prompt's effect, specifically for its own character. The hypothesis may be false; a measured null is a valid outcome.

## Setup

LiquidAI/LFM3-1.7B at revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`, macOS MPS, BF16, eager attention, greedy decoding, seed 42, thinking requested off, cached generation, checkpoint generation defaults preserved and recorded. No weights change. Frozen values are in [lfm-prompt-transfer.json](../configs/lfm-prompt-transfer.json); prompts are in [transfer-prompts.json](../prompts/transfer-prompts.json).

Three system prompts share one base instruction: **default**, **villain** (the LFM-010 wording) and **hamster**. All 36 prompts are new and carry no character wording. They are split before any inference:

| Split | Prompts | Used for |
|---|---:|---|
| extract | 16 | Building directions only |
| select | 8 | Choosing one layer and strength per character |
| test | 8 | The reported result; never used for choosing |
| check | 4 | Exact-answer damage checks (two sums, JSON, chemical symbol) |

## Directions

Reference replies are generated for every prompt under each system prompt (at most 40 new tokens). For the extract split, one uncached forward over prompt plus reply records the block output of every layer, averaged over non-special reply tokens.

- **villain**, **hamster**: mean over the 16 extract prompts of (character reply state minus default reply state), each under its own system prompt.
- **content** (natural-variation control): mean default-reply state of the first eight extract prompts minus the last eight. It has realistic activation geometry but no character meaning.
- **random-1..3**: seeded Gaussian directions.

Every direction is normalised to unit length. All six are applied at one **common scale** per layer, the average of the two character shifts' natural norms, times a strength α. So α = 1 is about one natural prompt-induced shift, and every control receives exactly the same magnitude. The intervention is added to the block output (the residual stream) at **all positions** of every forward: prompt tokens during prefill and each new token during decode. This differs from LFM-002 (last position only) and from the sparse MLP-channel experiments.

## Measurement

**Gap closed (primary, teacher-forced).** Take the reply the character prompt actually produced. Its negative log-likelihood is high under the default prompt and low under the character prompt. With the default prompt plus an intervention it lands somewhere between:

`gap closed = (NLL default − NLL steered) / (NLL default − NLL character prompt)`

summed over a split's prompts. 1 means the intervention is as good as the system prompt; 0 means nothing happened; negative means the reply became less likely. An intervention that merely breaks the model raises NLL and scores below zero, so the metric cannot be gamed by a larger dose. The same number is computed for every control and for the *other* character's replies. The NLL of the model's own default reply under each intervention is reported as the cost side.

**Select grid.** Layers 4, 8, 12, 16, 20 × α 1, 2, 4, 8 × six directions × three reply types × eight select prompts = **2,880** forwards.

**Frozen selection rule.** For each character, the cell with the largest own-direction gap closed on the select split; ties go to the smaller α, then the earlier layer. Nothing from the test split enters.

**Test (reported).** At each character's selected cell, all six directions × three reply types × eight test prompts = **288** forwards. Then free generation with the default prompt at that cell for the eight test and four check prompts, with the character direction and with random-1 at identical scale: **48** generations.

**Style score for free generations.** Each generated reply is scored, without any intervention, under the default and character prompts. `(NLL default − NLL character) / tokens` is the model's own per-token evidence that the text came from the character prompt. Default replies sit low, prompted replies high. This is a model-based measure, not a human rating; replies stay available for direct reading and a few are quoted in the results.

## Frozen decision rule

A character direction is **nominated for a separately frozen fresh replication** only if all hold on the test split at its selected cell:

| Criterion | Requirement |
|---|---|
| C1 transfer | Own gap closed ≥ 0.25 |
| C2 specificity | Own gap closed exceeds the best of the other five directions by ≥ 0.10 |
| C3 consistency | Character reply more likely than at baseline on ≥ 7 of 8 test prompts |
| C4 generated style | Style score above the default reply on ≥ 7 of 8 test prompts, and mean above random-matched generations |
| C5 damage | No exact-answer check that the default passes is lost |

Seven of eight in one direction has a one-sided sign-test probability of about 0.035 under no effect. Every criterion is reported individually whether or not it passes. Meeting all five nominates a replication; it does not validate a character representation, and nothing here bears on capability. If the selected cell fails, the result is a measured negative for this method, with the full grid published.

## Budget and resource policy

158 generations × at most 40 tokens (**6,320** generated tokens), 3,168 steered teacher-forced forwards plus unsteered scoring forwards, inputs capped at 160 tokens, **30-minute** cumulative wall ceiling including identity check and load. No retry, no dose beyond the grid, no replacement prompts.

This Mac has 18 GiB shared with the system and other applications. BACKEND-001 measured about 3.46 GB of MPS tensor allocation for LFM3 at BF16. The policy for this protocol is therefore: normal macOS memory pressure and **5 GiB available** inside the initialized process before loading; stop below **1 GiB available**, above **512 MiB** swap growth, above **6 GiB** RSS or MPS driver allocation, on any thermal or performance warning or missing telemetry, or below 5 GiB free disk. MPS allocator fraction 0.45, two CPU threads, niceness +10. LFM-010's 7 GiB floor belongs to that frozen protocol and is left unchanged there; this is a new protocol with its own stated floor, and the pressure, swap and thermal stops are what protect concurrent work. A resource stop preserves what was saved and is reported as interrupted, never as a result.

## Amendment before any steered forward: exact stage-level continuation (2026-11-11)

Two attempts under the policy above stopped on macOS memory pressure (details in the results section). Neither ran a steered forward, built a direction or selected anything, and no reply was read before this amendment. The limits are **unchanged**. What changes is how a stop is handled.

Every unit of this design is an independent, deterministic computation: a greedy generation or one uncached forward. Nothing depends on carried KV or sampler state. A stopped segment can therefore be continued exactly:

- `resume RUN-ID` creates a **new** directory, refuses unless every archived source hash and the config hash equal the parent's, and copies the parent's saved rows and direction tensors with their hashes recorded.
- Before using inherited rows it regenerates the first default reference reply and requires identical input and output tokens. A mismatch refuses the continuation.
- Saved units are skipped, never recomputed or replaced. A grid cell is saved only when complete. An inherited selection must equal the recomputed one.
- The 30-minute wall ceiling is **cumulative across segments**. At most **eight** segments. A segment starts only when its own admission check passes; nothing retries automatically.
- The reporter verifies the whole chain: identical code and config in every segment, each child's rows extending its parent's exactly, parent files unchanged since inheritance, and parameter versions unchanged in every segment that loaded the model.

Stopping when the system needs memory and continuing later is what a shared 18 GiB machine requires; relaxing the pressure stop would instead push other work into swap. The two earlier directories were produced by the pre-amendment runner and are preserved but not inherited; the amended run starts from zero.

## Commands

```sh
uv run --locked python scripts/transfer_experiment.py plan      # offline; imports no torch
uv run --locked python scripts/transfer_experiment.py run       # only after committing source, config and this protocol
uv run --locked python scripts/transfer_experiment.py resume RUN-ID   # continue a stopped segment exactly
uv run --locked python scripts/transfer_report.py RUN-ID --publish
```

## Evidence kept

Archived sources with hashes, predeclared commit, weight identity, generation defaults, every reply with input and output token IDs, direction tensors and diagnostics (norms, cosines, per-prompt agreement), every teacher-forced row (split, layer, α, direction, prompt, reply type, summed NLL, token count), the saved selection, zero-scale and removed-hook token-identity checks, parameter-version check, resource timeline and cleanup. The reporter reloads the archived plan, verifies the rows are exactly the frozen product, recomputes the selection and every published number, and refuses to publish if any differ. Raw text and tensors stay local; numeric evidence and a figure go to `experiments/evidence/LFM-012/`.

## Limits stated in advance

One model, one backend, two characters whose prompts differ in vocabulary as well as voice, greedy decoding, eight test prompts and three random directions. Reply states under a character prompt mix style with the content that style brings, so a direction may carry topic as well as voice. The gap-closed metric rewards making one specific prompted reply likelier; it is not a general style classifier. Replies are capped at 40 tokens. Strength is matched across directions by norm at the intervention site, not by downstream effect.

## 2026-11-11: attempts so far (no result)

All four stopped with `ResourceLimitError: macOS memory pressure is elevated or unknown`. Directory names are UTC. No reply text has been read or used for any decision; no direction, grid cell or selection exists.

| Run | Commit | Stopped at | Saved | Seconds | What the timeline shows |
|---|---|---|---|---:|---|
| `20261112T031956558125Z` | `9658d49` | loading model | 0 replies | 12.3 | 5.49 GiB available at start; load pushed pressure to 2 with 2,009 MiB swap growth |
| `20261112T032102590331Z` | `6e5ce6a` | reference generation | 88 / 108 reference replies | 124.1 | Loaded cleanly (about 6 GiB available, no swap growth, 29.7 tokens/s). At about 117 s available memory fell from 5.3 to 2.8 GiB in seven seconds while this process stayed flat at 4.06 GiB MPS: another workload paged back in |
| `20261112T032540576244Z` | `5957e35` | loading model | 0 | 16.4 | Amended runner, segment 1; pressure 2 during load, 1,543 MiB swap growth |
| `20261112T032613233119Z` | `5957e35` | loading model | 0 | 13.2 | Segment 2 (continuation of the row above); started at 7.78 GiB available, fell to 4.13 GiB with only 2.04 GiB loaded while other jobs started |

The first two used the pre-amendment runner and are preserved but not inherited. The last two form one chain that has used 29.6 s of the 1,800 s cumulative budget and two of eight segments, with nothing computed. Numeric interruption evidence for the latest segment is in [metrics.json](evidence/LFM-012/metrics.json).

**Observations.** LFM3 at BF16 holds about 3.47 GB of MPS tensors (4.06 to 4.36 GiB driver allocation). The load is the fragile step: psutil's "available" figure was above the 5 GiB floor each time, but most of it was inactive pages belonging to other applications, and making room for 4 GiB of GPU memory compressed or swapped them. System swap rose from about 1.9 GB to about 5.7 GB of 7.2 GB over the session; other projects' jobs were running at the time. Once loaded with real free memory, the model ran steadily for two minutes until an unrelated burst.

**Hypothesis, not observation.** With the other heavy jobs idle and roughly 7 GiB of genuinely free (not merely inactive) memory, one or two segments would finish the roughly ten-minute run. The start floor counts inactive pages and is therefore a weak admission test on this machine; a floor on free pages would be stricter and would have refused instead of stopping mid-load. That would be a further amendment and has not been made.

**To continue.** Nothing is running. When the machine is quiet:

```sh
uv run --locked python scripts/transfer_experiment.py resume 20261112T032613233119Z-LFM-012-prompt-transfer
uv run --locked python scripts/transfer_report.py RUN-ID --publish   # after a COMPLETE line
```

Each further stop leaves a new directory to resume from; always resume the newest. Do not relax the pressure, swap or thermal stops. Running on the Windows CUDA machine instead would be a separately declared backend experiment, as LFM-011 was for LFM-010.
