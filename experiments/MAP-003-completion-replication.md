# MAP-003: completion-candidate replication specification

Status: complete; anchor shortening reproduced, general completion repair not established.
Run: runs/20261106T185739757150Z-MAP-003-channel-exploration/

## Question and design

Does the pilot observation at L10N0204 generalize to fresh prompts, and is it more specific than changing other channels in the same layer?

LFM3-1.7B at the pinned revision and patched locked runtime. Same neutral system instruction as the pilot. Greedy decoding, 192-token cap, gains 0/1.5/2 on the pilot-selected channel. Three uniformly selected same-layer channels at gain 2 are matched controls for the gain-2 comparison only. Candidate selection seed 2718; intervention trial order shuffled deterministically. No weight edits.

Seven prompts: six fresh prompts plus the original evening anchor. Analyze the fresh cohort separately; anchor reproduction is not fresh validation. 7 baseline + 2 hook checks + 21 target replies + 21 control replies = 51 generations. All condition outputs are retained. A fixed regex set records explicit self-check phrases and think-boundary markers; these are fallible surface indicators, not internal reasoning measurements. One exact-JSON check is a narrow format guard.

## Interpretation rule fixed before inspection

Report per-prompt matched changes in EOS and explicit self-check markers at gain 2. A general completion claim requires improvements on at least two fresh prompts, no new EOS losses on other fresh prompts, and comparison against all three control channels. If the fresh baseline has too few relevant failures, report that the test is uninformative for repair rather than calling an unchanged good response a success. Lower gains and ablation are exploratory dose comparisons. Six fresh prompts and three control channels do not justify broad significance or personality claims.

An effect that repeats only on the anchor remains discovery-specific. Record format regressions, longer answers, altered imagery and no-effect cases independently.

## Results

51 generations completed; gain=1 and hook-removal checks both passed. The specification and source were committed as d813ab2 before generation. No prompt or candidate changes were made during the run.

| Fresh cohort only | Replies | Reached EOS | Think-boundary marker present | Explicit self-check phrase |
|---|---:|---:|---:|---:|
| Unsteered | 6 | 5 | 2 | 0 |
| L10N0204 gain 0 | 6 | 5 | 2 | 0 |
| L10N0204 gain 1.5 | 6 | 5 | 2 | 0 |
| L10N0204 gain 2 | 6 | 5 | 4 | 0 |
| Random L10N0107 gain 2 | 6 | 5 | 3 | 0 |
| Random L10N1780 gain 2 | 6 | 5 | 2 | 0 |
| Random L10N2578 gain 2 | 6 | 5 | 2 | 0 |

The locked-box prompt hit the budget in every condition. No fresh EOS failure was repaired and no fresh EOS success was lost. With only one baseline EOS failure, the fresh set offered limited room to test repair. The predeclared general completion criterion was not met. Think-boundary markers increased on the target's fresh replies at gain 2; this does not support a general suppression-of-reasoning interpretation.

On the original evening anchor, unsteered / gain 0 / gain 1.5 used 171 tokens and included a self-check and restarted answer. Gain 2 used 72 tokens with neither the explicit self-check nor a think-boundary marker. All reached EOS at the larger budget. Random controls used 172/144/144 tokens; two also omitted the fixed self-check phrases but retained a think-boundary marker. Phrase absence alone is therefore a weak effect label.

The exact-JSON prompt passed in all seven tested baseline/target/control conditions. This is one format check, not broad capability preservation.

## Interpretation

The original prompt-specific shortening reproduces. It has not generalized into a reliable completion or personality control. Retain the candidate as a narrow observation with a negative fresh-prompt completion result. Prior truncation at 128 tokens was budget-censored and was not a demonstrated loop. Broader mapping should continue without selecting only candidates resembling this effect.
