# MAP-003: completion-candidate replication specification

Status: predeclared before generation; outcome pending.

## Question and design

Does the pilot observation at L10N0204 generalize to fresh prompts, and is it more specific than changing other channels in the same layer?

LFM3-1.7B at the pinned revision and patched locked runtime. Same neutral system instruction as the pilot. Greedy decoding, 192-token cap, gains 0/1.5/2 on the pilot-selected channel. Three uniformly selected same-layer channels at gain 2 are matched controls for the gain-2 comparison only. Candidate selection seed 2718; intervention trial order shuffled deterministically. No weight edits.

Seven prompts: six fresh prompts plus the original evening anchor. Analyze the fresh cohort separately; anchor reproduction is not fresh validation. 7 baseline + 2 hook checks + 21 target replies + 21 control replies = 51 generations. All condition outputs are retained. A fixed regex set records explicit self-check phrases and think-boundary markers; these are fallible surface indicators, not internal reasoning measurements. One exact-JSON check is a narrow format guard.

## Interpretation rule fixed before inspection

Report per-prompt matched changes in EOS and explicit self-check markers at gain 2. A general completion claim requires improvements on at least two fresh prompts, no new EOS losses on other fresh prompts, and comparison against all three control channels. If the fresh baseline has too few relevant failures, report that the test is uninformative for repair rather than calling an unchanged good response a success. Lower gains and ablation are exploratory dose comparisons. Six fresh prompts and three control channels do not justify broad significance or personality claims.

An effect that repeats only on the anchor remains discovery-specific. Record format regressions, longer answers, altered imagery and no-effect cases independently.
