# BACKEND-001: Apple Silicon installation and compatibility

Status: completed short setup/compatibility milestone, 2026-11-09. No anti-loop, personality, capability or performance-improvement claim. These checks do not continue the interrupted Windows QWN-002 pilot.

## Environment and integrity

Apple M3 Pro, arm64, 18 GiB unified system memory, macOS 27.3. Existing user-installed uv 0.9.4, project Python 3.11.15, torch 2.14.1, Transformers 5.22.2, huggingface-hub 1.33.0, accelerate 1.15.0 and safetensors 0.8.0. Installed the locked environment plus analysis/eval extras. MPS/BF16 works without enabling CPU operator fallback, custom kernels or changing system settings. CUDA dispatch remains implemented/tested; actual Windows hardware was not available for replay.

All model revisions remain those in configs/models.json. LFM3's complete 3,457,635,448-byte checkpoint matches Hub SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. A new CPU audit of both Qwen checkpoints compared all 488 tensors and exactly matched the published audit: only two expected down-projections differ, 64 columns each, 130,188 bitwise changes. Both loaded MPS edited/base tensor digests also match their audited bytes.

[Reviewed numeric evidence](evidence/BACKEND-001/metrics.json) records model pins, runtime, source/raw-output hashes, all checked cache positions and per-case counts. Full replies, token sequences and exact source snapshots remain local.

## What ran

| Check | Observation | Limit |
|---|---|---|
| Five LFM3 baselines + gain=1 | Neutral hook token-identical; identity, fact and JSON reach EOS | Arithmetic/story hit 96 tokens; no task-quality gain |
| LFM3 SDPA full-prefix vs incremental cache | 17/17 next-token argmax matches; finite logits; max absolute difference 0.3647461 | One short fixed input, not cache equivalence at long lengths |
| Qwen base SDPA cache | 17/17 matches; finite; max absolute difference 0.546875 | Same limited teacher-forced check |
| Qwen edit SDPA cache | 17/17 matches; finite; max absolute difference 0.46875 | Same limited teacher-forced check |
| Qwen thinking-on/template/seed checks | Both variants pass template distinction/open prefix, first-logit checks and identical repeated eight-token samples | 32 generated sanity tokens total; no 4K/8K/32K trial |
| LFM3 paired HTTP unchanged control | Both return READY with identical tokens and EOS | One prompt |
| Saved Cookie Diplomat hook via HTTP | Default 54 tokens, edit 35; both EOS; tokens differ | Selected historical cookie anchor; no new persona validation |
| Qwen base/edit HTTP conversation | Both return 43, then 44 with saved branch histories; EOS | Tiny greedy arithmetic smoke only |

Greedy LFM baseline uses the existing baseline prompt file, seed 42, eager BF16 and 96-token output cap. HTTP cases use seed 42, thinking off, 64-token cap, the recorded shared neutral instruction, SDPA for default/Qwen and the saved recipe's eager attention for Cookie Diplomat. The cookie comparison retains the same attention within its pair. No weights changed; live parameter-version checks pass.

Qwen short sampled checks reuse temperature 0.6, top_p 0.95, top_k 20, output-only presence penalty 1.5 and seed 0, capped at eight tokens per sample. Their full raw samples/settings are saved inside the diagnostic, separately from the Windows pilot. Numerical differences above and maximum KL per position remain visible in the export; equal argmax does not mean equal distributions.

## Local evidence

- `20261109T193533110500Z-LFM-001-baseline`: six generated records, neutral check and source snapshots.
- `20261109T193729827265Z-BACKEND-001-cache-qwen08`: both variants, 34 teacher-forced positions and 32 sanity tokens.
- `20261109T193758035031Z-BACKEND-001-cache-lfm3`: 17 teacher-forced positions.
- `20261109T193831150180Z-LIVE-lfm3`: unchanged control.
- `20261109T193837681226Z-LIVE-lfm3`: saved nonzero recipe.
- `20261109T193847017965Z-LIVE-qwen08`: base/edit arithmetic.
- `20261109T193850745832Z-LIVE-qwen08`: continued histories.

The original Windows raw runs and handoff ZIP were absent. We checked committed reports, numeric selection/criteria and recipe identities, not unavailable raw historical text. The new local catalog still contains 135 public recipes and adds local LIVE observations without publishing private conversations. The inference worker was stopped after checking; the loopback viewer can remain available for manual use.

## Verification and next step

Python regression suite, Node comparison/startup/model-isolation checks and real-API DOM startup pass on Mac; final counts are in the [review](../docs/REVIEW-2026-11-09.md). Memory labels distinguish CUDA peaks from MPS current allocation. Failed summaries cannot be shown as completed; live and managed launches share a lock and stop targets owned process trees.

Visual/keyboard/responsive browser QA remains incomplete: no isolated browser tool was available and no native desktop automation was used. Long MPS campaigns, full CPU checkpoint inference and changed Windows hardware inference are not verified. Follow [NEXT-STEPS.md](../docs/NEXT-STEPS.md) to freeze a new bounded Mac protocol; the historical QWN-002/PERF designs reject MPS.
