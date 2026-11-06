# LIVE-001: local LFM3 and Qwen paired inference

Status: four paired API/GPU smoke cases passed. This verifies the implemented inference path and limited controls, not persona quality, anti-loop efficacy or benchmark improvement. Visual chat interaction QA remains blocked; see [QA record](../docs/UI-QA.md).

## Model identity and method

- LFM3: `LiquidAI/LFM3-1.7B` at `0dc4268b561499c648c7ba7c9466bf3d8d01517a`; verified architecture and checkpoint digest are documented in [PERF-001](PERF-001-long-context.md).
- Qwen base: `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`.
- Qwen published edit: `RikiyaT/Qwen3.5-0.8B-weight-space-programming` at `d1671a178705bcab7644806d23c535517f35723b`; both published edited down-projection tensor SHA-256 digests match. Exact expected digests are in [the registry](../configs/models.json). This is the 0.8B anti-loop recipe, not a transfer of the 4B Evil Qwen channel indices.

All cases used BF16, greedy manual cached decoding, seed 42, thinking off, a shared helpful/concise system instruction and 64-token output cap. The unchanged LFM control and Qwen used SDPA; the historical LFM hook recipe used its recorded eager backend for both branches. Hooks are reversible and implementation-fingerprinted; each completed pair checked unchanged parameter-version counters. Source archives/hashes, prompt, actual branch token counts, checkpoint identities and timing were independently read back into [reviewed evidence](evidence/LIVE-001/metrics.json). Routine private chats are excluded from public catalog publishing.

## Observations

| Case | Default | Edited / second control | What this establishes |
|---|---|---|---|
| LFM unchanged control: exactly READY | READY, 3 tokens, EOS | Identical token sequence, EOS | Same runtime/control can reproduce this greedy reply |
| LFM Cookie Diplomat recipe: playful response to last-cookie theft | 55 tokens, EOS | Different sequence, 64-token cap | Nonzero saved hook runs and changes output; adds a cap on this prompt |
| Qwen base versus published edit: 27 + 16 | 43, 3 tokens, EOS | 43, identical tokens, EOS | Both pinned checkpoints load and answer this arithmetic prompt |
| Qwen continued separate branch histories: add one | 44, 3 tokens, EOS | 44, identical tokens, EOS | Both saved histories feed the next comparison correctly |

The new LFM hook reply begins, “Ah, the cookie thief! I shall claim my crumbs as a tribute to your boldness.” It then speculates about a secret stash/shared treasure and reaches the cap. The historical **Cookie Diplomat** nickname remains provisional; this changed prompt does not establish a stable personality or beneficial edit. The default itself jokes about a shared cookie. Neither response is a capability result.

The Qwen reference implementation reported PyTorch fallbacks for optional convolution and gated-delta kernels on Windows. No admin packages or compiler were installed. Tiny three-token completions and a single warmed/cold trajectory are unsuitable for speed comparisons; an identical control itself had different measured decode rates. Loading time is recorded separately from first-token latency.

An earlier exploratory Qwen smoke emitted `im_end` then continued to its generation-config EOS. The worker was corrected to stop on either the tokenizer EOS or configured EOS, and the four final cases above used that corrected code. Earlier raw records remain local. The final worker was stopped after testing, releasing its GPU allocation.

## Still unverified

We have **not reproduced the published anti-loop benchmark**, tested long Qwen contexts, measured a quality gain, or discovered an internal evil persona. The author's evaluation settings differ from this short greedy chat path; reproducing them needs a separate frozen protocol with enough output budget to observe looping and collateral accuracy. The [author's method](https://rikiyat.github.io/blog/weight-space-programming/) and [released checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) are references, not local results.
