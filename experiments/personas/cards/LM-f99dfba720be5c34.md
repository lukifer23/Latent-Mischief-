# Mischief-F99DFB

Recipe: `LM-f99dfba720be5c34` · unreviewed

L20 N2253 · gain 0.0

No observed persona nickname has been assigned.

## Exact intervention

```json
{
  "schema": 1,
  "model": {
    "model_id": "LiquidAI/LFM3-1.7B",
    "revision": "0dc4268b561499c648c7ba7c9466bf3d8d01517a",
    "dtype": "bfloat16",
    "attention": "eager"
  },
  "family": "channel-gain",
  "operations": [
    {
      "layer": 20,
      "kind": "gain",
      "channels": [
        2253
      ],
      "gain": 0.0
    }
  ],
  "positions": "all prompt positions and every decode position",
  "hook_location": "gated MLP input before w2 and post_ffn_norm",
  "hook_function_ast_sha256": "5859ec6944a28515d8ba3e971a95e60d3bb6afd987ff966e7881db508f4fd79c"
}
```

## Differences from matched default

Each row is a separate run/cohort; numbers are descriptive, not personality scores.

| Run / cohort | n | Changed | Mean tokens: default → edit | EOS: default → edit | Exact checks: default → edit |
|---|---:|---:|---|---|---|
| `20261106T184602567236Z-MAP-001-channel-exploration` / legacy-exploration | 3 | 1 | 86.67 → 101.00 | 2 → 2 | not measured |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T184602567236Z-MAP-001-channel-exploration",
    "intended_character": null,
    "selection": "See frozen source configuration and extraction artifacts."
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
