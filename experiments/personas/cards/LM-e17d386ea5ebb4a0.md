# Mischief-E17D38

Recipe: `LM-e17d386ea5ebb4a0` · unreviewed

L10N4117 · +0.389729 offset

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
  "family": "channel-offset",
  "operations": [
    {
      "layer": 10,
      "kind": "offset",
      "channels": [
        4117
      ],
      "offsets": [
        0.38972894819835346
      ]
    }
  ],
  "positions": "all prompt positions and every decode position",
  "hook_location": "gated MLP input before w2 and post_ffn_norm",
  "hook_function_ast_sha256": "75b8f31ed67d9b5779a006948fb88f19901b49367a730f59f314d47d9e855386"
}
```

## Differences from matched default

Each row is a separate run/cohort; numbers are descriptive, not personality scores.

| Run / cohort | n | Changed | Mean tokens: default → edit | EOS: default → edit | Exact checks: default → edit |
|---|---:|---:|---|---|---|
| `20261106T193431789097Z-MAP-005-resumed-offsets` / behavior-exploration | 10 | 10 | 114.50 → 122.80 | 9 → 8 | not measured |
| `20261106T193431789097Z-MAP-005-resumed-offsets` / exact-check | 6 | 0 | 8.33 → 8.33 | 6 → 6 | 4 → 4 / 6 |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T193431789097Z-MAP-005-resumed-offsets",
    "intended_character": null,
    "selection": "See frozen source configuration and extraction artifacts.",
    "calibration_target": 0.2,
    "native_rms_units": 15.362548828125
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
