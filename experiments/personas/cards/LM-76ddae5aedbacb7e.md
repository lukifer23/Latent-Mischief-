# Mischief-76DDAE

Recipe: `LM-76ddae5aedbacb7e` · unreviewed

L20N1897 · +0.0657981 offset

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
      "layer": 20,
      "kind": "offset",
      "channels": [
        1897
      ],
      "offsets": [
        0.06579813217172159
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
| `20261106T200050651270Z-MAP-006-offset-replication` / fresh-behavior | 6 | 4 | 171.33 → 171.50 | 3 → 3 | not measured |
| `20261106T200050651270Z-MAP-006-offset-replication` / fresh-check | 4 | 1 | 10.25 → 10.25 | 4 → 4 | 1 → 1 / 4 |
| `20261106T200050651270Z-MAP-006-offset-replication` / discovery-anchor | 2 | 2 | 135.50 → 89.00 | 1 → 2 | not measured |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T200050651270Z-MAP-006-offset-replication",
    "intended_character": null,
    "selection": "See frozen source configuration and extraction artifacts.",
    "calibration_target": 0.05,
    "native_rms_units": 6.029052734375
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
