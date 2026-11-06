# Mischief-DA41AD

Recipe: `LM-da41add82c307c7b` · unreviewed

hamster-L15@0.05

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
  "family": "sparse-profile",
  "operations": [
    {
      "layer": 15,
      "kind": "offset",
      "channels": [
        6480,
        5552,
        3175,
        1783,
        4493,
        2379,
        4170,
        1594,
        4551,
        1435,
        1373,
        3735,
        6980,
        1325,
        4942,
        3703
      ],
      "offsets": [
        0.07490063458681107,
        -0.03158226236701012,
        -0.045734405517578125,
        -0.022478369995951653,
        0.060417044907808304,
        -0.019643330946564674,
        0.021261243149638176,
        0.017591377720236778,
        -0.02840241603553295,
        -0.020362474024295807,
        0.022432276979088783,
        0.020076945424079895,
        0.020935900509357452,
        0.02184750884771347,
        -0.01751696690917015,
        0.017469679936766624
      ]
    }
  ],
  "positions": "all prompt positions and every decode position",
  "hook_location": "gated MLP input before w2 and post_ffn_norm",
  "hook_function_ast_sha256": "5472b84a3c7dc122c363fe2c2c9253335e6eb6eb311726535c1898bc5c919c61"
}
```

## Differences from matched default

Each row is a separate run/cohort; numbers are descriptive, not personality scores.

| Run / cohort | n | Changed | Mean tokens: default → edit | EOS: default → edit | Exact checks: default → edit |
|---|---:|---:|---|---|---|
| `20261106T195602981114Z-CHAR-001-sparse-characters` / exact-check | 3 | 0 | 7.67 → 7.67 | 3 → 3 | 2 → 2 / 3 |
| `20261106T195602981114Z-CHAR-001-sparse-characters` / character-exploration | 3 | 0 | 83.33 → 83.33 | 2 → 2 | not measured |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T195602981114Z-CHAR-001-sparse-characters",
    "intended_character": "hamster",
    "selection": "See frozen source configuration and extraction artifacts.",
    "profile_id": "hamster-L15@0.05",
    "nominal_target": 0.05,
    "amplitude_multiplier": 1
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
