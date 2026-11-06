# Mischief-67F66A

Recipe: `LM-67f66a76daec1f1c` · unreviewed

villain-L15@0.05

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
        5289,
        4493,
        7070,
        3175,
        264,
        6074,
        1104,
        1783,
        7218,
        3401,
        6451,
        5314,
        6605,
        5776,
        3574,
        3756
      ],
      "offsets": [
        -0.09238831698894501,
        0.08260657638311386,
        -0.018625279888510704,
        -0.025830548256635666,
        -0.02134375087916851,
        -0.03920391574501991,
        0.013805216178297997,
        -0.01037358958274126,
        0.014995703473687172,
        -0.01310841552913189,
        0.02362309955060482,
        0.009636328555643559,
        -0.012961341999471188,
        -0.011795382015407085,
        0.010072636418044567,
        -0.011540887877345085
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
| `20261106T195602981114Z-CHAR-001-sparse-characters` / character-exploration | 3 | 1 | 83.33 → 64.00 | 2 → 3 | not measured |
| `20261106T195602981114Z-CHAR-001-sparse-characters` / exact-check | 3 | 1 | 7.67 → 7.67 | 3 → 3 | 2 → 2 / 3 |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T195602981114Z-CHAR-001-sparse-characters",
    "intended_character": "villain",
    "selection": "See frozen source configuration and extraction artifacts.",
    "profile_id": "villain-L15@0.05",
    "nominal_target": 0.05,
    "amplitude_multiplier": 1
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
