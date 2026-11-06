# Mischief-D5F6BA

Recipe: `LM-d5f6ba47aa6a8aa7` · unreviewed

random-L04@0.15

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
      "layer": 4,
      "kind": "offset",
      "channels": [
        4893,
        2352,
        5540,
        18,
        5965,
        6525,
        1555,
        7221,
        625,
        7558,
        6603,
        4202,
        3168,
        4075,
        1886,
        3959
      ],
      "offsets": [
        -0.11218062788248062,
        0.46262457966804504,
        0.46655145287513733,
        -0.19155682623386383,
        0.31472158432006836,
        -0.0697653591632843,
        0.22327697277069092,
        -0.03505300357937813,
        -0.13761404156684875,
        -0.09508868306875229,
        0.30924496054649353,
        0.15027424693107605,
        0.07321958243846893,
        0.2657622992992401,
        0.04685327038168907,
        -0.08720012754201889
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
| `20261106T195602981114Z-CHAR-001-sparse-characters` / character-exploration | 3 | 3 | 83.33 → 69.33 | 2 → 3 | not measured |
| `20261106T195602981114Z-CHAR-001-sparse-characters` / exact-check | 3 | 1 | 7.67 → 10.33 | 3 → 3 | 2 → 2 / 3 |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T195602981114Z-CHAR-001-sparse-characters",
    "intended_character": null,
    "selection": "See frozen source configuration and extraction artifacts.",
    "profile_id": "random-L04@0.15",
    "nominal_target": 0.15,
    "amplitude_multiplier": 1
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
