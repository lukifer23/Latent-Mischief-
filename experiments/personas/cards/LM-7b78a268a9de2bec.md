# Mischief-7B78A2

Recipe: `LM-7b78a268a9de2bec` · unreviewed

joint-villain@1

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
  "family": "joint-profile",
  "operations": [
    {
      "layer": 4,
      "kind": "offset",
      "channels": [
        6106,
        6517,
        3243,
        2570,
        6303,
        1402,
        5331,
        7309,
        3876,
        6640,
        4952,
        1480,
        145,
        6308,
        6871,
        2622
      ],
      "offsets": [
        -0.45029217004776,
        -0.5468134880065918,
        -0.27052509784698486,
        -0.1391282081604004,
        0.08729514479637146,
        0.148538738489151,
        0.08877789229154587,
        -0.05109928175806999,
        0.10070853680372238,
        0.16582359373569489,
        -0.14497235417366028,
        0.07832342386245728,
        0.08205173909664154,
        -0.09450578689575195,
        -0.09690221399068832,
        0.0827365517616272
      ]
    },
    {
      "layer": 10,
      "kind": "offset",
      "channels": [
        3775,
        4993,
        7232,
        2616,
        6625,
        2434,
        7057,
        6361,
        3549,
        1683,
        1305,
        6129,
        1755,
        1831,
        937,
        3576
      ],
      "offsets": [
        0.2526648938655853,
        0.12323684245347977,
        -0.08077004551887512,
        -0.04452952742576599,
        0.09873121231794357,
        -0.09782321006059647,
        0.05855589732527733,
        0.048772912472486496,
        0.04667666181921959,
        -0.06452865898609161,
        0.0475245825946331,
        -0.05114931985735893,
        0.04524531960487366,
        -0.02727910503745079,
        -0.05831233784556389,
        -0.08171197026968002
      ]
    },
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
        -0.28095895051956177,
        0.2512120306491852,
        -0.05664070323109627,
        -0.07855239510536194,
        -0.0649077519774437,
        -0.11922168731689453,
        0.04198257252573967,
        -0.03154676780104637,
        0.04560292139649391,
        -0.039863552898168564,
        0.07183939963579178,
        0.029304709285497665,
        -0.039416294544935226,
        -0.03587053343653679,
        0.030631549656391144,
        -0.03509660065174103
      ]
    },
    {
      "layer": 22,
      "kind": "offset",
      "channels": [
        1165,
        7202,
        1833,
        5189,
        6328,
        6841,
        2737,
        5672,
        73,
        5225,
        7237,
        5604,
        1767,
        7513,
        667,
        4665
      ],
      "offsets": [
        0.3181322515010834,
        0.5427161455154419,
        0.27184200286865234,
        -0.14058038592338562,
        -0.7972621321678162,
        -0.15498492121696472,
        0.12378247082233429,
        -0.1379697322845459,
        -0.12813405692577362,
        0.1234344094991684,
        -0.07272230833768845,
        0.11055577546358109,
        -0.12600280344486237,
        0.08441047370433807,
        -0.09414314478635788,
        -0.1257675737142563
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
| `20261106T201009924785Z-CHAR-002-joint-characters` / fresh-character | 3 | 3 | 60.67 → 88.67 | 3 → 3 | not measured |
| `20261106T201009924785Z-CHAR-002-joint-characters` / discovery-anchor | 1 | 1 | 65.00 → 65.00 | 1 → 1 | not measured |
| `20261106T201009924785Z-CHAR-002-joint-characters` / fresh-check | 3 | 2 | 8.67 → 10.33 | 3 → 3 | 2 → 2 / 3 |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T201009924785Z-CHAR-002-joint-characters",
    "intended_character": "villain",
    "selection": "See frozen source configuration and extraction artifacts.",
    "profile_id": "joint-villain@1",
    "nominal_target": null,
    "amplitude_multiplier": 1
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
