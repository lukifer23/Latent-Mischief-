# Mischief-AFF646

Recipe: `LM-aff64636d3dfef43` · unreviewed

villain-L10-reversed@0.15

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
        -0.24866563081741333,
        -0.12128620594739914,
        0.07949159294366837,
        0.043824702501297,
        -0.0971684604883194,
        0.09627483785152435,
        -0.0576290562748909,
        -0.048000920563936234,
        -0.04593784734606743,
        0.06350727379322052,
        -0.04677234962582588,
        0.0503397136926651,
        -0.04452916234731674,
        0.026847321540117264,
        0.05738934874534607,
        0.08041860908269882
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
| `20261106T195602981114Z-CHAR-001-sparse-characters` / exact-check | 3 | 2 | 7.67 → 9.00 | 3 → 3 | 2 → 2 / 3 |
| `20261106T195602981114Z-CHAR-001-sparse-characters` / character-exploration | 3 | 3 | 83.33 → 61.00 | 2 → 3 | not measured |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T195602981114Z-CHAR-001-sparse-characters",
    "intended_character": "villain",
    "selection": "See frozen source configuration and extraction artifacts.",
    "profile_id": "villain-L10-reversed@0.15",
    "nominal_target": 0.15,
    "amplitude_multiplier": 1
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
