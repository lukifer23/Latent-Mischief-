# Mischief-BFA0A0

Recipe: `LM-bfa0a0d995cca2f4` · unreviewed

villain · layer 13 · residual dose 0.15

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
  "family": "contrast-direction",
  "operations": [
    {
      "layer": 13,
      "kind": "residual-vector",
      "vector_sha256": "d9cb5d3a56b34de8c95fafca7ee8fa75344f39739f79f016deb120c48a38a974",
      "amplitude": 0.4656398892402649,
      "dtype": "float32 reference; cast delta to model dtype"
    }
  ],
  "positions": "last position of each forward: final prompt token and incremental decode tokens",
  "hook_location": "decoder output after residual addition",
  "hook_function_ast_sha256": "fdba1f134f2163525f7c88c028aaea0eaf8edb0eec653c2beadac61add58159f"
}
```

## Differences from matched default

Each row is a separate run/cohort; numbers are descriptive, not personality scores.

| Run / cohort | n | Changed | Mean tokens: default → edit | EOS: default → edit | Exact checks: default → edit |
|---|---:|---:|---|---|---|
| `20261106T182140177847Z-LFM-002-persona-discovery` / legacy-exploration | 3 | 3 | 96.00 → 96.00 | 0 → 0 | not measured |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T182140177847Z-LFM-002-persona-discovery",
    "intended_character": "villain",
    "selection": "See frozen source configuration and extraction artifacts.",
    "vector_artifact": {
      "run": "20261106T182140177847Z-LFM-002-persona-discovery",
      "file": "directions.pt",
      "key": "villain:13",
      "file_sha256_at_catalog_build": "df2bf5265a0377c1ab1786e4c93e4661759048f8875011b046e3b7fde0fb03bf"
    },
    "dose": 0.15
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
