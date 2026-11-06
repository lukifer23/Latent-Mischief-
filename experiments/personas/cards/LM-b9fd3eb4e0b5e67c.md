# Mischief-B9FD3E

Recipe: `LM-b9fd3eb4e0b5e67c` · unreviewed

hamster · layer 19 · residual dose 0.7

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
      "layer": 19,
      "kind": "residual-vector",
      "vector_sha256": "50f9d96776249ecdb275c5fdd62c7787780e75772be41c9b5f4fe533bf2f214e",
      "amplitude": 5.180704855918884,
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
| `20261106T182953749085Z-LFM-002-persona-discovery` / legacy-exploration | 3 | 3 | 192.00 → 192.00 | 0 → 0 | not measured |

## Methodology and evidence

Method derivation, intended character, source runs and local vector references:

```json
[
  {
    "run": "20261106T182953749085Z-LFM-002-persona-discovery",
    "intended_character": "hamster",
    "selection": "See frozen source configuration and extraction artifacts.",
    "vector_artifact": {
      "run": "20261106T182953749085Z-LFM-002-persona-discovery",
      "file": "directions.pt",
      "key": "hamster:19",
      "file_sha256_at_catalog_build": "eeb5ecefd7a505a4edee147e7810c04534702f4fc08d38445f36ebac3b12702a"
    },
    "dose": 0.7
  }
]
```

Full source configurations, hashes, prompts/settings and per-run provenance are keyed by run in [registry.json](../registry.json). Raw responses remain local and are browsable in the workbench. A missing nickname or no effect is a retained outcome.
