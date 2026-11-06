# MAP-006: fresh compression/completion comparisons

Status: predeclared; no results yet. Target candidates were selected from inspected MAP-005 outcomes. Fresh prompts below were written before this run and have not been used to select candidates.

Use the frozen CAL-001 numeric evidence to recover positive offsets for L04N5848 and L20N0472 at 5% and 20%. For each layer, choose two other channels randomly (seed 20261108) from those with both strengths uncapped, actual calibration means within ±25% of target, and mean next-token KL ≤0.5. Controls match sparse size and approximate achieved local strength; they are not matched on semantic selectivity. Save the selection before generating.

Six fresh behavioral prompts, four fresh exact checks, and two labeled discovery anchors. Same neutral system as MAP-005, greedy BF16, 192-token cap. Twelve intervention conditions × twelve prompts + twelve baselines + two hook checks = **158 generations**. Preserve every outcome. Read fresh and anchor cohorts separately.

Primary compression criterion (L04N5848 positive, 20%): at least four of six fresh behavioral replies shorter; their mean tokens reduced at least 10%; reduction exceeds both same-layer random controls; no fresh exact-check losses; no additional fresh behavioral EOS failures. Primary completion criterion (L20N0472 positive, 20%): at least one fresh behavioral baseline cap repaired, no new fresh behavioral cap, and no fresh exact-check loss. If baseline completes all six, completion replication is inconclusive. Show control outcomes even if a criterion passes. Five-percent doses explore sensitivity, not a substitute success criterion.

Capture actual prefill relative changes for each reply, source/config hashes, schedule, tokens/text, timing, EOS, exact checks and repetition. These checks are narrow and single-trajectory. Successful replication would support these limited prompt-level effects; it would not establish a stable personality or general capability gain.
