# Dependency advisory follow-up

GitHub reported three alerts against the initial lock on 2026-11-06:

| Package / advisory | Reported patched floor | Action |
|---|---|---|
| setuptools [GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w) | 84.1.0 | Pin a lower bound of 84.1; lock 84.2.1 |
| setuptools [GHSA-h35f-9h28-mq5c](https://github.com/advisories/GHSA-h35f-9h28-mq5c) | 83.0.0 | Covered by the same update |
| PyTorch [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) | 2.13.0 | Require >=2.13; lock 2.14.1+cu130 |

Moved the official PyTorch wheel index from cu128 to cu130 because the checked cu128 Windows/Python 3.11 index did not offer the required patched versions. The existing driver worked with the new prebuilt runtime. No driver or system CUDA toolkit installation was performed.

Validation: 10 API/persistence tests passed; a new GPU baseline and neutral-hook check passed; all six baseline/check token sequences matched the prior runtime in this small suite. The 41-generation MAP-001 pilot also completed on the patched runtime. These checks are compatibility evidence, not a general proof of numerical equivalence.

Earlier experiments retain their original package versions, lock hashes and source snapshots. Do not silently relabel them as having run on the new runtime. GitHub may take time to rescan the pushed lock; patched bounds and resolved versions are the local evidence of the update.
