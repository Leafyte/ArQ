# Independent saved-plan validation

## Scope

The saved results and frozen manifest permit reconstruction of every selected test plan without model fitting or selection. The results retain each test `scenario_id`, selected route IDs, and reported validator objective. The frozen protocol/runner seeds reconstruct the paired scenarios; the regenerated split and scenario hashes match the locked manifest. This check reused the recorded route-ID choices and did not rank plans, choose alternatives, fit models, or calculate new predictions.

## Procedure and result

Ran `.venv\Scripts\python.exe experiments_matched_budget\validate_saved_results.py` on 2026-10-28 UTC. The read-only script:

1. Captured byte hashes for protocol, frozen lock, manifest, runner, and result files.
2. Reconstructed the scenario inputs using the fixed seed/generator in the frozen runner and required exact equality with the locked manifest.
3. For each stored method/scenario pair, materialized only the recorded route IDs from the regenerated realized input, then called `backend.validator.validate_plan` to recompute coverage, route feasibility, timing, travel, SLA exposure, and objective from raw scenario inputs.
4. Compared recomputed objectives with the reported saved objectives and compared all tested scenario IDs with the locked test split.
5. Re-read the five file hashes and required exact equality with the initial values.

Outcome: **90/90** recorded plans validated (30 scenarios × 3 methods); maximum absolute difference between independently recomputed and saved validator objectives was **0.0**. Each method covered the same 30 frozen test scenario IDs. The before/after byte hashes were identical:

| File | SHA-256 (before = after) |
|---|---|
| protocol.json | `F6EB0B4DAC361CD214239574A33E5CA311327AD81CB96241669C45D61C6C9C57` |
| frozen_lock.json | `9D0EA679B631EDE077EE8CB8F1D64FF4351D56B952983E3F527CF1FA0A307FB9` |
| dataset_manifest.json | `0AD8E4099411B87C33F9742A674C132BB3A3BB624AAE2A856C7613B1DA31D952` |
| run_experiment.py | `65441499AC17BBD724FC49610C1EF4D2FB5BF8DB232BA16BD1531026DC0C7E30` |
| results.json | `5E108733119FD5480BE09497B01E05099587E4130E93A475694D55600F302BA0` |

## Limits and date correction

The realized matrices were reconstructed deterministically from the locked seeds and the same frozen generator implementation; this was not an independent reimplementation of scenario generation. The validator independently recomputes plan facts from scenario inputs, while model selection was not repeated. The check confirms internal reconstruction and plan validity, not the scientific validity of the generator or method comparison.

The frozen lock's `freeze_date_utc` says `2026-10-29`, but the clock records this work at `2026-10-28 UTC`; the earlier value came from the local Asia/Calcutta date being placed in a UTC field. [DATE-CORRECTION.md](DATE-CORRECTION.md) records this interpretation. No frozen or result file was changed.
