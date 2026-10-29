# Independent matched-budget comparison

## Status and freeze chronology

One local CPU evaluation ran once. No tuning or method/seed selection used validation or test outcomes. The new family uses seeds 2026103001/2026103002 and is separate from `experiments_strong_baseline/`; prior protocols and results were not changed.

- Protocol SHA-256: `F6EB0B4DAC361CD214239574A33E5CA311327AD81CB96241669C45D61C6C9C57`
- Runner SHA-256: `65441499AC17BBD724FC49610C1EF4D2FB5BF8DB232BA16BD1531026DC0C7E30`
- Canonical dataset-manifest SHA-256: `DB95F5F18BA15577321AED9CB381033B4FD23A6C7AE485C9D37AF576C790BB5C`
- Output file-byte SHA-256: `5E108733119FD5480BE09497B01E05099587E4130E93A475694D55600F302BA0`

The protocol and runner hashes were recorded before new synthetic worlds were generated. The generated data/split manifest was then frozen and all three hashes were checked before the single evaluation. `run_experiment.py` refuses evaluation if the frozen hashes mismatch and refuses to overwrite the existing results file.

## Design and compute limit

Five independent replicate worlds each contain 18 training, 6 validation and 6 test scenario groups. All three methods share those splits, forecast/realized scenarios and common random numbers: decision-focused VQC, predictive-only VQC, and logistic regression. Forecast-only selection precedes realized scoring; exact reference and independent plan validation occur post-selection.

Each method has the same maximum of 120 full-batch updates and a 0.25-second process-CPU ceiling per replicate, checked between updates. Because an update can overrun the ceiling and per-update costs differ, this is an approximate common upper bound rather than equal compute delivered. Observed per-replicate updates were 26–27 for decision VQC, 27–35 for predictive VQC, and 120 for logistic regression. The VQC methods each reached the 0.25-second cap; logistic fit time was about 0.016–0.031 seconds. These differences are material and limit interpretation. Test evaluation count is identical: 30 scenarios per method (six per replicate).

## Held-out results

| Method | Mean validator objective | Mean regret vs exact | Valid plans | Mean route Brier |
|---|---:|---:|---:|---:|
| Decision-focused VQC | 13.14671 | 0.19357 | 100% | 0.07258 |
| Predictive-only VQC | 13.13396 | 0.18082 | 100% | 0.06844 |
| Logistic regression | 13.13498 | 0.18184 | 100% | 0.07054 |

Paired test objective difference versus logistic, averaged across replicate means: decision VQC +0.01172 (SD 0.05122; range -0.04571 to +0.08754); predictive VQC -0.00103 (SD 0.03193; range -0.04571 to +0.03376). The outputs show no demonstrated decision-quality advantage. There were no observed lateness-exposure differences (0 for all methods).

All data are synthetic and tiny; there are only five replicate worlds and six held-out groups per replicate. Route rows are clustered within scenarios. The CPU ceiling produces unequal delivered compute; objectives also have inherently different per-update costs. These results establish neither quantum advantage nor calibration, operational performance, generalization, or superiority.

## Checks and execution

- `python -m py_compile experiments_matched_budget/run_experiment.py` — passed before freeze.
- `python -m pytest experiments_matched_budget/tests -q` — 3 passed before held-out evaluation.
- `python experiments_matched_budget/run_experiment.py --freeze` — generated the 150-group manifest; no model fit or score was performed by this command.
- `python experiments_matched_budget/run_experiment.py` — single locked local evaluation; 30 held-out scenarios per method, all 90 selected plans passed independent validation; total reported wall time 2.968 seconds, no external calls. Runner records CPython 3.12.14 on Windows 11.
- No installs, network/cloud/QPU calls, commits, or changes outside `experiments_matched_budget/`.

## UTC date correction
The frozen lock says freeze_date_utc 2026-10-29; that date is inconsistent with observed UTC. Filesystem last-write UTC timestamps are 2026-10-28 19:37:36.732 for pre_generation_freeze.json, 19:37:50.611 for frozen_lock.json, and 19:38:04.781 for results.json. The original lock is preserved. Filesystem timestamps support the reported sequence but are mutable and are not independent tamper-proof evidence.
