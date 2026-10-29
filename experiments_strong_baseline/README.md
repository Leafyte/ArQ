# Independent matched classical baseline

This is a new synthetic experiment family; it does not overwrite the earlier
prospective result. The original scikit-learn protocol was frozen but could not
be executed because package downloads were unavailable; its protocol and lock
remain preserved as `protocol-v1-scikit-blocked.json` and
`frozen-lock-v1-scikit-blocked.json`. Before any model evaluation, the method
was changed to a fixed dependency-free grouped-bootstrap CART ensemble. The
updated protocol and unchanged scenario/split hashes are frozen in
[`protocol.json`](protocol.json), [`dataset_manifest.json`](dataset_manifest.json),
and [`frozen_lock.json`](frozen_lock.json), with chronology in
[`protocol-history.md`](protocol-history.md). The family contains five independent
replicate worlds, each with 18 train, 6 validation and 6 test scenario groups.

Decision-focused VQC, predictive-only VQC, logistic regression and a nonlinear
grouped-bootstrap CART ensemble use the same train groups and forecast-only
route features. The same held-out forecast/realized pairs are used by every
method. The ensemble is fixed at 64 trees, depth 3, minimum leaf size 2, with
one of two features sampled at each split and whole training scenario groups
sampled with replacement. It is a small comparison model, not a validated
production baseline. VQC uses 120 gradient updates per method; logistic uses
500 updates. These are not equal computational budgets; fit wall time and
work counts are reported per replicate. Model selection is completed before
the experiment opens any realized validation/test matrix. The exact realized
solver is a post-selection reference, not a model input. Selected plans are
recomputed and checked by the independent domain validator.

## Environment and commands

Use the workspace `.venv` (CPython 3.12.14). The comparator implementation is
in-repository Python and requires no package installation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_strong_baseline.py -q
.\.venv\Scripts\python.exe experiments_strong_baseline\run_experiment.py
```

The runner verifies protocol, manifest, and source hashes before evaluation and refuses
to overwrite an existing `results.json`. Per-scenario paired differences and five replicate means
are saved, along with Brier score, independent validation, exact references,
training work, and timing. The runner must not be used to tune models, seeds,
features, or thresholds from validation/test outcomes.

The one preregistered V2 run passed all 30 test plans through the independent
validator. Mean regret was 0.15796 (decision VQC), 0.14065 (predictive VQC),
0.14084 (logistic), and 0.22329 (grouped-bootstrap CART). The CART achieved
the lowest route Brier (0.04942) but had the highest mean objective. Results,
paired per-scenario deltas, replicate spread, training work and timing are in
[`results.json`](results.json); no method-specific advantage was demonstrated.

All data are artificial and tiny. Route-level Brier values are grouped within
scenarios and should not be treated as independent observations. Five test
replicates are not enough to support operational, calibration, advantage, or
deployment claims.
