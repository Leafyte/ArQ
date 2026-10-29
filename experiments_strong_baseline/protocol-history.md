# Protocol history

- **2026-10-28:** Protocol v1 and the independent 150-group scenario/split
  manifest were frozen before any model evaluation. V1 protocol SHA-256 was
  `55E88AFBCA8CDC4A4146022965C868FFD179EB2083336AB44D2733DC23A16B7B`; its
  dataset-manifest SHA-256 is
  `FAC4731D26AF3D359E6822E4C5DAAB422D2101AABCFC234F8FF4F8903720120A`.
  V1 specified scikit-learn `RandomForestClassifier` 1.8.0.
- The pinned installation attempt failed because the sandbox could not resolve
  PyPI DNS. A coordinator retry with pip and uv also stalled on the wheel
  download and was stopped. No V1 model fit or held-out evaluation ran.
- **Before any held-out evaluation**, V2 changed only the nonlinear comparator
  implementation: a deterministic, dependency-free, grouped-bootstrap CART
  ensemble with 64 trees, maximum depth 3, minimum leaf size 2 and one feature
  sampled per split. Whole training scenario groups are bootstrap units. This
  small comparator is not asserted to be equivalent to scikit-learn or a
  proven strong production baseline.
- V2 preserves the 5×(18 train, 6 validation, 6 test) scenario split, generator,
  scenario seeds, metrics, VQC/logistic settings and exact/validator procedure.
  V1 protocol/lock files are preserved verbatim; V2 protocol and split hashes
  are pinned by `frozen_lock.json`. No model or seed choice used validation or
  test outcomes.
- The first structural test exposed that a generic prospective route object
  still carried a `train_label` attribute into inference, even when that field
  was `None`. Before test-set evaluation, inference was changed to use a
  separate forecast-only route/scenario type with no label, realized matrix,
  or realized plan-cost fields. The structural regression checks passed.

The V2 structural suite passed **5 tests** before model evaluation. Then the
single frozen evaluation ran once with
`.venv\\Scripts\\python.exe experiments_strong_baseline\\run_experiment.py`.
It evaluated 30 held-out scenarios across five replicates; all four methods
had 100% validator-valid plans. Mean exact regret was 0.15796 for decision VQC,
0.14065 for predictive VQC, 0.14084 for logistic regression, and 0.22329 for
the grouped-bootstrap CART forest. Mean route Brier was 0.09134, 0.09027,
0.09146, and 0.04942 respectively. Decision-VQC minus logistic paired test
objective averaged +0.01712 per replicate (sample SD 0.02656; range 0 to
0.06332); predictive VQC averaged -0.00019 (SD 0.04794; range -0.07162 to
0.06332); CART averaged +0.08245 (SD 0.19544; range -0.01545 to +0.43187).
The saved artifact records every per-scenario result and per-replicate fit
work/timing. Total runner time was 14.653 seconds.

The CART had the lowest route Brier but the highest mean decision objective
and regret; this is not evidence of calibrated risk or better dispatch. The
two VQC and logistic methods had nearly identical mean objective. The test set
is only 30 synthetic scenarios, route rows are clustered, and each replicate
has six test groups. The optimizer work differs (120 VQC updates versus 500
logistic updates; the CART uses 64 depth-3 trees), so this is not a compute-
matched superiority comparison. The code refuses to overwrite `results.json`;
do not tune from these outcomes or rerun into this protocol/artifact.

Saved output SHA-256:
`D2B444D2CBBABBDE83F15D4EB53A75CCA95B214DA04C1B9A5B3EB0A73FAB321A`.
