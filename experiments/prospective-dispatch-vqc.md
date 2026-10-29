# Prospective forecast-to-dispatch simulation

**Protocol:** `experiments/protocol/prospective-dispatch-vqc-v1.json`  
**Protocol SHA-256:** `60CDC7A0B6E676B2BE9FFA7928BC0BE505BBF9870B9D65CF48D833F935AD86F8`  
**Execution command:** `python -m experiments.run_prospective_dispatch`  
**Raw result:** `experiments/artifacts/prospective-dispatch-results.json`

The protocol and source fixture hash were written and checked before this evaluation. The runner rejects any change in either hash. This is a small synthetic simulation; the six test scenarios were evaluated once and were not used to modify the model.

## Data boundary

The fixed source is `fixtures/dispatch_tiny.json` (file-byte SHA-256 `39C2AEC74DA0C7A71B6E3BEC32BC91D16FC3DD283F7F73D056F0684A255F9733`). A seeded generator creates 24 grouped scenarios. It samples a latent realized-traffic multiplier from `Uniform[0.7,2.5]`, then separately samples a noisy forecast multiplier as `clip(realized + Normal(0,0.30),0.5,3.0)`. Independent per-edge jitter is applied to nominal travel for forecast and realized matrices. Scenario groups are shuffled with a separate seed into 12 train, 6 validation, and 6 test cases.

The inference interface is `ForecastDecisionInput`; it contains only the scenario's forecast matrix, forecast-derived route rows, and exact-cover plan route IDs. It has no realized matrix or held-out labels. Non-training route labels are `None` and stripped from model-visible plan input. The VQC and baselines select a complete plan before evaluation opens paired realized travel. The realized plan is then independently checked with `backend.validator.validate_plan`; the exact reference is solved on realized inputs and independently checked as well. All models use the same forecast and realized matrices for each scenario (common random numbers).

Training and validation/test scenarios remain grouped by whole scenario. Only train scenarios contribute realized route-event labels and realized plan objectives to fitting. Validation and test values are never used for parameter, feature, calibration, or scenario selection. Scenario hashes, forecast hashes, realized hashes, group counts, route candidates, training examples, selected routes, raw plans, and validator outputs are retained in the JSON artifact.

## Decision and circuit

Feature `x0` is forecast candidate-route travel duration divided by 30 and clipped to `[0,1]`. Feature `x1` is nominal stop-window pressure from the source scenario. The two-qubit circuit encodes them with `RY(pi*x0)` and `RY(pi*x1)`, entangles with `CNOT(0,1)`, applies trainable `RY(theta0)` and `RY(theta1)`, then `CNOT(1,0)`. Its output is `p = sigmoid(-2 <Z0>)`.

Each candidate route receives estimated cost `forecast travel duration + p * sum(contract penalty rates) * 1 minute`. The optimizer uses a soft minimum over the four exact-cover forecast-feasible plans in each training scenario. Plan loss is weighted by the paired realized objective independently recomputed by the validator. At inference, the minimum estimated-cost forecast plan is selected; its exact ordered routes are materialized on realized travel only afterward for validation and reporting.

The objective units are travel duration plus contract penalty-rate times lateness minutes, as defined by the tiny domain fixture. These are proxy units, not calibrated currency. The one-minute expected-exposure assumption is explicit and uncalibrated.

## Executed results

The decision-loss parameter-shift/chain-rule gradient was checked on training scenarios `prospective-00` and `prospective-02`. Analytic values were `[0.8809412569,-0.4045884433]`; centered finite differences at `h=1e-5` were `[0.8809412567,-0.4045884435]`; maximum absolute error was `2.04e-10`.

Test results (`n=6` paired scenarios):

| Measure | Decision-focused VQC | Predictive-only VQC | Logistic |
|---|---:|---:|---:|
| Feasibility / full coverage | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Mean validator objective | 15.3962 | 15.3962 | 15.3962 |
| Mean validator travel duration | 15.3962 | 15.3962 | 15.3962 |
| Mean validator lateness exposure | 0.0000 | 0.0000 | 0.0000 |
| Mean estimated exposure | 1.0607 | 1.0981 | 0.6729 |
| Mean regret vs exact realized reference | 0.1567 | 0.1567 | 0.1567 |

All methods selected `V1: B -> C`, `V2: A` in all six test cases. The test cases produced no realized lateness for those selections, while model estimates were positive. Two forecast-selected plans had positive regret from travel duration versus the exact realized-scenario reference. The results do not show model-specific decision improvement or demonstrate a useful probability forecast; the small sample and absence of selected-plan SLA events make calibration impossible here.

## Evidence limits

- Scenarios and outcomes are synthetic, from only three orders and two vehicles.
- The frozen runner verifies protocol and fixture hashes, but the recorded result does not include a hash of the implementation source tree or Git commit. The protocol and data hashes therefore do not uniquely pin the code used to generate the output; reproduce against the same reviewed source files and recorded Python runtime.
- After the recorded V1 evaluation, the training-label implementation was corrected to derive SLA events from actual stop lateness, independent of financial penalty. The regression `test_prospective_sla_label_uses_lateness_even_when_penalty_is_zero` now passes. The frozen V1 fixture uses strictly positive penalty rates (4, 2, and 1), so labels are logically equivalent for that fixture, but the held-out result was not rerun after this code change; no post-change reproduction claim is made. The frozen protocol and recorded output remain unchanged.
- The forecast generator is a designed simulation distribution, not a fitted traffic forecast model.
- Only six held-out scenarios were executed; there are no uncertainty intervals or generalization claim.
- Estimated exposure uses a fixed one-minute severity and was not calibrated.
- The tiny fixture objective is not a dollar-valued financial model.
- Execution used an exact local two-qubit statevector with zero shots. No QpiAI SDK, cloud API, Opt service, or QPU was used.
- No quantum advantage or operational readiness is claimed.
