# Candidate-route decision-focused VQC

**Status:** exploratory local conditional-dispatch demonstration. The test scenarios were inspected during integration; treat their metrics as diagnostic, not as a blind evaluation.  
**Command:** `python -m experiments.run_dispatch_vqc`  
**Raw results:** `experiments/artifacts/dispatch-vqc-results.json`

## What is connected

`backend/learning_dispatch.py` composes the two-qubit circuit in `backend/quantum_learning.py` with `backend/domain.py`'s `generate_routes` and `solve_exact`, and `backend/validator.py`'s independent `validate_plan` contract. It builds candidate routes from the supplied tiny dispatch scenario, enumerates exact-cover plan combinations with no vehicle assigned twice and no order omitted or duplicated, and keeps a candidate plan only when `validate_plan` recomputes it as feasible. The JSON preserves the selected raw route plans and independent recomputation separately.

The circuit's two feature rotations encode:

1. `x0 = min(candidate route travel_cost / 20, 1)` using the current scenario travel matrix.
2. `x1 = 1 - clipped nominal route slack / 4`, where nominal slack is computed from the unscaled source scenario's stop timing and contractual close times.

Its output becomes route SLA-event probability `p = sigmoid(-2 * <Z0>)`. Candidate-plan estimated cost is the sum of route scenario travel cost plus `p * (sum of route order penalty rates) * 1 minute`. The one-minute exposure is an explicit severity approximation. The plan loss for each training scenario is the soft-min probability over independently feasible plans (temperature 2.0), weighted by each plan's validator-recomputed scenario objective. This gives a nonzero parameter-shift gradient through the circuit and the candidate-plan objective. Inference selects the feasible plan with minimum estimated cost; the validator then recomputes coverage, capacity, stop timing, travel duration, lateness exposure, and total objective from scenario inputs.

The domain's objective is travel duration plus order penalty-rate times late minutes. It is a transparent cost proxy, **not calibrated currency**. There is no claim of financial exposure in dollars, forecast calibration, or operational fleet readiness.

### Information timing boundary

The scaled scenario travel matrix is available both when route features are built and when the validator computes realized outcomes. Thus the circuit sees scenario travel information that is reused as ground truth. This tests conditional plan selection when those scenario travel values are supplied. It does **not** test prospective delay-risk forecasting using only information available before travel occurs. Do not describe it as a predictive dispatch forecast.

## Frozen fixture and comparison

Source is `fixtures/dispatch_tiny.json`; its canonical JSON SHA-256 and each derived scenario hash are in the raw result. Each scenario scales every non-diagonal travel duration by the frozen multiplier vector in the result. Vehicle, capacity, demand, service, deadline, and penalty inputs remain fixed. Splits by scenario index are train 0–9, validation 10–11, test 12–15. Training uses 100 route rows across 10 train scenarios; all models receive identical candidate and test scenario inputs. The predictive VQC and logistic baseline fit route SLA-event labels on training scenarios only. Validation scenarios are preserved but were not used for model or hyperparameter selection.

| Test metric (4 scenarios) | Decision-focused VQC | Predictive-only VQC | Logistic |
|---|---:|---:|---:|
| Feasibility rate | 1.00 | 1.00 | 1.00 |
| Full order coverage rate | 1.00 | 1.00 | 1.00 |
| Mean validator objective | 26.600 | 26.600 | 26.600 |
| Mean validator travel cost | 24.000 | 24.000 | 24.000 |
| Mean validator lateness exposure | 2.600 | 2.600 | 2.600 |
| Mean model-estimated exposure | 1.368 | 1.736 | 0.824 |
| Mean regret vs exact tiny reference | 0.000 | 0.000 | 0.000 |

Every model selected the same route assignment on all four test scenarios: `V1: B -> C`, `V2: A`. Exact-reference regret is therefore zero for all three here; the experiment does not demonstrate a model-driven assignment change or advantage. All models underpredicted realized exposure on average. This is a useful failure signal for the one-minute severity approximation and the two-feature model, not a calibration result. The stress test includes route travel cost 21 while `x0` clips values above 20 to 1; this saturates the travel feature on that route and is another reason the stress result is only exploratory.

The dispatch decision-loss gradient was checked on two training scenarios with centered finite differences (`h=1e-5`): analytic `[0.0617138646, 0.0412223548]`, finite difference `[0.0617138647, 0.0412223548]`, maximum absolute error `8.08e-11`. The test artifact records per-route feature values, predicted probability and exposure, raw plan, validator result, exact reference, scenario hashes, counts, and local resource settings.

## Refinement record and evidence limits

| Cycle | Failure hypothesis and evidence | Material change | Relevant rerun |
|---|---|---|---|
| 1. Connect dispatch domain | The earlier VQC's two abstract costs had no vehicle, route, capacity, coverage, or validator contract. The repository domain provides those concrete interfaces. | Added route candidate generation, route-level inputs, exact-cover feasible-plan enumeration, a soft objective over validator-recomputed candidate plans, selected-plan validation, and matched model comparison. | `python -m unittest tests.test_quantum_learning -v`: dispatch gradient and exact plan validation tests passed; `python -m experiments.run_dispatch_vqc`: all selected plans validated. |
| 2. Make circuit features distinguish route structure | Initial integrated run selected identical plans and used a scenario multiplier feature shared across routes, so stop-order risk was not represented. | Replaced that feature with nominal contractual window pressure, which varies by route order, and retained scenario travel cost as the second actual route-specific input. | `python -m experiments.run_dispatch_vqc`: four test scenarios, feature values and model estimates recorded. Plans remained identical across models; this did not establish an advantage. |
| 3. Exercise SLA exposure under congestion | The preceding diagnostic test split had zero realized lateness for selected plans, so it could not exercise positive penalty exposure. | Expanded the frozen multiplier vector to include fixed high-congestion scenarios and reported estimated versus validator-realized exposure separately. No model parameter was selected using the new test outcomes. | `python -m experiments.run_dispatch_vqc`: test set included 2.6 mean realized lateness exposure; all models still selected the same feasible plan and had zero exact-reference regret. |

The current scenario design and feature iteration occurred after an earlier integration test split had been inspected. Consequently, the final artifact is explicitly **exploratory and not blind**, even though the final model weights were fit only on train scenarios and no validation/test labels enter the training loss. A fresh, separately frozen prospective test set is required before making a forecasting or comparative decision-quality claim. Test-only assertions are verification hardening and are not counted as a refinement cycle.

## Limits

- All scenario data is synthetic; the source is a three-order fixture with two vehicles.
- Route candidate travel values are simultaneously input features and validator scenario values, so future travel uncertainty is not forecast.
- `x0` clips route travel costs above 20; at least one stress-test candidate reaches 21 and loses magnitude information.
- Predicted SLA event probability times one minute of penalty exposure is an approximation; no calibration study was run.
- The four test cases are too few to support uncertainty estimates or generalization claims.
- QpiAI Quantum, QCloud, QpiAI Opt, authenticated APIs, shots, and QPU were not invoked.
- No quantum advantage, novel method, operational feasibility, or currency-valued exposure is claimed.
