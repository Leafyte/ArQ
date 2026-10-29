# Scenario-based dispatch risk scoring

`backend.risk.evaluate_plan_risk(forecast_scenario, nominal_scenario, plan,
realized_scenarios, *, cvar_alpha=None)` scores a **fixed** dispatch plan over
explicit alternative travel matrices. The plan must first pass the independent
validator under the forecast scenario. Nominal and forecast scenarios must
describe the same depot, vehicles, and orders.

Each realized case has exactly this shape:

```python
{"scenario_id": "case-01", "travel": realized_travel_matrix}
```

Use the same ordered case list and IDs for every method being compared; this is
the common-random-number (CRN) pairing contract. IDs must be unique and travel
matrices must validate completely. Repeated matrices are permitted under
distinct IDs: each supplied row is one empirical observation with weight
`1/N`, including repeated outcomes. This allows distinct scenarios to share
an outcome, as can happen with zero-noise or discrete simulations. The
evaluator fails the full request on any malformed case; it never drops an
invalid case.

The evaluator keeps every assignment and stop order from the supplied plan. It
reconstructs arrival, service start, departure, lateness, penalty exposure,
travel cost, and return time for every route in every case, starting each
vehicle at the depot at time zero. It passes each reconstructed plan to the
independent validator with that case's travel matrix. The response includes
every case's matrix hash, stop-level replay, recomputed totals, and validation
report.

The summary reports empirical probability of any late stop, late-stop
frequency, per-order late frequency, mean travel cost, mean penalty exposure,
and mean proxy objective. The estimator is an equal-weight finite-sample
estimate: each of the `N` scenarios receives weight `1/N`. The output reports
`N` and exact SHA-256 hashes for the forecast, nominal, input plan, each travel
matrix, and ordered scenario set. It does not provide confidence intervals or
claim calibrated population risk.

Optional `cvar_alpha` must be in `[0, 1)`. At confidence level alpha, the
system-wide CVaR is the empirical mean over exactly the worst `(1-alpha)` mass
of scenario outcomes. If the tail mass ends between observations, the boundary
scenario contributes fractionally. Both penalty-exposure CVaR and total proxy
objective CVaR are returned. For example, with two equally weighted outcomes
and alpha `0.75`, the tail has mass `0.5` scenarios; CVaR is the worse outcome,
not a rounded or interpolated percentile.

This is a tiny synthetic or user-supplied scenario evaluator, not an outcome
forecast. Penalty/travel units are the dispatch domain's uncalibrated proxy
units. No currency, contractual risk, operational readiness, or live-state
claim follows from the score. The evaluator does not call QpiAI, cloud
services, or a QPU.

Bounds are at most 8 orders, 16 vehicles, and 100 distinct realized matrices.
Input and plan validation errors raise `RiskError`; `as_dict()` provides a
stable error payload. Principal codes include `invalid_scenario`,
`invalid_plan`, `invalid_scenarios`, `invalid_cvar_alpha`,
`unsupported_capability`, and `validation_failure`.

Run the focused unit tests in the isolated project environment:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_risk -v
```
