# Bounded VQC inference interface

`backend.learning_inference.predict_dispatch` runs one synchronous local inference using caller-supplied frozen parameters. It does not train, tune, generate realized outcomes, or call a cloud service. `execution_backend` is optional and defaults to `reference` for compatibility.

```python
from backend.learning_inference import predict_dispatch

result = predict_dispatch(
    forecast_scenario=forecast,    # strict domain scenario; forecast travel matrix
    nominal_scenario=nominal,      # same depot/fleet/orders; nominal travel matrix
    theta=[-0.0925538742, -0.4402943343],
    model_version="prospective-synthetic-v1",
    execution_backend="reference",  # or "qpiai_local"
    financial_terms={
        "currency": "USD",
        "operating_cost_per_travel_minute": 1.25,
        "penalty_rate_unit": "currency_per_late_minute",
    },  # optional; omit to retain the legacy proxy objective
)
```

The example parameter vector is from the synthetic prospective experiment. It is not an approved production model. Callers must supply a frozen two-angle vector and a version identifier; the response returns a SHA-256 fingerprint of the vector. The supplied version string and vector are not cryptographically authenticated or checked against a trusted model registry. The response labels their training origin as **unverified caller-supplied input** and does not infer synthetic lineage or calibration status from the version string.

## Input and bounds

Both scenario objects must have exactly `depot`, `vehicles`, `orders`, and `travel`. Vehicle objects have `id` and `capacity`; order objects require `id`, `demand`, `window`, `service`, and `penalty`, and may include `hard_window` as a boolean. Omitted `hard_window` means `false`. Forecast and nominal scenarios must have matching effective hard-window flags and identical remaining depot, vehicle, and order definitions; their travel matrices may differ. Malformed flags or mismatches return `invalid_scenario`. Unknown fields, including any `realized` or outcome field, are rejected.

The bounded local path accepts at most 3 orders, 4 vehicles, 20 generated candidate routes, and 500 exact-cover plans. `theta` must be a list/tuple of exactly two finite numbers with absolute value at most 10. `model_version` is a required non-empty string of at most 128 characters. These bounds keep route enumeration finite for the current fixture; larger inputs return `unsupported_capability`.

Optional `financial_terms` must contain exactly `currency`,
`operating_cost_per_travel_minute`, and `penalty_rate_unit`. Currency syntax is
an uppercase three-letter ISO-style string; this is not checked against a
currency registry. The operating rate must be finite and non-negative. The
penalty unit must explicitly equal `currency_per_late_minute`. Missing, null,
extra, or malformed terms return `invalid_financial_terms`; there is no
default currency or rate. Omitting the field preserves the legacy proxy mode.

## Inference and validation

The VQC is the exact two-qubit gate circuit recorded by `backend.quantum_learning.CircuitMetadata`. It encodes forecast-route travel duration (`min(travel_cost/30,1)`) and nominal contractual window pressure. The risk output is `sigmoid(-2 <Z0>)`.

When `hard_window` is true, `backend.domain.generate_routes` excludes any
forecast route with positive lateness at that order; exact-cover selection then
returns `infeasible` if no hard-feasible cover exists. The selected plan is
also checked by the independent validator, which recomputes hard-window
constraints. Nominal travel is used only to calculate a model feature; nominal
route schedules may be hard-infeasible without making the forecast plan
infeasible. The flag itself must still match between nominal and forecast
inputs.

Without `financial_terms`, the legacy route score is forecast travel duration
plus one minute of probability-weighted source penalties. With explicit terms,
the score is `forecast travel minutes * supplied operating rate + risk
probability * sum(source penalty rates) * 1 minute`. The latter is an
uncalibrated caller-unit estimate; the one-minute expected lateness assumption
is explicit and is not learned or calibrated. The supplied VQC vector has
unverified training provenance and is not known to have been trained for the
caller-provided financial rates.

An exact-cover enumerator checks distinct vehicle use and complete,
non-overlapping order coverage. The selected route plan is passed to
`backend.validator.validate_plan` with the **forecast** scenario, which
independently recomputes its constraints, timing, and domain proxy objective.
In financial mode, the response's `financial_estimate` reports the
model-estimated transport cost, exposure, and total in the declared currency.
`backend.financial.validate_financial_plan` separately recomputes the selected
plan's deterministic monetary objective directly from forecast travel, order
windows, and penalty rates. This `forecast_financial_objective` is based on
the supplied forecast scenario's recomputed lateness, not on model-estimated
future risk or an observed realized outcome. `financial_validation` records
the independent check. The plan's ordinary `objective` remains the domain
proxy value and is not relabeled as money.

`reference` uses the project exact statevector implementation. `qpiai_local` calls `backend.qpiai_vqc.sdk_risk_probability`, backed by QpiAI Quantum's local `StatevectorSimulator`; it makes no cloud calls. The result labels the selected execution backend and SDK version in model provenance. There is no automatic fallback: a missing SDK returns `unsupported_capability`, and an SDK runtime failure returns `execution_failure`.

The validator result proves consistency with the supplied forecast matrix only. No realized travel or future outcome is available to this function, so it cannot prove actual SLA outcomes, prospective forecast accuracy, or feasibility under unobserved conditions. The penalty and travel units come from the tiny domain proxy and are not currency-calibrated. The model vector's training origin remains unverified and it has no calibration guarantee.

## Return and errors

Success returns `status`, `execution_type`, model and surrogate provenance,
chosen forecast plan, estimated plan cost/exposure, per-route features and
risk probabilities, and the independent forecast-plan validation report. If
financial terms are provided, cost and exposure estimates are in the declared
currency and the response additionally includes `financial_terms`,
`estimated_cost_currency`, `financial_estimate`,
`forecast_financial_objective`, `financial_validation`, and a model-suitability
warning.

`InferenceError.as_dict()` returns the stable `{error: {code, message, details}}` shape. Codes are:

- `invalid_scenario` — malformed or inconsistent nominal/forecast input
- `invalid_model_parameters` — invalid vector or version identifier
- `unsupported_capability` — input exceeds local route/plan bounds
- `invalid_execution_backend` — `execution_backend` is not `reference` or `qpiai_local`
- `invalid_financial_terms` — explicit financial terms are malformed or lack the required unit/rate
- `financial_validation_failure` — independent source-input financial recomputation failed
- `execution_failure` — the selected local simulator failed; no alternate backend was substituted
- `infeasible` — no capacity-feasible exact cover exists
- `validation_failure` — selected forecast plan fails the independent validator

Unexpected implementation exceptions should be caught by the API boundary and mapped to its generic `inference_failure` response; raw exception text and input values should not be returned to clients.

## Verification

Run the checks in both isolated project environments:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_learning_inference -v
& '.\.venv-sdk\Scripts\python.exe' -m unittest tests.test_learning_inference -v
```

The tests cover deterministic inference, differing forecast travel, model/version bounds, strict schema and problem identity, infeasibility, unsupported sizes, and independent plan validation. When the SDK is installed, they compare SDK probabilities and selected plans with the reference backend; when absent, they verify explicit `unsupported_capability`. They also verify that SDK runtime failure does not fall back silently. They do not run training or inspect held-out prospective outcomes.
