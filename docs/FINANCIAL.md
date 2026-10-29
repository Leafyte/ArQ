# Explicit financial dispatch scoring

`backend.financial.solve_financial_dispatch(scenario, *, currency,
operating_cost_per_travel_minute, penalty_rate_unit)` searches for a feasible
exact-cover plan using monetary units supplied by the caller. `currency` must
be an uppercase three-letter ISO-style code (syntax only; no currency registry
or exchange-rate lookup is performed). The operating rate must be finite and
non-negative. The required `penalty_rate_unit` must be exactly
`"currency_per_late_minute"`; every order's existing `penalty` value is
interpreted in that unit. Missing or unspecified rates are rejected.

The selected score is:

```text
sum(route travel minutes * operating cost per travel minute)
+ sum(order lateness minutes * source order penalty rate)
```

The returned `plan` retains the domain's ordinary plan fields. Those fields
remain **proxy** timing/objective outputs and are reported separately from
`financial_objective`. `proxy_validation` independently validates coverage,
capacity, route timing, and domain objective. `financial_validation` then
recomputes route timing and monetary cost directly from the raw scenario's
travel matrix, windows, services, and penalty rates, and compares them with the
reported financial fields. It does not trust a solver-supplied monetary cost.
`validate_financial_plan` exposes the same independent check for external
reports and fails a tampered cost.

Calculations retain Python binary floating-point precision throughout; no
rounding is applied. Format for display only at the client boundary, and do not
feed rounded display values back into planning. This module does not verify
that supplied rates reflect actual contracts, wages, fuel, currency, or
accounting rules. It makes no cost-calibration or operational-readiness claim.

Bounds are six orders, eight vehicles, a conservative candidate-generation
upper bound of 3,000, and 5,000 deterministic search work units. Exceeding any
bound returns `unsupported_capability` or `work_budget_exceeded` without a
partial plan. No feasible exact cover returns `infeasible`. The method is a
local classical exact search; it does not call QpiAI, a cloud service, or a
QPU.

Run the focused checks in the project environment:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_financial -v
```
