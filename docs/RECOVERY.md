# Tiny dispatch recovery

`backend.recovery.recover_dispatch(...)` provides bounded exact recovery for
the existing tiny dispatch model. It accepts an original scenario, its
original plan, updated scenario inputs, and `locked_prefixes`, a mapping from
vehicle ID to the ordered stops that must stay assigned to that vehicle.
Every prefix must match the beginning of that vehicle's route in the original
plan, and that plan must pass `backend.validator.validate_plan` first.

The helper enumerates capacity-feasible routes, keeps only routes that preserve
each locked prefix, and searches for an exact cover of all orders. Unlocked
orders may move between available vehicles. A removed vehicle with locked work
returns `locked_work_vehicle_unavailable`; a disruption with no feasible plan
returns `recovery_infeasible`. Unsupported order, depot, or vehicle-capacity
changes are rejected. The original plan is copied into the response and is not
modified.

Search is deliberately capped at 2,000 generated route candidates and 10,000
recursive search nodes. Exceeding these limits returns `unsupported_capability`
or `recovery_budget_exceeded`, respectively, and does not return a partial
recovery plan. The fixed limits bound this local demonstration; larger problems
need a separately engineered solver and resource budget.

`change_penalty` is optional and defaults to zero. It adds a uniform selection
cost for each order reassigned to a different vehicle. The response reports
both this selection cost and the plan's independently recomputed travel plus
lateness objective; the validator objective never includes the change penalty.
The response also compares objective, exposure, assignment changes, and route
sequence changes. When feasible, `static_original_baseline` keeps the original
vehicle assignment and order sequence but recomputes route timing and cost
under the updated scenario, then independently validates it. This baseline
measures the effect of the updated inputs on the old dispatch; it does not
reuse stale timing from the original plan. If unavailable vehicles or capacity
make that assignment invalid, the response marks the baseline unavailable.

Recovery recomputes all route times by replaying from the depot at time zero.
That does not establish an actual vehicle's current location, elapsed service,
or live clock. This implementation has no live vehicle-state integration. If
an operator cannot establish that the supplied prefixes and assignments still
match the actual fleet state, retain the last independently validated plan and
do not apply the proposed recovery. This is a tiny-instance local classical
search, not a VQC or live dispatch capability.

Run the bounded tests with:

```powershell
& 'C:\Users\ommd3\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe' -m unittest tests.test_recovery -v
```
