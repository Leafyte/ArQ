"""Route-candidate QUBO/QAOA adapter into the independently checked domain."""
from __future__ import annotations

from typing import Any, Sequence

from .domain import generate_routes, solve_exact, validate_scenario
from .qaoa import QAOAResult, continuous_relaxation, run_qaoa
from .qubo import Qubo, build_route_qubo
from .validator import validate_plan


def _plan_for_bits(candidates: Sequence[dict[str, Any]], bits: str) -> dict[str, Any]:
    routes = [route for route, bit in zip(candidates, bits) if bit == "1"]
    return {
        "status": "feasible", "routes": routes,
        "objective": sum(route["objective"] for route in routes),
        "travel_cost": sum(route["travel_cost"] for route in routes),
        "lateness_exposure": sum(route["lateness_exposure"] for route in routes),
    }


def _attach_independent_checks(scenario: dict[str, Any], candidates: Sequence[dict[str, Any]],
                               result: QAOAResult) -> None:
    """Check each distinct raw and repaired bitstring once, retaining all rows."""
    cache: dict[str, dict[str, Any]] = {}

    def check(bits: str) -> dict[str, Any]:
        if bits not in cache:
            cache[bits] = validate_plan(scenario, _plan_for_bits(candidates, bits))
        return cache[bits]

    for sample in result.raw_samples:
        sample["decoded"] = {"bitstring": sample["bits"],
                              "selected_route_ids": sample["validation"]["selected_route_ids"]}
        sample["independent_plan_validation"] = check(sample["bits"])
        if sample.get("repair"):
            repaired = sample["repair"]["bits"]
            sample["repair"]["independent_plan_validation"] = check(repaired)


def solve_qaoa_dispatch(
    scenario: dict[str, Any], *, seed: int = 7, shots: int = 512,
    warm_start: bool = True, angle_grid: int = 10,
) -> dict[str, Any]:
    """Optimize generated route candidates, independently validate, and fallback.

    A sampled QUBO-feasible bitstring is accepted only after `validate_plan`
    recomputes domain coverage, capacity, ordered ETAs and business objective.
    Repaired samples are preserved and checked but never treated as quantum
    output. If no raw sample survives, exact classical dispatch is explicit.
    """
    checked = validate_scenario(scenario)
    if not checked["valid"]:
        return {"status": "invalid_input", "errors": checked["errors"],
                "execution_type": None, "decision": None,
                "fallback": {"used": False, "reason": None}}
    if isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 10_000:
        return {"status": "invalid_input", "errors": ["shots: expected an integer in [1, 10000]"],
                "execution_type": None, "decision": None, "fallback": {"used": False, "reason": None}}
    if isinstance(angle_grid, bool) or not isinstance(angle_grid, int) or not 2 <= angle_grid <= 16:
        return {"status": "invalid_input", "errors": ["angle_grid: expected an integer in [2, 16]"],
                "execution_type": None, "decision": None, "fallback": {"used": False, "reason": None}}
    if not scenario["orders"]:
        plan = {"status": "feasible", "routes": [], "objective": 0.0,
                "travel_cost": 0.0, "lateness_exposure": 0.0}
        validation = validate_plan(scenario, plan)
        return {"status": "success" if validation["valid"] else "validation_failed",
                "execution_type": "classical_trivial", "decision": {"origin": "empty_instance", "plan": plan,
                "validation": validation}, "fallback": {"used": False, "reason": None},
                "optimization": {"skipped": "no orders"}}

    # Candidate pool contains route facts from domain inputs. QUBO selects only;
    # feasibility/financial acceptance remains the independent domain validator.
    route_pool = generate_routes(scenario)
    candidates = []
    for index, route in enumerate(route_pool):
        route_id = f"{route['vehicle_id']}:{index}:" + ",".join(route["order_ids"])
        candidates.append({"id": route_id, "vehicle": route["vehicle_id"],
                           "stops": route["order_ids"], "cost": route["objective"]})

    qaoa_result: QAOAResult | None = None
    q: Qubo | None = None
    failure: dict[str, str] | None = None
    selected_plan: dict[str, Any] | None = None
    selected_validation: dict[str, Any] | None = None
    selected_bits: str | None = None
    origin = "qaoa_raw_sample"
    fallback_reason: str | None = None
    if len(candidates) > 12:
        failure = {"code": "UNSUPPORTED_CAPABILITY", "error_type": "RouteVariableLimit",
                   "limit": 12, "candidate_count": len(candidates)}
        fallback_reason = "candidate pool exceeds the 12-qubit local statevector limit"
    else:
        try:
            q = build_route_qubo(candidates, [o["id"] for o in scenario["orders"]])
            relaxation = continuous_relaxation(q) if warm_start else None
            qaoa_result = run_qaoa(q, seed=seed, shots=shots, warm_start=warm_start,
                                   relaxation=relaxation, angle_grid=angle_grid)
            _attach_independent_checks(scenario, route_pool, qaoa_result)
            good = [row for row in qaoa_result.raw_samples
                    if row["validation"]["feasible"] and row["independent_plan_validation"]["valid"]]
            if good:
                chosen = min(good, key=lambda row: (row["validation"]["business_cost"], row["bits"]))
                selected_bits = chosen["bits"]
                selected_plan = _plan_for_bits(route_pool, selected_bits)
                selected_validation = chosen["independent_plan_validation"]
            else:
                fallback_reason = "no sampled raw QAOA bitstring passed both QUBO and independent domain validation"
        except Exception as exc:
            # Keep errors stable and omit exception text that could contain input or environment details.
            failure = {"code": "QAOA_FAILED", "error_type": type(exc).__name__}
            fallback_reason = "QAOA candidate construction or local run failed"

    if selected_plan is None:
        origin = "classical_exact_fallback"
        exact = solve_exact(scenario)
        if exact.get("status") == "feasible":
            validation = validate_plan(scenario, exact)
            if validation["valid"]:
                selected_plan, selected_validation = exact, validation
            else:
                return {
                    "status": "validation_failed", "execution_type": "classical",
                    "decision": {"origin": origin, "plan": exact, "validation": validation},
                    "fallback": {"used": True, "reason": fallback_reason},
                    "optimization": {"qaoa_failure": failure, "raw_result": qaoa_result.__dict__ if qaoa_result else None},
                }
        else:
            return {
                "status": "infeasible", "execution_type": "classical",
                "decision": {"origin": origin, "plan": exact, "validation": None},
                "fallback": {"used": True, "reason": fallback_reason or exact.get("reason")},
                "optimization": {"qaoa_failure": failure, "raw_result": qaoa_result.__dict__ if qaoa_result else None},
            }

    optimization: dict[str, Any] = {
        "candidate_count": len(candidates), "variable_order": [c["id"] for c in candidates],
        "execution_type": qaoa_result.backend if qaoa_result else "not_run",
        "qaoa_failure": failure,
        "selected_raw_bits": selected_bits,
        "raw_result": qaoa_result.__dict__ if qaoa_result else None,
    }
    if q is not None:
        optimization["qubo"] = {"linear": list(q.linear),
                                 "quadratic": {f"{i},{j}": v for (i, j), v in q.quadratic.items()},
                                 "offset": q.offset, "coverage_penalty": q.coverage_penalty,
                                 "vehicle_penalty": q.vehicle_penalty}
    return {
        "status": "success", "execution_type": "local_simulator" if origin == "qaoa_raw_sample" else "classical",
        "decision": {"origin": origin, "plan": selected_plan, "validation": selected_validation},
        "fallback": {"used": origin == "classical_exact_fallback", "reason": fallback_reason},
        "optimization": optimization,
    }
