"""Independent plan checker that rebuilds route facts from scenario inputs.

Public interface: ``validate_plan(scenario, plan)`` returns
``{valid: bool, feasible: bool, errors: [...], recomputed: {...}}``. It does
not call the route generator or trust solver-reported ETAs/objective fields.
"""

from __future__ import annotations

from math import isclose
from typing import Any

from .domain import validate_scenario


def validate_plan(scenario: dict[str, Any], plan: Any) -> dict[str, Any]:
    scenario_check = validate_scenario(scenario)
    if not scenario_check["valid"]:
        return {"valid": False, "feasible": False,
                "errors": ["scenario invalid: " + "; ".join(scenario_check["errors"])],
                "recomputed": None}
    errors: list[str] = []
    if not isinstance(plan, dict):
        return {"valid": False, "feasible": False, "errors": ["plan: expected an object"], "recomputed": None}
    if plan.get("status") != "feasible":
        return {"valid": False, "feasible": False,
                "errors": [f"plan.status: expected 'feasible', got {plan.get('status')!r}"], "recomputed": None}
    routes = plan.get("routes")
    if not isinstance(routes, list):
        return {"valid": False, "feasible": False, "errors": ["routes: expected a list"], "recomputed": None}
    vehicles = {v["id"]: v for v in scenario["vehicles"]}
    orders = {o["id"]: o for o in scenario["orders"]}
    used_vehicles: set[str] = set()
    served: list[str] = []
    recomputed_routes = []
    for ri, route in enumerate(routes):
        prefix = f"routes[{ri}]"
        if not isinstance(route, dict):
            errors.append(f"{prefix}: expected an object")
            continue
        vid = route.get("vehicle_id")
        if not isinstance(vid, str) or vid not in vehicles:
            errors.append(f"{prefix}.vehicle_id: unknown vehicle {vid!r}")
            continue
        if vid in used_vehicles:
            errors.append(f"{prefix}.vehicle_id: vehicle {vid!r} assigned more than once")
        used_vehicles.add(vid)
        ids = route.get("order_ids")
        if not isinstance(ids, list) or not ids:
            errors.append(f"{prefix}.order_ids: expected a non-empty ordered list")
            continue
        unknown = [oid for oid in ids if not isinstance(oid, str) or oid not in orders]
        if unknown:
            errors.append(f"{prefix}.order_ids: unknown orders {unknown!r}")
            continue
        served.extend(ids)
        load = sum(orders[oid]["demand"] for oid in ids)
        if load > vehicles[vid]["capacity"] + 1e-9:
            errors.append(f"{prefix}: load {load} exceeds vehicle capacity {vehicles[vid]['capacity']}")

        # Recompute from raw input, deliberately duplicating the timing/objective
        # equations instead of importing the candidate route implementation.
        current_location = scenario["depot"]
        clock = 0.0
        travel_cost = 0.0
        exposure = 0.0
        stop_metrics = []
        for oi, oid in enumerate(ids):
            order = orders[oid]
            leg = float(scenario["travel"][current_location][oid])
            travel_cost += leg
            arrival = clock + leg
            start = max(arrival, float(order["window"][0]))
            late = max(0.0, start - float(order["window"][1]))
            if order.get("hard_window", False) and late > 0:
                errors.append(f"{prefix}.stops[{oi}].hard_window: service starts after hard close")
            penalty = late * float(order["penalty"])
            exposure += penalty
            departure = start + float(order["service"])
            stop_metrics.append({"order_id": oid, "arrival": arrival, "service_start": start,
                                 "departure": departure, "lateness": late, "penalty_exposure": penalty})
            # If submitted stop details are present, check each independently.
            submitted_stops = route.get("stops")
            if not isinstance(submitted_stops, list) or oi >= len(submitted_stops):
                errors.append(f"{prefix}.stops: missing details for {oid!r}")
            else:
                submitted = submitted_stops[oi]
                for field, expected in stop_metrics[-1].items():
                    actual = submitted.get(field) if isinstance(submitted, dict) else None
                    if field == "order_id":
                        if actual != expected:
                            errors.append(f"{prefix}.stops[{oi}].{field}: does not match ordered stop")
                    elif not _matches(actual, expected):
                        errors.append(f"{prefix}.stops[{oi}].{field}: expected {expected}, got {actual!r}")
            current_location, clock = oid, departure
        if isinstance(route.get("stops"), list) and len(route["stops"]) != len(ids):
            errors.append(f"{prefix}.stops: count does not match order_ids")
        back = float(scenario["travel"][current_location][scenario["depot"]])
        travel_cost += back
        route_values = {"return_time": clock + back, "travel_cost": travel_cost,
                        "lateness_exposure": exposure, "objective": travel_cost + exposure}
        for field, expected in route_values.items():
            actual = route.get(field)
            if not _matches(actual, expected):
                errors.append(f"{prefix}.{field}: expected {expected}, got {actual!r}")
        recomputed_routes.append({"vehicle_id": vid, "order_ids": list(ids), **route_values,
                                  "stops": stop_metrics})
    for oid in orders:
        count = served.count(oid)
        if count != 1:
            errors.append(f"coverage.{oid}: expected exactly once, found {count}")
    totals = {"travel_cost": sum(r["travel_cost"] for r in recomputed_routes),
              "lateness_exposure": sum(r["lateness_exposure"] for r in recomputed_routes),
              "objective": sum(r["objective"] for r in recomputed_routes)}
    for field, expected in totals.items():
        if not _matches(plan.get(field), expected):
            errors.append(f"plan.{field}: expected {expected}, got {plan.get(field)!r}")
    return {"valid": not errors, "feasible": not errors, "errors": errors,
            "recomputed": {**totals, "routes": recomputed_routes}}


def _matches(actual: Any, expected: float) -> bool:
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool) and
            isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9))
