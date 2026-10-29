"""Conservative, bounded dispatch recovery for tiny scenarios.

This static depot-replay helper preserves caller-specified route prefixes
and independently validates the selected plan against updated inputs. It
does not observe live vehicle locations, clocks, or telemetry.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from .domain import generate_routes, validate_scenario
from .validator import validate_plan

MAX_RECOVERY_ROUTE_CANDIDATES = 2_000
MAX_RECOVERY_SEARCH_NODES = 10_000


class RecoveryError(ValueError):
    """Stable machine-readable recovery failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


def recover_dispatch(
    original_scenario: dict[str, Any],
    original_plan: dict[str, Any],
    updated_scenario: dict[str, Any],
    locked_prefixes: dict[str, list[str]],
    *,
    change_penalty: float = 0.0,
) -> dict[str, Any]:
    """Select a feasible exact-cover plan while preserving ordered prefixes.

    Each supplied prefix must match the start of that vehicle's route in the
    independently validated original plan. ``change_penalty`` adds a uniform
    selection cost per order reassigned to a different vehicle; it does not
    alter the objective recomputed by the independent validator.
    """
    _require_scenario(original_scenario, "invalid_original_scenario")
    _require_scenario(updated_scenario, "invalid_updated_scenario")
    original_check = validate_plan(original_scenario, original_plan)
    if not original_check["valid"]:
        raise RecoveryError("original_plan_unvalidated", "original plan must pass independent validation")
    if (original_scenario["depot"] != updated_scenario["depot"] or
            original_scenario["orders"] != updated_scenario["orders"]):
        raise RecoveryError("unsupported_disruption", "recovery supports travel and vehicle availability changes only")
    original_vehicles = {v["id"]: v for v in original_scenario["vehicles"]}
    updated_vehicles_by_id = {v["id"]: v for v in updated_scenario["vehicles"]}
    if any(vid not in original_vehicles or original_vehicles[vid] != vehicle
           for vid, vehicle in updated_vehicles_by_id.items()):
        raise RecoveryError("unsupported_disruption", "vehicle additions and capacity changes are unsupported")
    if not isinstance(change_penalty, (int, float)) or isinstance(change_penalty, bool) or not isfinite(change_penalty) or change_penalty < 0:
        raise RecoveryError("invalid_change_penalty", "change penalty must be finite and non-negative")
    original_routes = {r["vehicle_id"]: r for r in original_plan["routes"]}
    original_vehicle_ids = {v["id"] for v in original_scenario["vehicles"]}
    if not isinstance(locked_prefixes, dict):
        raise RecoveryError("invalid_locked_prefix", "locked prefixes must map vehicle IDs to ordered order IDs")
    updated_vehicles = {v["id"] for v in updated_scenario["vehicles"]}
    prefixes: dict[str, tuple[str, ...]] = {}
    for vehicle_id, raw_prefix in locked_prefixes.items():
        if not isinstance(vehicle_id, str) or not isinstance(raw_prefix, list) or any(not isinstance(x, str) for x in raw_prefix):
            raise RecoveryError("invalid_locked_prefix", "each locked prefix must be an ordered list of order IDs")
        prefix = tuple(raw_prefix)
        original_route = original_routes.get(vehicle_id)
        if (vehicle_id not in original_vehicle_ids or
                (prefix and (original_route is None or tuple(original_route["order_ids"][:len(prefix)]) != prefix))):
            raise RecoveryError("locked_prefix_unproven", "locked prefix must match the validated original vehicle route")
        if vehicle_id not in updated_vehicles and prefix:
            raise RecoveryError("locked_work_vehicle_unavailable", "a vehicle holding locked work is unavailable")
        prefixes[vehicle_id] = prefix

    original_assignment = {oid: route["vehicle_id"] for route in original_plan["routes"] for oid in route["order_ids"]}
    generated = generate_routes(updated_scenario)
    if len(generated) > MAX_RECOVERY_ROUTE_CANDIDATES:
        raise RecoveryError("unsupported_capability", "recovery route pool exceeds its local bound")
    candidates: dict[str, list[dict[str, Any] | None]] = {}
    for vehicle in updated_scenario["vehicles"]:
        vid = vehicle["id"]
        prefix = prefixes.get(vid, ())
        matching = [route for route in generated
                    if route["vehicle_id"] == vid and tuple(route["order_ids"][:len(prefix)]) == prefix]
        if prefix and not matching:
            raise RecoveryError("locked_prefix_infeasible", "updated inputs cannot serve the locked prefix with this vehicle")
        candidates[vid] = matching + ([None] if not prefix else [])

    vehicles = sorted(candidates)
    order_ids = {o["id"] for o in updated_scenario["orders"]}
    best: tuple[float, tuple[dict[str, Any], ...], int] | None = None
    search_nodes = 0

    def search(index: int, covered: frozenset[str], chosen: tuple[dict[str, Any], ...], moved: int, cost: float) -> None:
        nonlocal best, search_nodes
        search_nodes += 1
        if search_nodes > MAX_RECOVERY_SEARCH_NODES:
            raise RecoveryError("recovery_budget_exceeded", "recovery search exceeded its bounded node budget")
        if index == len(vehicles):
            if covered == order_ids and (best is None or cost < best[0]):
                best = (cost, chosen, moved)
            return
        vid = vehicles[index]
        for route in candidates[vid]:
            if route is None:
                search(index + 1, covered, chosen, moved, cost)
                continue
            route_orders = set(route["order_ids"])
            if covered & route_orders:
                continue
            changes = sum(original_assignment[oid] != vid for oid in route_orders)
            search(index + 1, covered | route_orders, chosen + (route,), moved + changes,
                   cost + float(route["objective"]) + float(change_penalty) * changes)

    search(0, frozenset(), (), 0, 0.0)
    if best is None:
        raise RecoveryError("recovery_infeasible", "no exact-cover plan preserves the requested locked prefixes")
    _, selected, moved = best
    routes = sorted((deepcopy(route) for route in selected), key=lambda row: row["vehicle_id"])
    recovered = {"status": "feasible", "routes": routes,
                 "travel_cost": sum(r["travel_cost"] for r in routes),
                 "lateness_exposure": sum(r["lateness_exposure"] for r in routes)}
    recovered["objective"] = recovered["travel_cost"] + recovered["lateness_exposure"]
    validation = validate_plan(updated_scenario, recovered)
    if not validation["valid"]:
        raise RecoveryError("validation_failure", "recovered plan failed independent validation")
    original_copy = deepcopy(original_plan)
    original_route_sequences = {r["vehicle_id"]: list(r["order_ids"]) for r in original_plan["routes"]}
    recovered_sequences = {r["vehicle_id"]: list(r["order_ids"]) for r in recovered["routes"]}
    original_objective = float(original_check["recomputed"]["objective"])
    updated_objective = float(validation["recomputed"]["objective"])
    # Static baseline keeps the original assignment and order sequence, but
    # recomputes all timing and cost from the updated scenario inputs.
    generated_by_key = {(r["vehicle_id"], tuple(r["order_ids"])): r for r in generated}
    baseline_routes = []
    for old_route in original_plan["routes"]:
        candidate = generated_by_key.get((old_route["vehicle_id"], tuple(old_route["order_ids"])))
        if candidate is None:
            baseline_routes = []
            break
        baseline_routes.append(deepcopy(candidate))
    baseline = None
    if baseline_routes or not updated_scenario["orders"]:
        baseline_routes.sort(key=lambda row: row["vehicle_id"])
        static_plan = {"status": "feasible", "routes": baseline_routes,
                       "travel_cost": sum(r["travel_cost"] for r in baseline_routes),
                       "lateness_exposure": sum(r["lateness_exposure"] for r in baseline_routes)}
        static_plan["objective"] = static_plan["travel_cost"] + static_plan["lateness_exposure"]
        baseline_validation = validate_plan(updated_scenario, static_plan)
        if baseline_validation["valid"]:
            baseline = {"available": True, "plan": static_plan,
                        "validation": baseline_validation,
                        "comparison_basis": "original assignment and order sequence; timing and cost recomputed under updated inputs"}
    if baseline is None:
        baseline = {"available": False,
                    "reason": "original assignment and order sequence are infeasible under updated inputs"}
    return {
        "status": "recovered", "execution_type": "local_classical_discrete_search",
        "original_plan": original_copy, "recovered_plan": recovered,
        "validation": validation,
        "static_original_baseline": baseline,
        "comparison": {
            "original_objective": original_objective,
            "updated_objective": updated_objective,
            "objective_delta": updated_objective - original_objective,
            "original_exposure": original_check["recomputed"]["lateness_exposure"],
            "updated_exposure": validation["recomputed"]["lateness_exposure"],
            "reassigned_order_count": moved,
            "change_penalty": float(change_penalty),
            "selection_cost_including_change_penalty": updated_objective + float(change_penalty) * moved,
            "route_sequences_changed": original_route_sequences != recovered_sequences,
            "search_nodes": search_nodes,
        },
        "state_basis": "depot_replay; live vehicle location and clock are not available",
    }


def _require_scenario(scenario: Any, code: str) -> None:
    if not isinstance(scenario, dict) or not validate_scenario(scenario)["valid"]:
        raise RecoveryError(code, "scenario failed validation")
