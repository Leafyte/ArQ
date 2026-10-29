"""Classical dispatch domain for tiny deterministic scenarios.

Scenario contract: ``vehicles`` contain ``id`` and ``capacity``; ``orders``
contain ``id``, ``demand``, ``window=[open, close]``, ``service`` duration,
and ``penalty`` per time unit late. Order IDs are also their travel-matrix
locations. ``travel`` is a complete nested mapping ``from -> to -> duration``.
All vehicles start at ``depot`` at time 0 and return to the depot.

The close time is a soft SLA deadline by default; an order may opt into a hard
close with ``hard_window: true``. Service starts at max(arrival, open);
lateness is max(0, service_start-close). Objective is travel duration plus
penalty * lateness. It is a transparent tiny-instance cost proxy, not a claim
that minutes equal monetary units; deployments should provide calibrated
travel/labor costs and contract penalties.

Public interfaces:
* validate_scenario(scenario) -> {valid: bool, errors: [field-level strings]}
* generate_routes(scenario) -> list of per-vehicle route dictionaries
* solve_exact(scenario) -> feasible plan or explicit infeasible result;
  raises SolveBudgetExceeded if optimality/infeasibility is not proven within
  the deterministic search-work budget
"""

from __future__ import annotations

from itertools import permutations
from math import factorial, isfinite
from typing import Any

MAX_EXACT_ORDERS = 8
MAX_VEHICLES = 16
MAX_ROUTE_CANDIDATES = 10_000
MAX_EXACT_SEARCH_WORK = 100_000


class SolveBudgetExceeded(RuntimeError):
    """Exact-cover search stopped before proving feasibility or infeasibility."""

    code = "solver_budget_exceeded"

    def __init__(self) -> None:
        super().__init__("exact dispatch search exceeded its deterministic work budget")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def validate_scenario(scenario: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(scenario, dict):
        return {"valid": False, "errors": ["scenario: expected an object"]}
    vehicles = scenario.get("vehicles")
    orders = scenario.get("orders")
    travel = scenario.get("travel")
    depot = scenario.get("depot")
    if not isinstance(vehicles, list) or not vehicles:
        errors.append("vehicles: expected a non-empty list")
        vehicles = []
    elif len(vehicles) > MAX_VEHICLES:
        errors.append(f"vehicles: exact solver supports at most {MAX_VEHICLES}")
    if not isinstance(orders, list):
        errors.append("orders: expected a list")
        orders = []
    if len(orders) > MAX_EXACT_ORDERS:
        errors.append(f"orders: exact solver supports at most {MAX_EXACT_ORDERS}")
    for kind, rows in (("vehicles", vehicles), ("orders", orders)):
        seen: set[str] = set()
        for index, row in enumerate(rows):
            prefix = f"{kind}[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{prefix}: expected an object")
                continue
            ident = row.get("id")
            if not isinstance(ident, str) or not ident:
                errors.append(f"{prefix}.id: expected a non-empty string")
            elif ident in seen:
                errors.append(f"{prefix}.id: duplicate identifier {ident!r}")
            else:
                seen.add(ident)
            fields = ("capacity",) if kind == "vehicles" else ("demand", "service", "penalty")
            for field in fields:
                value = row.get(field)
                if not _number(value) or value < 0:
                    errors.append(f"{prefix}.{field}: expected a finite non-negative number")
            if kind == "orders":
                hard_window = row.get("hard_window", False)
                if not isinstance(hard_window, bool):
                    errors.append(f"{prefix}.hard_window: expected a boolean")
                window = row.get("window")
                if (not isinstance(window, list) or len(window) != 2 or
                        not all(_number(x) for x in window) or window[0] > window[1]):
                    errors.append(f"{prefix}.window: expected [open, close] finite times with open <= close")
    # Count actual capacity-feasible order subsets before route permutations.
    # This keeps the pre-enumeration bound useful for fleets whose vehicle
    # capacities naturally prevent large subsets from being routed together.
    if isinstance(vehicles, list) and isinstance(orders, list) and len(orders) <= MAX_EXACT_ORDERS:
        candidate_bound = 0
        demands = [o.get("demand") if isinstance(o, dict) else None for o in orders]
        for vehicle in vehicles:
            if not isinstance(vehicle, dict) or not _number(vehicle.get("capacity")):
                continue
            if not all(_number(demand) for demand in demands):
                continue
            for mask in range(1, 1 << len(demands)):
                subset_demand = sum(demands[index] for index in range(len(demands))
                                    if mask & (1 << index))
                if subset_demand <= vehicle["capacity"]:
                    candidate_bound += factorial(mask.bit_count())
        if candidate_bound > MAX_ROUTE_CANDIDATES:
            errors.append(f"vehicles: route candidate bound exceeds {MAX_ROUTE_CANDIDATES}")
    if not isinstance(depot, str) or not depot:
        errors.append("depot: expected a non-empty string")
    if not isinstance(travel, dict):
        errors.append("travel: expected a nested mapping")
    else:
        locations = {depot} if isinstance(depot, str) and depot else set()
        locations.update(row.get("id") for row in orders if isinstance(row, dict) and isinstance(row.get("id"), str))
        for origin in locations:
            row = travel.get(origin)
            if not isinstance(row, dict):
                errors.append(f"travel.{origin}: missing origin row")
                continue
            for destination in locations:
                value = row.get(destination)
                if not _number(value) or value < 0:
                    errors.append(f"travel.{origin}.{destination}: expected finite non-negative duration")
    return {"valid": not errors, "errors": errors}


def _route_metrics(scenario: dict[str, Any], vehicle: dict[str, Any], order_ids: tuple[str, ...]) -> dict[str, Any]:
    orders = {o["id"]: o for o in scenario["orders"]}
    travel = scenario["travel"]
    location = scenario["depot"]
    time = 0.0
    travel_cost = 0.0
    exposure = 0.0
    stops = []
    for order_id in order_ids:
        order = orders[order_id]
        leg = float(travel[location][order_id])
        travel_cost += leg
        arrival = time + leg
        service_start = max(arrival, float(order["window"][0]))
        late_by = max(0.0, service_start - float(order["window"][1]))
        penalty = late_by * float(order["penalty"])
        exposure += penalty
        departure = service_start + float(order["service"])
        stops.append({"order_id": order_id, "arrival": arrival, "service_start": service_start,
                      "departure": departure, "lateness": late_by, "penalty_exposure": penalty})
        location, time = order_id, departure
    back = float(travel[location][scenario["depot"]])
    travel_cost += back
    return {"vehicle_id": vehicle["id"], "order_ids": list(order_ids), "stops": stops,
            "return_time": time + back, "travel_cost": travel_cost,
            "lateness_exposure": exposure, "objective": travel_cost + exposure}


def generate_routes(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate every non-empty capacity-feasible ordered route per vehicle.

    Route enumeration is intentionally exact and bounded to eight orders.
    Soft windows remain eligible with lateness exposure. Routes violating an
    explicitly hard window are excluded. Fleet-level exclusivity/coverage is
    solved later.
    """
    result = validate_scenario(scenario)
    if not result["valid"]:
        raise ValueError("invalid scenario: " + "; ".join(result["errors"]))
    routes = []
    orders = scenario["orders"]
    orders_by_id = {order["id"]: order for order in orders}
    for vehicle in scenario["vehicles"]:
        eligible = [o for o in orders if o["demand"] <= vehicle["capacity"]]

        def feasible_orders(prefix: tuple[str, ...], remaining: tuple[dict[str, Any], ...],
                            slots: int, demand: float):
            if slots == 0:
                yield prefix
                return
            for index, order in enumerate(remaining):
                next_demand = demand + order["demand"]
                if next_demand <= vehicle["capacity"]:
                    yield from feasible_orders(prefix + (order["id"],),
                        remaining[:index] + remaining[index + 1:], slots - 1, next_demand)

        for size in range(1, len(eligible) + 1):
            # Track remaining capacity while emitting permutations. Infeasible
            # prefixes are pruned, and feasible candidates retain the stable
            # itertools permutation order used by prior solver/QUBO versions.
            for route_order in feasible_orders((), tuple(eligible), size, 0.0):
                candidate = _route_metrics(scenario, vehicle, route_order)
                if any(orders_by_id[stop["order_id"]].get("hard_window", False) and stop["lateness"] > 0
                       for stop in candidate["stops"]):
                    continue
                routes.append(candidate)
    return routes


def solve_exact(scenario: dict[str, Any]) -> dict[str, Any]:
    """Find the minimum-objective exact cover with at most one route/vehicle.

    The recursive search has a deterministic budget counting both search nodes
    and route-option visits. Exhaustion raises :class:`SolveBudgetExceeded`;
    an incumbent is never returned unless optimality has been proven.
    """
    checked = validate_scenario(scenario)
    if not checked["valid"]:
        raise ValueError("invalid scenario: " + "; ".join(checked["errors"]))
    if not scenario["orders"]:
        return {"status": "feasible", "routes": [], "objective": 0.0,
                "travel_cost": 0.0, "lateness_exposure": 0.0}
    routes = generate_routes(scenario)
    by_order: dict[str, list[dict[str, Any]]] = {o["id"]: [] for o in scenario["orders"]}
    for route in routes:
        for order_id in route["order_ids"]:
            by_order[order_id].append(route)
    for options in by_order.values():
        # Prefer routes that cover more work per unit cost. This only changes
        # incumbent discovery order; the branch-and-bound still proves the
        # exact minimum before returning.
        options.sort(key=lambda r: (r["objective"] / len(r["order_ids"]), r["objective"],
                                    r["vehicle_id"], tuple(r["order_ids"])))
    best: tuple[float, tuple[dict[str, Any], ...]] | None = None
    work = 0

    def tick() -> None:
        nonlocal work
        if work >= MAX_EXACT_SEARCH_WORK:
            raise SolveBudgetExceeded()
        work += 1

    def search(covered: frozenset[str], used: frozenset[str], chosen: tuple[dict[str, Any], ...], cost: float) -> None:
        nonlocal best
        tick()
        if len(covered) == len(by_order):
            if best is None or cost < best[0]:
                best = (cost, chosen)
            return
        if best is not None and cost >= best[0]:
            return
        remaining = [key for key in by_order if key not in covered]
        target = min(remaining, key=lambda key: sum(r["vehicle_id"] not in used and
                       not (set(r["order_ids"]) & covered) for r in by_order[key]))
        for route in by_order[target]:
            tick()
            route_orders = set(route["order_ids"])
            if route["vehicle_id"] in used or route_orders & covered:
                continue
            search(covered | route_orders, used | {route["vehicle_id"]}, chosen + (route,), cost + route["objective"])

    search(frozenset(), frozenset(), (), 0.0)
    if best is None:
        return {"status": "infeasible", "routes": [], "objective": None,
                "reason": "orders cannot be covered exactly once by distinct capacity-feasible vehicles"}
    chosen = sorted(best[1], key=lambda r: r["vehicle_id"])
    return {"status": "feasible", "routes": chosen, "objective": sum(r["objective"] for r in chosen),
            "travel_cost": sum(r["travel_cost"] for r in chosen),
            "lateness_exposure": sum(r["lateness_exposure"] for r in chosen)}
