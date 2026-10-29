"""Exact-cover selection with explicitly declared monetary input units."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

from .domain import generate_routes, validate_scenario
from .validator import validate_plan

MAX_FINANCIAL_ORDERS = 6
MAX_FINANCIAL_VEHICLES = 8
MAX_FINANCIAL_ROUTE_BOUND = 3_000
MAX_FINANCIAL_WORK = 5_000
PENALTY_RATE_UNIT = "currency_per_late_minute"


class FinancialError(ValueError):
    """Stable machine-readable financial scoring error."""

    def __init__(self, code: str, message: str, field: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.field = field

    def as_dict(self) -> dict[str, Any]:
        details = [{"field": self.field, "message": "invalid or unsupported value"}] if self.field else []
        return {"error": {"code": self.code, "message": str(self), "details": details}}


def solve_financial_dispatch(
    scenario: dict[str, Any],
    *,
    currency: str,
    operating_cost_per_travel_minute: float,
    penalty_rate_unit: str,
) -> dict[str, Any]:
    """Select an exact-cover plan using supplied travel cost and source penalties."""
    _validate_terms(currency, operating_cost_per_travel_minute, penalty_rate_unit)
    if not isinstance(scenario, dict) or set(scenario) != {"depot", "vehicles", "orders", "travel"}:
        raise FinancialError("invalid_scenario", "scenario must contain exactly depot, vehicles, orders, and travel", "scenario")
    checked = validate_scenario(scenario)
    if not checked["valid"]:
        raise FinancialError("invalid_scenario", "scenario failed domain validation", "scenario")
    if len(scenario["orders"]) > MAX_FINANCIAL_ORDERS or len(scenario["vehicles"]) > MAX_FINANCIAL_VEHICLES:
        raise FinancialError("unsupported_capability", "scenario exceeds financial exact-cover size bounds", "scenario")
    route_bound = _route_bound(scenario)
    if route_bound > MAX_FINANCIAL_ROUTE_BOUND:
        raise FinancialError("unsupported_capability", "scenario exceeds financial candidate-generation bound", "scenario")
    try:
        candidates = generate_routes(scenario)
    except Exception as exc:
        raise FinancialError("invalid_scenario", "financial route candidates could not be generated", "scenario") from exc
    if len(candidates) > MAX_FINANCIAL_ROUTE_BOUND:
        raise FinancialError("unsupported_capability", "scenario exceeds financial candidate-generation bound", "scenario")
    if not scenario["orders"]:
        plan = {"status": "feasible", "routes": [], "travel_cost": 0.0,
                "lateness_exposure": 0.0, "objective": 0.0}
        return _format_result(scenario, plan, currency, float(operating_cost_per_travel_minute), 0)
    if not candidates:
        raise FinancialError("infeasible", "no capacity-feasible routes cover the orders", "scenario")

    by_order = {row["id"]: [] for row in scenario["orders"]}
    route_cost: dict[tuple[str, tuple[str, ...]], float] = {}
    for route in candidates:
        key = (route["vehicle_id"], tuple(route["order_ids"]))
        route_cost[key] = (float(route["travel_cost"]) * float(operating_cost_per_travel_minute) +
                           float(route["lateness_exposure"]))
        for oid in route["order_ids"]:
            by_order[oid].append(route)
    best: tuple[float, tuple[dict[str, Any], ...]] | None = None
    work = 0

    def tick() -> None:
        nonlocal work
        work += 1
        if work > MAX_FINANCIAL_WORK:
            raise FinancialError("work_budget_exceeded", "financial exact-cover search exceeded its deterministic work budget")

    def search(covered: frozenset[str], used: frozenset[str], chosen: tuple[dict[str, Any], ...], cost: float) -> None:
        nonlocal best
        tick()
        if len(covered) == len(by_order):
            if best is None or cost < best[0]:
                best = cost, chosen
            return
        if best is not None and cost >= best[0]:
            return
        remaining = [oid for oid in by_order if oid not in covered]
        target = min(remaining, key=lambda oid: sum(
            route["vehicle_id"] not in used and not (set(route["order_ids"]) & covered)
            for route in by_order[oid]))
        for route in by_order[target]:
            tick()
            route_orders = set(route["order_ids"])
            if route["vehicle_id"] in used or route_orders & covered:
                continue
            key = (route["vehicle_id"], tuple(route["order_ids"]))
            search(covered | route_orders, used | {route["vehicle_id"]}, chosen + (route,), cost + route_cost[key])

    search(frozenset(), frozenset(), (), 0.0)
    if best is None:
        raise FinancialError("infeasible", "no exact-cover plan satisfies fleet and capacity constraints", "scenario")
    selected = sorted((deepcopy(route) for route in best[1]), key=lambda row: row["vehicle_id"])
    plan = {"status": "feasible", "routes": selected,
            "travel_cost": sum(float(route["travel_cost"]) for route in selected),
            "lateness_exposure": sum(float(route["lateness_exposure"]) for route in selected)}
    plan["objective"] = plan["travel_cost"] + plan["lateness_exposure"]
    return _format_result(scenario, plan, currency, float(operating_cost_per_travel_minute), work)


def validate_financial_plan(
    scenario: dict[str, Any],
    plan: dict[str, Any],
    reported_financial: dict[str, Any],
    *,
    currency: str,
    operating_cost_per_travel_minute: float,
    penalty_rate_unit: str,
) -> dict[str, Any]:
    """Recompute plan timing with the independent validator and money from raw inputs."""
    _validate_terms(currency, operating_cost_per_travel_minute, penalty_rate_unit)
    proxy_validation = validate_plan(scenario, plan)
    if not proxy_validation["valid"]:
        return {"valid": False, "errors": ["plan failed independent domain validation"],
                "proxy_validation": proxy_validation, "recomputed_financial": None}
    recomputed = _recompute_financial(scenario, plan, currency, float(operating_cost_per_travel_minute))
    errors = []
    if not isinstance(reported_financial, dict):
        errors.append("financial report is not an object")
    else:
        for key in ("currency", "penalty_rate_unit"):
            if reported_financial.get(key) != recomputed[key]:
                errors.append(f"{key} mismatch")
        for key in ("operating_cost_per_travel_minute", "travel_minutes", "transport_cost",
                    "late_penalty_cost", "total_cost"):
            actual = reported_financial.get(key)
            expected = recomputed[key]
            if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
                    not math.isfinite(float(actual)) or not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10)):
                errors.append(f"{key} mismatch")
    return {"valid": not errors, "errors": errors,
            "proxy_validation": proxy_validation, "recomputed_financial": recomputed}


def _format_result(scenario: dict[str, Any], plan: dict[str, Any], currency: str,
                   rate: float, work: int) -> dict[str, Any]:
    proxy_validation = validate_plan(scenario, plan)
    if not proxy_validation["valid"]:
        raise FinancialError("validation_failure", "selected plan failed independent domain validation")
    financial = _recompute_financial(scenario, plan, currency, rate)
    financial_validation = validate_financial_plan(
        scenario, plan, financial, currency=currency,
        operating_cost_per_travel_minute=rate, penalty_rate_unit=PENALTY_RATE_UNIT)
    if not financial_validation["valid"]:
        raise FinancialError("financial_validation_failure", "independent financial objective verification failed")
    return {
        "status": "feasible", "execution_type": "local_classical_exact_cover",
        "plan": plan,
        "proxy_validation": proxy_validation,
        "financial_objective": financial,
        "financial_validation": financial_validation,
        "assumptions": [
            "order penalty values are caller-declared currency per late minute",
            "operating cost is caller-supplied currency per travel minute",
            "objective sums transport cost and source penalty rates without unit conversion",
            "all calculations retain full binary floating-point precision; no rounding is applied",
            "scenario values are not calibrated by this module; no currency realism claim is made",
        ],
        "provenance": {"scenario_sha256": _hash(scenario), "plan_sha256": _hash(plan),
                       "currency": currency, "penalty_rate_unit": PENALTY_RATE_UNIT,
                       "route_candidates": sum(len(routes) for routes in _routes_by_vehicle(scenario).values()),
                       "search_work": work, "limits": {"orders": MAX_FINANCIAL_ORDERS,
                       "vehicles": MAX_FINANCIAL_VEHICLES, "route_bound": MAX_FINANCIAL_ROUTE_BOUND,
                       "search_work": MAX_FINANCIAL_WORK}},
    }


def _recompute_financial(scenario: dict[str, Any], plan: dict[str, Any], currency: str, rate: float) -> dict[str, Any]:
    """Compute monetary totals from scenario matrices and order source fields."""
    orders = {row["id"]: row for row in scenario["orders"]}
    travel_minutes = 0.0
    penalty_cost = 0.0
    for route in plan["routes"]:
        location = scenario["depot"]
        clock = 0.0
        for oid in route["order_ids"]:
            order = orders[oid]
            leg = float(scenario["travel"][location][oid])
            travel_minutes += leg
            arrival = clock + leg
            start = max(arrival, float(order["window"][0]))
            lateness = max(0.0, start - float(order["window"][1]))
            penalty_cost += lateness * float(order["penalty"])
            clock = start + float(order["service"])
            location = oid
        travel_minutes += float(scenario["travel"][location][scenario["depot"]])
    transport_cost = travel_minutes * rate
    return {"currency": currency, "operating_cost_per_travel_minute": rate,
            "penalty_rate_unit": PENALTY_RATE_UNIT, "travel_minutes": travel_minutes,
            "transport_cost": transport_cost, "late_penalty_cost": penalty_cost,
            "total_cost": transport_cost + penalty_cost}


def _validate_terms(currency: Any, rate: Any, penalty_rate_unit: Any) -> None:
    if not isinstance(currency, str) or re.fullmatch(r"[A-Z]{3}", currency, flags=re.ASCII) is None:
        raise FinancialError("invalid_currency", "currency must be an uppercase three-letter ISO-style code", "currency")
    if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or
            not math.isfinite(rate) or rate < 0):
        raise FinancialError("invalid_operating_rate", "operating cost per travel minute must be finite and non-negative",
                            "operating_cost_per_travel_minute")
    if penalty_rate_unit != PENALTY_RATE_UNIT:
        raise FinancialError("invalid_penalty_rate_unit", f"penalty_rate_unit must be {PENALTY_RATE_UNIT!r}",
                            "penalty_rate_unit")


def _route_bound(scenario: dict[str, Any]) -> int:
    total = 0
    for vehicle in scenario["vehicles"]:
        eligible = [order for order in scenario["orders"] if order["demand"] <= vehicle["capacity"]]
        n = len(eligible)
        term = 1
        for k in range(1, n + 1):
            term *= n - k + 1
            total += term
    return total


def _routes_by_vehicle(scenario: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    routes = generate_routes(scenario)
    result = {row["id"]: [] for row in scenario["vehicles"]}
    for route in routes:
        result[route["vehicle_id"]].append(route)
    return result


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
