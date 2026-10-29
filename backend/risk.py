"""Scenario-based tiny-dispatch risk scoring using fixed, validated plans."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

from .domain import MAX_EXACT_ORDERS, MAX_VEHICLES, validate_scenario
from .validator import validate_plan

MAX_REALIZED_SCENARIOS = 100
_HARD_WINDOW_ERROR = re.compile(
    r"routes\[\d+\]\.stops\[\d+\]\.hard_window: service starts after hard close"
)


class RiskError(ValueError):
    """Stable machine-readable risk-scoring error."""

    def __init__(self, code: str, message: str, field: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.field = field

    def as_dict(self) -> dict[str, Any]:
        details = [{"field": self.field, "message": "invalid or unsupported value"}] if self.field else []
        return {"error": {"code": self.code, "message": str(self), "details": details}}


def evaluate_plan_risk(
    forecast_scenario: dict[str, Any],
    nominal_scenario: dict[str, Any],
    plan: dict[str, Any],
    realized_scenarios: list[dict[str, Any]],
    *,
    cvar_alpha: float | None = None,
) -> dict[str, Any]:
    """Replay one fixed plan over explicit alternative travel matrices.

    CVaR uses an equal-weight empirical distribution and a fractional worst-tail
    mean at confidence level ``cvar_alpha``. Both penalty exposure and total
    proxy objective are reported when requested.
    """
    forecast = _scenario(forecast_scenario, "forecast_scenario")
    nominal = _scenario(nominal_scenario, "nominal_scenario")
    if (forecast["depot"] != nominal["depot"] or forecast["vehicles"] != nominal["vehicles"] or
            forecast["orders"] != nominal["orders"]):
        raise RiskError("invalid_scenario", "forecast and nominal inputs must describe the same dispatch problem",
                        "nominal_scenario")
    if len(forecast["orders"]) > MAX_EXACT_ORDERS or len(forecast["vehicles"]) > MAX_VEHICLES:
        raise RiskError("unsupported_capability", "scenario exceeds the tiny-domain size bound", "forecast_scenario")
    original_validation = validate_plan(forecast, plan)
    if not original_validation["valid"]:
        raise RiskError("invalid_plan", "plan must pass independent validation under the forecast scenario", "plan")
    alpha = _alpha(cvar_alpha)
    if (not isinstance(realized_scenarios, list) or not realized_scenarios or
            len(realized_scenarios) > MAX_REALIZED_SCENARIOS):
        raise RiskError("invalid_scenarios", f"provide 1 to {MAX_REALIZED_SCENARIOS} realized scenarios",
                        "realized_scenarios")

    ids: set[str] = set()
    scenario_results: list[dict[str, Any]] = []
    total_travel: list[float] = []
    total_penalty: list[float] = []
    total_objective: list[float] = []
    late_scenarios = 0
    hard_window_violations = 0
    order_late_counts = {row["id"]: 0 for row in forecast["orders"]}
    stop_observations = 0
    late_stop_observations = 0

    for index, item in enumerate(realized_scenarios):
        field = f"realized_scenarios[{index}]"
        if not isinstance(item, dict) or set(item) != {"scenario_id", "travel"}:
            raise RiskError("invalid_scenarios", "each record must contain only scenario_id and travel", field)
        scenario_id = item["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id.strip() or len(scenario_id) > 128 or scenario_id in ids:
            raise RiskError("invalid_scenarios", "scenario IDs must be unique non-empty strings of at most 128 characters", field)
        ids.add(scenario_id)
        matrix = item["travel"]
        realized = deepcopy(forecast)
        realized["travel"] = matrix
        _scenario(realized, field)
        matrix_digest = _hash(matrix)
        replay = _replay_plan(realized, plan)
        if not _finite_plan(replay):
            raise RiskError("unsupported_capability", "scenario values overflow supported finite arithmetic", field)
        validation = validate_plan(realized, replay)
        hard_window_only = (
            not validation["valid"]
            and validation["recomputed"] is not None
            and bool(validation["errors"])
            and all(isinstance(error, str) and _HARD_WINDOW_ERROR.fullmatch(error)
                    for error in validation["errors"])
        )
        if not validation["valid"] and not hard_window_only:
            raise RiskError("validation_failure", "scenario replay failed independent plan validation", field)
        hard_window_violations += int(hard_window_only)
        totals = validation["recomputed"]
        is_late = any(stop["lateness"] > 0 for route in totals["routes"] for stop in route["stops"])
        late_scenarios += int(is_late)
        for route in totals["routes"]:
            for stop in route["stops"]:
                stop_observations += 1
                late = stop["lateness"] > 0
                late_stop_observations += int(late)
                order_late_counts[stop["order_id"]] += int(late)
        total_travel.append(float(totals["travel_cost"]))
        total_penalty.append(float(totals["lateness_exposure"]))
        total_objective.append(float(totals["objective"]))
        scenario_results.append({
            "scenario_id": scenario_id,
            "travel_matrix_sha256": matrix_digest,
            "late_event": is_late,
            "feasible": validation["valid"],
            "hard_window_violation": hard_window_only,
            "travel_cost": totals["travel_cost"],
            "penalty_exposure": totals["lateness_exposure"],
            "proxy_objective": totals["objective"],
            "routes": totals["routes"],
            "validation": validation,
        })

    count = len(scenario_results)
    result: dict[str, Any] = {
        "status": "completed",
        "execution_type": "local_deterministic_scenario_replay",
        "estimator": {"method": "equal-weight empirical mean and frequency",
                      "scenario_count": count,
                      "weight_per_scenario": 1.0 / count,
                      "confidence_intervals": "not computed; no population calibration claim"},
        "summary": {
            "empirical_probability_any_late": late_scenarios / count,
            "empirical_probability_hard_window_violation": hard_window_violations / count,
            "empirical_late_stop_frequency": late_stop_observations / stop_observations if stop_observations else 0.0,
            "per_order_late_frequency": {oid: value / count for oid, value in order_late_counts.items()},
            "expected_travel_cost": sum(total_travel) / count,
            "expected_penalty_exposure": sum(total_penalty) / count,
            "expected_proxy_objective": sum(total_objective) / count,
        },
        "input_validation": {"forecast_plan": original_validation},
        "scenarios": scenario_results,
        "provenance": {
            "forecast_scenario_sha256": _hash(forecast),
            "nominal_scenario_sha256": _hash(nominal),
            "plan_sha256": _hash(plan),
            "realized_scenario_set_sha256": _hash(realized_scenarios),
            "method": "fixed vehicle/order plan replayed from depot at time zero under each alternative travel matrix",
            "claims": "synthetic or user-supplied empirical scenario estimate only; no calibration or live-state claim",
        },
    }
    if alpha is not None:
        result["cvar"] = {
            "confidence_level": alpha,
            "tail_fraction": 1.0 - alpha,
            "sample_count": count,
            "system_wide_penalty_exposure": _fractional_tail_mean(total_penalty, alpha),
            "system_wide_proxy_objective": _fractional_tail_mean(total_objective, alpha),
            "definition": "equal-weight empirical mean of the worst (1-alpha) probability mass, including a fractional boundary scenario",
        }
    return result


def _scenario(value: Any, field: str) -> dict[str, Any]:
    expected = {"depot", "vehicles", "orders", "travel"}
    if not isinstance(value, dict) or set(value) != expected:
        raise RiskError("invalid_scenario", "scenario must contain exactly depot, vehicles, orders, and travel", field)
    checked = validate_scenario(value)
    if not checked["valid"]:
        raise RiskError("invalid_scenario", "scenario failed domain validation", field)
    return value


def _alpha(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise RiskError("invalid_cvar_alpha", "CVaR confidence must be finite and in [0, 1)", "cvar_alpha")
    return float(value)


def _replay_plan(scenario: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    orders = {row["id"]: row for row in scenario["orders"]}
    vehicle_ids = {row["id"] for row in scenario["vehicles"]}
    routes = []
    for submitted in plan["routes"]:
        vehicle_id = submitted["vehicle_id"]
        if vehicle_id not in vehicle_ids:
            raise RiskError("validation_failure", "plan vehicle is absent from realized scenario", "plan")
        location = scenario["depot"]
        clock = 0.0
        travel_cost = 0.0
        exposure = 0.0
        stops = []
        for order_id in submitted["order_ids"]:
            order = orders[order_id]
            leg = float(scenario["travel"][location][order_id])
            travel_cost += leg
            arrival = clock + leg
            start = max(arrival, float(order["window"][0]))
            lateness = max(0.0, start - float(order["window"][1]))
            penalty = lateness * float(order["penalty"])
            departure = start + float(order["service"])
            stops.append({"order_id": order_id, "arrival": arrival, "service_start": start,
                          "departure": departure, "lateness": lateness, "penalty_exposure": penalty})
            exposure += penalty
            location, clock = order_id, departure
        back = float(scenario["travel"][location][scenario["depot"]])
        travel_cost += back
        routes.append({"vehicle_id": vehicle_id, "order_ids": list(submitted["order_ids"]), "stops": stops,
                       "return_time": clock + back, "travel_cost": travel_cost,
                       "lateness_exposure": exposure, "objective": travel_cost + exposure})
    return {"status": "feasible", "routes": routes,
            "travel_cost": sum(route["travel_cost"] for route in routes),
            "lateness_exposure": sum(route["lateness_exposure"] for route in routes),
            "objective": sum(route["objective"] for route in routes)}


def _fractional_tail_mean(values: list[float], alpha: float) -> float:
    """Mean of exactly (1-alpha) empirical mass, including fractional boundary."""
    ordered = sorted((float(value) for value in values), reverse=True)
    tail_mass = (1.0 - alpha) * len(ordered)
    full = int(math.floor(tail_mass))
    fraction = tail_mass - full
    weighted = sum(ordered[:full])
    if fraction > 0:
        weighted += fraction * ordered[full]
    return weighted / tail_mass


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_plan(plan: dict[str, Any]) -> bool:
    fields = ("travel_cost", "lateness_exposure", "objective", "return_time")
    for route in plan["routes"]:
        if any(not math.isfinite(route[field]) for field in fields):
            return False
        for stop in route["stops"]:
            if any(not math.isfinite(stop[field]) for field in
                   ("arrival", "service_start", "departure", "lateness", "penalty_exposure")):
                return False
    return all(math.isfinite(plan[field]) for field in ("travel_cost", "lateness_exposure", "objective"))
