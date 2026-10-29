"""Bounded single-request VQC inference; no training or realized outcomes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

from .domain import _route_metrics, generate_routes, validate_scenario
from .prospective_dispatch import _nominal_pressure
from .quantum_learning import risk_probability
from .validator import validate_plan


MAX_ORDERS = 3
MAX_VEHICLES = 4
MAX_ROUTE_CANDIDATES = 20
MAX_EXACT_PLANS = 500
MAX_ABS_THETA = 10.0
FEATURE_MAX_TRAVEL = 30.0
PENALTY_HORIZON_MINUTES = 1.0
MODEL_ARCHITECTURE = "debugleaf.vqc.dispatch.two-qubit.ry-cnot-z0.v1"
SURROGATE_ID = "soft-risk-adjusted-exact-cover.v1"
EXECUTION_TYPE = "local_cpu_exact_statevector"
_FINANCIAL_TERMS_MISSING = object()


class InferenceError(ValueError):
    """Stable public inference error with safe machine-readable code/details."""

    def __init__(self, code: str, message: str, details: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


@dataclass(frozen=True)
class _Candidate:
    route: dict[str, Any]
    route_id: str
    features: tuple[float, float]
    risk_probability: float
    estimated_exposure: float
    estimated_cost: float


def _route_id(route: dict[str, Any]) -> str:
    return str(route["vehicle_id"]) + ":" + ">".join(route["order_ids"])


def _scenario_check(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceError("invalid_scenario", f"{field} must be an object", [{"field": field, "message": "expected object"}])
    if set(value) != {"depot", "vehicles", "orders", "travel"}:
        raise InferenceError("invalid_scenario", f"{field} has missing or unsupported fields", [{"field": field, "message": "unsupported or missing fields"}])
    allowed_vehicle = {"id", "capacity"}
    required_order = {"id", "demand", "window", "service", "penalty"}
    allowed_order = required_order | {"hard_window"}
    if isinstance(value.get("vehicles"), list):
        if any(not isinstance(row, dict) or set(row) != allowed_vehicle for row in value["vehicles"]):
            raise InferenceError("invalid_scenario", f"{field} has invalid vehicle fields", [{"field": "vehicles", "message": "unsupported or missing fields"}])
    if isinstance(value.get("orders"), list):
        if any(not isinstance(row, dict) or not required_order.issubset(row) or set(row) - allowed_order
               for row in value["orders"]):
            raise InferenceError("invalid_scenario", f"{field} has invalid order fields", [{"field": "orders", "message": "unsupported or missing fields"}])
    checked = validate_scenario(value)
    if not checked["valid"]:
        fields = sorted({message.split(":", 1)[0] for message in checked["errors"]})
        raise InferenceError("invalid_scenario", f"{field} failed validation",
                             [{"field": key, "message": "invalid or missing value"} for key in fields])
    return value


def _check_same_problem(nominal: dict[str, Any], forecast: dict[str, Any]) -> None:
    if nominal["depot"] != forecast["depot"]:
        raise InferenceError("invalid_scenario", "nominal and forecast scenarios must describe the same dispatch problem",
                             [{"field": "depot", "message": "mismatch"}])
    if nominal["vehicles"] != forecast["vehicles"]:
        raise InferenceError("invalid_scenario", "nominal and forecast scenarios must describe the same dispatch problem",
                             [{"field": "vehicles", "message": "mismatch"}])
    def core_orders(scenario: dict[str, Any]) -> list[dict[str, Any]]:
        return [{key: row[key] for key in ("id", "demand", "window", "service", "penalty")}
                for row in scenario["orders"]]
    if core_orders(nominal) != core_orders(forecast):
        raise InferenceError("invalid_scenario", "nominal and forecast scenarios must describe the same dispatch problem",
                             [{"field": "orders", "message": "mismatch"}])
    for index, (nominal_order, forecast_order) in enumerate(zip(nominal["orders"], forecast["orders"])):
        if nominal_order.get("hard_window", False) != forecast_order.get("hard_window", False):
            raise InferenceError("invalid_scenario", "nominal and forecast hard-window flags must match",
                                 [{"field": f"orders[{index}].hard_window", "message": "mismatch"}])


def _validate_theta(theta: Any) -> tuple[float, float]:
    if not isinstance(theta, (list, tuple)) or len(theta) != 2:
        raise InferenceError("invalid_model_parameters", "theta must contain exactly two finite angles",
                             [{"field": "theta", "message": "expected two finite numbers"}])
    values = []
    for index, value in enumerate(theta):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > MAX_ABS_THETA:
            raise InferenceError("invalid_model_parameters", "theta values exceed supported bounds",
                                 [{"field": f"theta[{index}]", "message": "expected a finite number within supported bounds"}])
        values.append(float(value))
    return values[0], values[1]


def _enumerate_plans(routes: Sequence[dict[str, Any]], order_ids: set[str]) -> list[tuple[dict[str, Any], ...]]:
    by_order = {order_id: [] for order_id in sorted(order_ids)}
    for route in routes:
        for order_id in route["order_ids"]:
            by_order[order_id].append(route)
    plans: list[tuple[dict[str, Any], ...]] = []

    def visit(covered: frozenset[str], used_vehicles: frozenset[str], selected: tuple[dict[str, Any], ...]) -> None:
        if covered == order_ids:
            plans.append(tuple(sorted(selected, key=lambda route: route["vehicle_id"])))
            if len(plans) > MAX_EXACT_PLANS:
                raise InferenceError("unsupported_capability", "scenario exceeds the exact plan enumeration bound",
                                     [{"field": "scenario", "message": "too many feasible plans"}])
            return
        remaining = sorted(order_id for order_id in order_ids if order_id not in covered)
        target = min(remaining, key=lambda order_id: sum(
            route["vehicle_id"] not in used_vehicles and not (set(route["order_ids"]) & covered)
            for route in by_order[order_id]
        ))
        for route in sorted(by_order[target], key=lambda item: (item["vehicle_id"], tuple(item["order_ids"]))):
            route_orders = set(route["order_ids"])
            if route["vehicle_id"] in used_vehicles or route_orders & covered:
                continue
            visit(covered | route_orders, used_vehicles | {route["vehicle_id"]}, selected + (route,))

    visit(frozenset(), frozenset(), ())
    return plans


def predict_dispatch(
    forecast_scenario: dict[str, Any],
    nominal_scenario: dict[str, Any],
    theta: Sequence[float],
    *,
    model_version: str,
    execution_backend: str = "reference",
    financial_terms: Any = _FINANCIAL_TERMS_MISSING,
) -> dict[str, Any]:
    """Choose and validate one tiny dispatch plan from caller-supplied frozen VQC angles.

    This API is synchronous, deterministic, CPU-only, and bounded. Inputs contain
    only the nominal scenario, forecast travel matrix, model vector, and a caller
    label for the model version. No training or realized outcomes are accepted.
    """
    if not isinstance(model_version, str) or not model_version.strip() or len(model_version) > 128:
        raise InferenceError("invalid_model_parameters", "model_version must be a non-empty identifier",
                             [{"field": "model_version", "message": "expected 1 to 128 characters"}])
    frozen_theta = _validate_theta(theta)
    terms = _financial_terms(financial_terms)
    forecast = _scenario_check(forecast_scenario, "forecast_scenario")
    nominal = _scenario_check(nominal_scenario, "nominal_scenario")
    _check_same_problem(nominal, forecast)
    if len(forecast["orders"]) > MAX_ORDERS or len(forecast["vehicles"]) > MAX_VEHICLES:
        raise InferenceError("unsupported_capability", "scenario exceeds the bounded VQC inference size",
                             [{"field": "scenario", "message": "too many orders or vehicles"}])
    probability_fn, execution_type, execution_info = _execution_backend(execution_backend)
    if not forecast["orders"]:
        empty_plan = {"status": "feasible", "routes": [], "travel_cost": 0.0,
                      "lateness_exposure": 0.0, "objective": 0.0}
        empty_validation = validate_plan(forecast, empty_plan)
        response = {"status": "completed", "execution_type": execution_type,
                "model": _model_provenance(frozen_theta, model_version, execution_backend, execution_info),
                "plan": empty_plan,
                "estimated_plan_cost": 0.0, "estimated_exposure": 0.0,
                "route_predictions": [],
                "validation": empty_validation}
        if terms is not None:
            response.update(_financial_outputs(forecast, empty_plan, empty_validation, terms, 0.0, 0.0))
        return response
    try:
        routes = generate_routes(forecast)
    except Exception as exc:
        raise InferenceError("invalid_scenario", "route candidates could not be generated") from exc
    if len(routes) > MAX_ROUTE_CANDIDATES:
        raise InferenceError("unsupported_capability", "scenario exceeds the bounded route candidate limit",
                             [{"field": "scenario", "message": "too many route candidates"}])
    if not routes:
        raise InferenceError("infeasible", "no capacity-feasible candidate routes cover the orders")
    order_map = {order["id"]: order for order in forecast["orders"]}
    candidates: dict[str, _Candidate] = {}
    nominal_vehicles = {row["id"]: row for row in nominal["vehicles"]}
    for route in routes:
        route_id = _route_id(route)
        # The nominal route is a feature-only schedule. Do not apply hard-window
        # candidate filtering to it: forecast candidates and final plans are
        # filtered and independently checked below.
        nominal_route = _route_metrics(nominal, nominal_vehicles[route["vehicle_id"]],
                                       tuple(route["order_ids"]))
        features = (min(1.0, float(route["travel_cost"]) / FEATURE_MAX_TRAVEL),
                    _nominal_pressure(nominal_route, nominal))
        try:
            probability = probability_fn(features, frozen_theta)
        except Exception as exc:
            raise InferenceError("execution_failure", "selected local VQC execution backend failed") from exc
        severity = PENALTY_HORIZON_MINUTES * sum(float(order_map[order_id]["penalty"]) for order_id in route["order_ids"])
        transport_cost = (float(route["travel_cost"]) * terms["operating_cost_per_travel_minute"]
                          if terms is not None else float(route["travel_cost"]))
        candidates[route_id] = _Candidate(route, route_id, features, probability, probability * severity,
                                          transport_cost + probability * severity)
    plans = _enumerate_plans(routes, set(order_map))
    if not plans:
        raise InferenceError("infeasible", "no exact-cover plan satisfies fleet and capacity constraints")
    scored = []
    for plan_routes in plans:
        estimate = sum(candidates[_route_id(route)].estimated_cost for route in plan_routes)
        ids = tuple((route["vehicle_id"], tuple(route["order_ids"])) for route in plan_routes)
        scored.append((estimate, ids, plan_routes))
    estimated_plan_cost, _, selected_routes = min(scored, key=lambda item: (item[0], item[1]))
    plan = {"status": "feasible", "routes": list(selected_routes),
            "travel_cost": sum(float(route["travel_cost"]) for route in selected_routes),
            "lateness_exposure": sum(float(route["lateness_exposure"]) for route in selected_routes),
            "objective": sum(float(route["objective"]) for route in selected_routes)}
    validation = validate_plan(forecast, plan)
    if not validation["valid"] or not validation["feasible"]:
        raise InferenceError("validation_failure", "independent forecast-plan validation failed")
    route_predictions = []
    for route in selected_routes:
        candidate = candidates[_route_id(route)]
        route_predictions.append({"route_id": candidate.route_id,
                                  "features": list(candidate.features),
                                  "risk_probability": candidate.risk_probability,
                                  "estimated_exposure": candidate.estimated_exposure,
                                  "forecast_travel_cost": candidate.route["travel_cost"]})
        if terms is not None:
            route_predictions[-1]["estimated_transport_cost"] = (
                float(candidate.route["travel_cost"]) * terms["operating_cost_per_travel_minute"])
            route_predictions[-1]["estimated_route_cost"] = candidate.estimated_cost
    result = {"status": "completed", "execution_type": execution_type,
            "model": _model_provenance(frozen_theta, model_version, execution_backend, execution_info),
            "surrogate": {"id": SURROGATE_ID,
                          "estimated_route_cost": ("forecast travel minutes * caller rate + risk probability * sum(source penalty rates) * 1 minute"
                                                    if terms is not None else
                                                    "forecast travel duration + risk probability * sum(contract penalty rates) * 1 minute"),
                          "currency_calibration": ("caller-supplied rates; no calibration performed"
                                                   if terms is not None else "none; tiny-domain proxy units only")},
            "plan": plan,
            "estimated_plan_cost": estimated_plan_cost,
            "estimated_exposure": sum(row["estimated_exposure"] for row in route_predictions),
            "route_predictions": route_predictions,
            "validation": validation}
    if terms is not None:
        estimated_transport = sum(row["estimated_transport_cost"] for row in route_predictions)
        estimated_exposure = sum(row["estimated_exposure"] for row in route_predictions)
        result.update(_financial_outputs(forecast, plan, validation, terms,
                                        estimated_transport, estimated_exposure))
    return result


def _financial_terms(value: Any) -> dict[str, Any] | None:
    if value is _FINANCIAL_TERMS_MISSING:
        return None
    expected = {"currency", "operating_cost_per_travel_minute", "penalty_rate_unit"}
    if not isinstance(value, dict) or set(value) != expected:
        raise InferenceError("invalid_financial_terms", "financial_terms must contain currency, operating_cost_per_travel_minute, and penalty_rate_unit",
                             [{"field": "financial_terms", "message": "unsupported or missing fields"}])
    currency = value["currency"]
    rate = value["operating_cost_per_travel_minute"]
    unit = value["penalty_rate_unit"]
    if not isinstance(currency, str) or re.fullmatch(r"[A-Z]{3}", currency, flags=re.ASCII) is None:
        raise InferenceError("invalid_financial_terms", "currency must be an uppercase three-letter ISO-style code",
                             [{"field": "financial_terms.currency", "message": "invalid currency"}])
    if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or
            not math.isfinite(rate) or rate < 0):
        raise InferenceError("invalid_financial_terms", "operating cost per travel minute must be finite and non-negative",
                             [{"field": "financial_terms.operating_cost_per_travel_minute", "message": "invalid rate"}])
    if unit != "currency_per_late_minute":
        raise InferenceError("invalid_financial_terms", "penalty_rate_unit must be 'currency_per_late_minute'",
                             [{"field": "financial_terms.penalty_rate_unit", "message": "explicit unit declaration required"}])
    return {"currency": currency, "operating_cost_per_travel_minute": float(rate),
            "penalty_rate_unit": unit}


def _financial_outputs(forecast: dict[str, Any], plan: dict[str, Any], validation: dict[str, Any],
                       terms: dict[str, Any], estimated_transport: float,
                       estimated_exposure: float) -> dict[str, Any]:
    from .financial import validate_financial_plan

    forecast_recomputed = validation["recomputed"]
    reported = {
        "currency": terms["currency"],
        "operating_cost_per_travel_minute": terms["operating_cost_per_travel_minute"],
        "penalty_rate_unit": terms["penalty_rate_unit"],
        "travel_minutes": float(forecast_recomputed["travel_cost"]),
        "transport_cost": float(forecast_recomputed["travel_cost"]) * terms["operating_cost_per_travel_minute"],
        "late_penalty_cost": float(forecast_recomputed["lateness_exposure"]),
        "total_cost": (float(forecast_recomputed["travel_cost"]) * terms["operating_cost_per_travel_minute"] +
                       float(forecast_recomputed["lateness_exposure"])),
    }
    financial_validation = validate_financial_plan(
        forecast, plan, reported, currency=terms["currency"],
        operating_cost_per_travel_minute=terms["operating_cost_per_travel_minute"],
        penalty_rate_unit=terms["penalty_rate_unit"])
    if not financial_validation["valid"]:
        raise InferenceError("financial_validation_failure", "independent forecast financial validation failed")
    return {
        "financial_terms": dict(terms),
        "estimated_cost_currency": terms["currency"],
        "financial_estimate": {
            "currency": terms["currency"],
            "estimated_transport_cost": estimated_transport,
            "estimated_exposure": estimated_exposure,
            "estimated_total_cost": estimated_transport + estimated_exposure,
            "exposure_assumption": "one-minute expected lateness per route; multiplied by VQC probability and source penalty rates",
            "calibration": "uncalibrated caller rate and one-minute assumption",
        },
        "forecast_financial_objective": financial_validation["recomputed_financial"],
        "financial_validation": financial_validation,
        "model_suitability_warning": "training provenance is unverified; parameters are not known to have been trained for the supplied financial rates and are not calibrated",
    }


def _execution_backend(name: str):
    if name == "reference":
        return risk_probability, EXECUTION_TYPE, {"sdk": "not used"}
    if name != "qpiai_local":
        raise InferenceError("invalid_execution_backend", "execution_backend must be 'reference' or 'qpiai_local'",
                             [{"field": "execution_backend", "message": "unsupported value"}])
    from .qpiai_vqc import sdk_risk_probability, sdk_versions
    versions = sdk_versions()
    if versions.get("qpiai-quantum") == "unavailable":
        raise InferenceError("unsupported_capability", "QpiAI Quantum SDK is unavailable in this environment",
                             [{"field": "execution_backend", "message": "qpiai_local requires qpiai-quantum"}])
    return sdk_risk_probability, "local_qpiai_statevector", versions


def _model_provenance(theta: Sequence[float], model_version: str, execution_backend: str,
                      execution_info: dict[str, str]) -> dict[str, Any]:
    vector_digest = hashlib.sha256(json.dumps(list(theta), separators=(",", ":")).encode()).hexdigest()
    return {"version": model_version,
            "version_source": "caller-supplied identifier; not registry-verified",
            "parameter_source": "caller-supplied two-angle vector; training origin unverified",
            "parameter_vector_sha256": vector_digest,
            "architecture": MODEL_ARCHITECTURE,
            "qubits": 2,
            "feature_map": "RY(pi*x0) q0; RY(pi*x1) q1",
            "ansatz": "CNOT(0,1); RY(theta0) q0; RY(theta1) q1; CNOT(1,0)",
            "observable": "Z0; risk=sigmoid(-2*<Z0>)",
            "execution": ("exact local statevector, zero shots" if execution_backend == "reference"
                          else "QpiAI Quantum local StatevectorSimulator, zero shots"),
            "execution_backend": execution_backend,
            "execution_backend_versions": dict(execution_info),
            "training_provenance": "unverified caller-supplied vector",
            "calibration": "unverified; inference has no calibration metadata"}
