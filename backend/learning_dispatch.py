"""Decision-focused VQC over exact tiny dispatch candidates.

The VQC predicts route-level SLA-event probability from route travel cost and
an observed scenario congestion multiplier. It then scores exact-cover fleet
plans by travel cost plus model-estimated one-minute penalty exposure. Every
selected plan is independently checked by ``validator.validate_plan`` against
the scenario inputs. Domain costs retain the documented travel-minutes plus
penalty-units proxy; production monetary calibration is outside this fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

from .domain import generate_routes, solve_exact
from .quantum_learning import (
    Example,
    _fit_classical,
    _probability_gradient,
    _sigmoid,
    _loss_gradient,
    risk_probability,
)
from .validator import validate_plan


MULTIPLIERS: tuple[float, ...] = (
    0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.50, 2.00,
    2.50, 3.00, 0.90, 1.70, 3.50, 2.80, 2.30, 1.00,
)
SPLITS = {"train": range(0, 10), "validation": range(10, 12), "test": range(12, 16)}
FEATURE_MAX_TRAVEL = 20.0
SOFT_PLAN_TEMPERATURE = 2.0
RISK_HORIZON_MINUTES = 1.0


@dataclass(frozen=True)
class RouteRow:
    features: tuple[float, float]
    label: int
    split: str
    index: int
    scenario_id: str
    multiplier: float
    route_id: str
    travel_cost: float
    exposure_scale: float


@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    split: str
    multiplier: float
    scenario: dict[str, Any]
    data_hash: str
    route_rows: tuple[RouteRow, ...]
    feasible_plans: tuple[dict[str, Any], ...]


def _scale_scenario(base: dict[str, Any], multiplier: float) -> dict[str, Any]:
    scenario = json.loads(json.dumps(base))
    for origin, edges in scenario["travel"].items():
        for destination, value in edges.items():
            if origin != destination:
                edges[destination] = round(float(value) * multiplier, 6)
    return scenario


def _route_id(route: dict[str, Any]) -> str:
    return route["vehicle_id"] + ":" + ">".join(route["order_ids"])


def _exposure_scale(scenario: dict[str, Any], route: dict[str, Any]) -> float:
    order_map = {row["id"]: row for row in scenario["orders"]}
    return RISK_HORIZON_MINUTES * sum(float(order_map[oid]["penalty"]) for oid in route["order_ids"])


def _feasible_plans(scenario: dict[str, Any], routes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enumerate exact covers and retain only plans passing independent checks."""
    order_ids = {row["id"] for row in scenario["orders"]}
    by_order = {oid: [] for oid in sorted(order_ids)}
    for route in routes:
        for oid in route["order_ids"]:
            by_order[oid].append(route)
    results: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()

    def visit(covered: frozenset[str], used: frozenset[str], chosen: tuple[dict[str, Any], ...]) -> None:
        if covered == order_ids:
            canonical = tuple(sorted((r["vehicle_id"], tuple(r["order_ids"])) for r in chosen))
            if canonical in seen:
                return
            seen.add(canonical)
            sorted_routes = sorted(chosen, key=lambda r: r["vehicle_id"])
            plan = {
                "status": "feasible",
                "routes": sorted_routes,
                "travel_cost": sum(float(r["travel_cost"]) for r in sorted_routes),
                "lateness_exposure": sum(float(r["lateness_exposure"]) for r in sorted_routes),
                "objective": sum(float(r["objective"]) for r in sorted_routes),
            }
            independent = validate_plan(scenario, plan)
            if independent["valid"] and independent["feasible"]:
                plan["validated_objective"] = float(independent["recomputed"]["objective"])
                plan["validation"] = independent
                results.append(plan)
            return
        remaining = sorted(oid for oid in order_ids if oid not in covered)
        target = min(remaining, key=lambda oid: sum(
            route["vehicle_id"] not in used and not (set(route["order_ids"]) & covered)
            for route in by_order[oid]
        ))
        for route in sorted(by_order[target], key=lambda r: (r["vehicle_id"], tuple(r["order_ids"]))):
            route_orders = set(route["order_ids"])
            if route["vehicle_id"] in used or route_orders & covered:
                continue
            visit(covered | route_orders, used | {route["vehicle_id"]}, chosen + (route,))

    visit(frozenset(), frozenset(), ())
    return results


def _nominal_pressure(route: dict[str, Any], scenario: dict[str, Any]) -> float:
    """Encode contractual tightness from nominal schedule inputs, not held-out outcomes."""
    orders = {row["id"]: row for row in scenario["orders"]}
    slacks = [float(orders[stop["order_id"]]["window"][1]) - float(stop["service_start"])
              for stop in route["stops"]]
    tightest_slack = min(slacks, default=4.0)
    return 1.0 - min(4.0, max(0.0, tightest_slack)) / 4.0


def build_scenarios(base: dict[str, Any]) -> list[ScenarioRecord]:
    nominal_routes = {_route_id(route): route for route in generate_routes(base)}
    records = []
    for index, multiplier in enumerate(MULTIPLIERS):
        split = next(name for name, ids in SPLITS.items() if index in ids)
        scenario_id = f"traffic-{index:02d}"
        scenario = _scale_scenario(base, multiplier)
        routes = generate_routes(scenario)
        if not routes:
            raise ValueError(f"{scenario_id}: no capacity-feasible route candidates")
        route_rows = []
        for route in routes:
            scale = _exposure_scale(scenario, route)
            nominal_route = nominal_routes.get(_route_id(route))
            if nominal_route is None:
                raise ValueError(f"candidate route {_route_id(route)!r} has no nominal feature reference")
            features = (min(1.0, float(route["travel_cost"]) / FEATURE_MAX_TRAVEL),
                        _nominal_pressure(nominal_route, base))
            label = int(float(route["lateness_exposure"]) > 0.0)
            route_rows.append(RouteRow(
                features=features,
                label=label,
                split=split,
                index=len(route_rows),
                scenario_id=scenario_id,
                multiplier=multiplier,
                route_id=_route_id(route),
                travel_cost=float(route["travel_cost"]),
                exposure_scale=scale,
            ))
        plans = _feasible_plans(scenario, routes)
        if not plans:
            raise ValueError(f"{scenario_id}: no independently validated feasible exact-cover plans")
        digest = hashlib.sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        records.append(ScenarioRecord(scenario_id, split, multiplier, scenario, digest, tuple(route_rows), tuple(plans)))
    return records


def _probabilities(rows: Sequence[RouteRow], theta: Sequence[float]) -> list[float]:
    return [risk_probability(row.features, theta) for row in rows]


def _plan_estimated_cost(plan: dict[str, Any], rows_by_id: dict[str, RouteRow], probabilities: dict[str, float]) -> float:
    return sum(
        float(route["travel_cost"]) + probabilities[_route_id(route)] * rows_by_id[_route_id(route)].exposure_scale
        for route in plan["routes"]
    )


def _decision_objective_and_gradient(theta: Sequence[float], scenarios: Sequence[ScenarioRecord]) -> tuple[float, list[float]]:
    total_loss = 0.0
    total_gradient = [0.0, 0.0]
    if not scenarios:
        raise ValueError("decision objective requires at least one scenario")
    for record in scenarios:
        rows_by_id = {row.route_id: row for row in record.route_rows}
        ps = {row.route_id: risk_probability(row.features, theta) for row in record.route_rows}
        costs = [_plan_estimated_cost(plan, rows_by_id, ps) for plan in record.feasible_plans]
        actual = [float(plan["validated_objective"]) for plan in record.feasible_plans]
        baseline = min(costs)
        raw_weights = [math.exp(-(cost - baseline) / SOFT_PLAN_TEMPERATURE) for cost in costs]
        normalizer = sum(raw_weights)
        weights = [value / normalizer for value in raw_weights]
        expected_actual = sum(w * value for w, value in zip(weights, actual))
        total_loss += expected_actual
        # d E[actual]/d estimated-plan-cost = -w*(actual-E[actual])/T.
        dloss_dcost = [-(w * (value - expected_actual)) / SOFT_PLAN_TEMPERATURE for w, value in zip(weights, actual)]
        for row in record.route_rows:
            dloss_dp = 0.0
            for plan, coefficient in zip(record.feasible_plans, dloss_dcost):
                if any(_route_id(route) == row.route_id for route in plan["routes"]):
                    dloss_dp += coefficient * row.exposure_scale
            dp = _probability_gradient(row.features, theta)
            for k in range(2):
                total_gradient[k] += dloss_dp * dp[k]
    n = len(scenarios)
    return total_loss / n, [value / n for value in total_gradient]


def finite_difference_decision_gradient(theta: Sequence[float], scenarios: Sequence[ScenarioRecord], h: float = 1e-5) -> list[float]:
    if h <= 0 or not math.isfinite(h):
        raise ValueError("h must be finite and positive")
    result = []
    for k in range(2):
        plus, minus = list(theta), list(theta)
        plus[k] += h
        minus[k] -= h
        result.append((_decision_objective_and_gradient(plus, scenarios)[0] -
                       _decision_objective_and_gradient(minus, scenarios)[0]) / (2 * h))
    return result


def _fit_decision_vqc(scenarios: Sequence[ScenarioRecord], steps: int = 120, rate: float = 0.025) -> tuple[list[float], list[float]]:
    theta = [0.31, -0.47]
    history = []
    for _ in range(steps):
        loss, gradient = _decision_objective_and_gradient(theta, scenarios)
        history.append(loss)
        theta = [theta[k] - rate * gradient[k] for k in range(2)]
    history.append(_decision_objective_and_gradient(theta, scenarios)[0])
    return theta, history


def _fit_predictive_vqc(rows: Sequence[RouteRow], steps: int = 120) -> list[float]:
    generic = [Example(row.features, row.label, row.split, row.index) for row in rows]
    theta = [0.31, -0.47]
    for _ in range(steps):
        _, gradient = _loss_gradient(theta, generic, "predictive")
        theta = [theta[k] - 0.08 * gradient[k] for k in range(2)]
    return theta


def _fit_classical_rows(rows: Sequence[RouteRow]) -> list[float]:
    generic = [Example(row.features, row.label, row.split, row.index) for row in rows]
    return _fit_classical(generic)


def _classical_probability(features: Sequence[float], weights: Sequence[float]) -> float:
    return _sigmoid(weights[0] + weights[1] * features[0] + weights[2] * features[1])


def _select_plan(record: ScenarioRecord, model_kind: str, model: Sequence[float]) -> dict[str, Any]:
    rows_by_id = {row.route_id: row for row in record.route_rows}
    if model_kind == "classical":
        probabilities = {row.route_id: _classical_probability(row.features, model) for row in record.route_rows}
    else:
        probabilities = {row.route_id: risk_probability(row.features, model) for row in record.route_rows}
    scored = [(_plan_estimated_cost(plan, rows_by_id, probabilities), plan) for plan in record.feasible_plans]
    predicted_cost, chosen = min(scored, key=lambda item: (item[0], tuple(
        (r["vehicle_id"], tuple(r["order_ids"])) for r in item[1]["routes"]
    )))
    plan = {key: value for key, value in chosen.items() if key not in ("validation", "validated_objective")}
    independent = validate_plan(record.scenario, plan)
    selected_routes = []
    predicted_exposure = 0.0
    for route in plan["routes"]:
        route_id = _route_id(route)
        p = probabilities[route_id]
        expected_exposure = p * rows_by_id[route_id].exposure_scale
        predicted_exposure += expected_exposure
        selected_routes.append({"route_id": route_id, "feature_values": list(rows_by_id[route_id].features),
                                "travel_cost_input": rows_by_id[route_id].travel_cost,
                                "one_minute_exposure_scale": rows_by_id[route_id].exposure_scale,
                                "risk_probability": p,
                                "estimated_exposure": expected_exposure,
                                "validator_realized_exposure": route["lateness_exposure"],
                                "validator_realized_objective": route["objective"]})
    if not independent["valid"]:
        raise RuntimeError(f"selected plan failed independent validation: {independent['errors']}")
    return {
        "scenario_id": record.scenario_id,
        "multiplier": record.multiplier,
        "selected_routes": selected_routes,
        "estimated_plan_cost": predicted_cost,
        "estimated_exposure": predicted_exposure,
        "raw_plan": plan,
        "served_order_count": len({oid for route in plan["routes"] for oid in route["order_ids"]}),
        "required_order_count": len(record.scenario["orders"]),
        "validation": independent,
    }


def run_dispatch_study(base_scenario: dict[str, Any], steps: int = 120) -> dict[str, Any]:
    if not 1 <= steps <= 2000:
        raise ValueError("steps must be between 1 and 2000")
    records = build_scenarios(base_scenario)
    train_records = [record for record in records if record.split == "train"]
    train_rows = [row for record in train_records for row in record.route_rows]
    theta_decision, decision_history = _fit_decision_vqc(train_records, steps=steps)
    theta_predictive = _fit_predictive_vqc(train_rows, steps=steps)
    logistic = _fit_classical_rows(train_rows)
    models: dict[str, tuple[str, Sequence[float]]] = {
        "decision_focused_vqc": ("vqc", theta_decision),
        "predictive_only_vqc": ("vqc", theta_predictive),
        "classical_logistic": ("classical", logistic),
    }
    model_rows = {name: [] for name in models}
    exact_rows = []
    test_records = [record for record in records if record.split == "test"]
    for record in test_records:
        exact = solve_exact(record.scenario)
        exact_validation = validate_plan(record.scenario, exact)
        if not exact_validation["valid"]:
            raise RuntimeError(f"exact reference failed independent validation: {exact_validation['errors']}")
        exact_rows.append({"scenario_id": record.scenario_id, "objective": exact_validation["recomputed"]["objective"],
                           "validation": exact_validation})
        for name, (kind, model) in models.items():
            model_rows[name].append(_select_plan(record, kind, model))
    reference = {row["scenario_id"]: float(row["objective"]) for row in exact_rows}
    metrics = {}
    for name, outcomes in model_rows.items():
        metrics[name] = {
            "n_test_scenarios": len(outcomes),
            "feasibility_rate": sum(bool(row["validation"]["feasible"]) for row in outcomes) / len(outcomes),
            "coverage_rate": sum(row["served_order_count"] == row["required_order_count"] and
                                  row["validation"]["valid"] for row in outcomes) / len(outcomes),
            "mean_validator_objective": sum(row["validation"]["recomputed"]["objective"] for row in outcomes) / len(outcomes),
            "mean_validator_travel_cost": sum(row["validation"]["recomputed"]["travel_cost"] for row in outcomes) / len(outcomes),
            "mean_validator_lateness_exposure": sum(row["validation"]["recomputed"]["lateness_exposure"] for row in outcomes) / len(outcomes),
            "mean_model_estimated_exposure": sum(row["estimated_exposure"] for row in outcomes) / len(outcomes),
            "mean_regret_vs_exact_reference": sum(
                row["validation"]["recomputed"]["objective"] - reference[row["scenario_id"]] for row in outcomes
            ) / len(outcomes),
        }
    split_meta = {}
    for name, indexes in SPLITS.items():
        selected = [record for i, record in enumerate(records) if i in indexes]
        split_meta[name] = [{"scenario_id": record.scenario_id, "multiplier": record.multiplier,
                             "scenario_sha256": record.data_hash,
                             "route_rows": len(record.route_rows), "feasible_plans": len(record.feasible_plans),
                             "route_examples": [{"route_id": row.route_id, "features": list(row.features),
                                                 "sla_event_label": row.label, "travel_cost_input": row.travel_cost,
                                                 "one_minute_exposure_scale": row.exposure_scale}
                                                for row in record.route_rows]}
                            for record in selected]
    grad_records = train_records[:min(2, len(train_records))]
    analytic = _decision_objective_and_gradient([0.31, -0.47], grad_records)[1]
    numeric = finite_difference_decision_gradient([0.31, -0.47], grad_records)
    base_hash = hashlib.sha256(json.dumps(base_scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "evidence_label": "exploratory synthetic-scaled conditional dispatch; local exact CPU statevector; no QCloud/QPU",
        "evaluation_status": "exploratory, not blind: integration scenarios were inspected during feature/scenario iteration; do not treat test metrics as prospective evidence",
        "information_timing_limit": "Each scenario travel matrix is exposed to feature construction and then reused as validator ground truth. This measures conditional plan selection given scenario travel times; it does not test prospective delay-risk forecasting from pre-trip data.",
        "source_scenario": {"name": "fixtures/dispatch_tiny.json", "canonical_json_sha256": base_hash,
                            "transform": "travel durations scaled by frozen traffic multiplier; vehicle, order, capacity, service, deadline, and penalty inputs held fixed"},
        "objective_contract": "travel duration + penalty_rate * lateness minutes; travel duration is the domain's proxy cost, not calibrated currency",
        "feature_contract": {"x0": "candidate route travel_cost in the current scenario / 20, clipped to [0,1]", "x1": "nominal route window pressure = 1 - clipped minimum nominal stop slack / 4", "label": "1 iff independently recomputed candidate route lateness_exposure > 0"},
        "decision_contract": "enumerate capacity-feasible candidate routes; enumerate exact-cover plans with one route per vehicle; score by sum(travel_cost + VQC risk_probability * one_minute_penalty_rate_sum); choose minimum; validator recomputes selected plan from scenario inputs",
        "scenario_design": {"frozen_multipliers": list(MULTIPLIERS),
                            "split_ranges_by_index": {key: [min(ids), max(ids)] for key, ids in SPLITS.items()},
                            "common_test_scenarios_for_all_models": [row["scenario_id"] for row in split_meta["test"]]},
        "split": split_meta,
        "model_training": {"train_scenarios": len(train_records), "route_rows": len(train_rows), "steps": steps,
                           "decision_initial_soft_plan_loss": decision_history[0], "decision_final_soft_plan_loss": decision_history[-1],
                           "decision_theta": theta_decision, "predictive_theta": theta_predictive,
                           "classical_logistic_weights": logistic,
                           "decision_gradient_analytic": analytic, "decision_gradient_finite_difference_h_1e-5": numeric,
                           "decision_gradient_max_abs_error": max(abs(a-b) for a, b in zip(analytic, numeric)),
                           "gradient_check_train_scenarios": len(grad_records)},
        "test_metrics": metrics,
        "test_outcomes": model_rows,
        "exact_reference": exact_rows,
        "resources": {"qubits": 2, "simulator": "exact statevector", "shots": 0,
                      "scenario_count": len(records), "test_scenario_count": len(test_records),
                      "max_orders": max(len(record.scenario["orders"]) for record in records),
                      "optimizer_steps_per_model": steps},
    }
