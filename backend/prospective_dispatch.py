"""Frozen-seed prospective simulation with forecast/realized data separation.

Model selection receives nominal inputs and noisy forecast travel only. A
separate realized travel matrix is kept for training supervision (train groups
only) and is opened for evaluation only after a plan has been selected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Sequence

from .domain import generate_routes, solve_exact
from .learning_dispatch import _exposure_scale, _feasible_plans, _nominal_pressure, _route_id
from .quantum_learning import Example, _fit_classical, _loss_gradient, _probability_gradient, _sigmoid, risk_probability
from .validator import validate_plan


SCENARIO_SEED = 20261028
FORECAST_NOISE_SEED = 20261029
SPLIT_SEED = 20261030
SCENARIO_COUNT = 24
SPLIT_COUNTS = {"train": 12, "validation": 6, "test": 6}
FEATURE_MAX_TRAVEL = 30.0
PLAN_TEMPERATURE = 2.0
PENALTY_HORIZON_MINUTES = 1.0
VQC_STEPS = 120


@dataclass(frozen=True)
class ProspectiveRoute:
    route_id: str
    features: tuple[float, float]
    train_label: int | None
    forecast_travel_cost: float
    exposure_scale: float


@dataclass(frozen=True)
class ProspectiveScenario:
    scenario_id: str
    split: str
    forecast_multiplier: float
    # `realized` is only accessed by train-label construction and post-selection evaluation.
    forecast: dict[str, Any]
    realized: dict[str, Any]
    data_hash: str
    routes: tuple[ProspectiveRoute, ...]
    plans: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ForecastDecisionInput:
    """Model-visible view; deliberately contains no realized matrix or labels."""
    scenario_id: str
    forecast_multiplier: float
    forecast: dict[str, Any]
    routes: tuple[ProspectiveRoute, ...]
    plans: tuple[dict[str, Any], ...]


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _clone_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(scenario))


def _travel_copy(base: dict[str, Any], multiplier: float, rng: random.Random, low: float, high: float) -> dict[str, Any]:
    result = _clone_scenario(base)
    for origin in sorted(result["travel"]):
        for destination in sorted(result["travel"][origin]):
            if origin != destination:
                jitter = rng.uniform(low, high)
                result["travel"][origin][destination] = round(
                    float(base["travel"][origin][destination]) * multiplier * jitter, 6
                )
    return result


def _scenario_specs() -> list[tuple[str, str, float, float]]:
    """Return frozen IDs, groups and forecast multipliers before any model fit."""
    latent_rng = random.Random(SCENARIO_SEED)
    forecast_rng = random.Random(FORECAST_NOISE_SEED)
    latent = [latent_rng.uniform(0.7, 2.5) for _ in range(SCENARIO_COUNT)]
    forecast = [min(3.0, max(0.5, value + forecast_rng.gauss(0.0, 0.30))) for value in latent]
    indices = list(range(SCENARIO_COUNT))
    random.Random(SPLIT_SEED).shuffle(indices)
    split_by_index: dict[int, str] = {}
    cursor = 0
    for split, count in SPLIT_COUNTS.items():
        for index in indices[cursor:cursor + count]:
            split_by_index[index] = split
        cursor += count
    return [(f"prospective-{i:02d}", split_by_index[i], forecast[i], latent[i]) for i in range(SCENARIO_COUNT)]


def build_prospective_scenarios(base: dict[str, Any]) -> list[ProspectiveScenario]:
    """Build grouped worlds; only train rows receive model-visible outcome labels."""
    forecast_noise_rng = random.Random(FORECAST_NOISE_SEED + 1000)
    realized_noise_rng = random.Random(SCENARIO_SEED + 1000)
    nominal_routes = {_route_id(route): route for route in generate_routes(base)}
    records = []
    for scenario_id, split, forecast_multiplier, latent_multiplier in _scenario_specs():
        forecast = _travel_copy(base, forecast_multiplier, forecast_noise_rng, 0.97, 1.03)
        realized = _travel_copy(base, latent_multiplier, realized_noise_rng, 0.90, 1.10)
        forecast_routes = generate_routes(forecast)
        realized_routes = ({_route_id(route): route for route in generate_routes(realized)}
                           if split == "train" else {})
        feasible_forecast_plans = _feasible_plans(forecast, forecast_routes)
        if not feasible_forecast_plans:
            raise ValueError(f"{scenario_id}: no forecast-feasible exact-cover plan")
        labels = {}
        if split == "train":
            # SLA occurrence is a constraint event, independent of whether
            # this fixture assigns a positive financial penalty to lateness.
            labels = {route_id: int(any(float(stop["lateness"]) > 0 for stop in route["stops"]))
                      for route_id, route in realized_routes.items()}
        route_rows = []
        for route in forecast_routes:
            route_id = _route_id(route)
            if route_id not in nominal_routes or (split == "train" and route_id not in realized_routes):
                raise ValueError(f"{scenario_id}: forecast/nominal route candidates do not align")
            pressure = _nominal_pressure(nominal_routes[route_id], base)
            features = (min(1.0, float(route["travel_cost"]) / FEATURE_MAX_TRAVEL), pressure)
            route_rows.append(ProspectiveRoute(
                route_id=route_id,
                features=features,
                train_label=labels.get(route_id),
                forecast_travel_cost=float(route["travel_cost"]),
                exposure_scale=_exposure_scale(base, route) * PENALTY_HORIZON_MINUTES,
            ))
        plans = []
        for forecast_plan in feasible_forecast_plans:
            # Keep forecast route sequence for selection; pair actual objective only for train supervision.
            route_ids = [_route_id(route) for route in forecast_plan["routes"]]
            item: dict[str, Any] = {"route_ids": route_ids, "realized_objective_train_only": None}
            if split == "train":
                actual_plan = _materialize_plan(route_ids, realized_routes)
                checked = validate_plan(realized, actual_plan)
                if not checked["valid"]:
                    raise RuntimeError(f"{scenario_id}: train plan failed realized validator: {checked['errors']}")
                item["realized_objective_train_only"] = float(checked["recomputed"]["objective"])
            plans.append(item)
        digest = _canonical_hash({"forecast": forecast, "realized": realized})
        records.append(ProspectiveScenario(scenario_id, split, forecast_multiplier,
                                           forecast, realized, digest, tuple(route_rows), tuple(plans)))
    return records


def _materialize_plan(route_ids: Sequence[str], actual_routes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    routes = [actual_routes[route_id] for route_id in route_ids]
    routes = sorted(routes, key=lambda route: route["vehicle_id"])
    return {"status": "feasible", "routes": routes,
            "travel_cost": sum(float(route["travel_cost"]) for route in routes),
            "lateness_exposure": sum(float(route["lateness_exposure"]) for route in routes),
            "objective": sum(float(route["objective"]) for route in routes)}


def _plan_estimated_cost(plan: dict[str, Any], rows: dict[str, ProspectiveRoute], probabilities: dict[str, float]) -> float:
    return sum(rows[route_id].forecast_travel_cost + probabilities[route_id] * rows[route_id].exposure_scale
               for route_id in plan["route_ids"])


def _decision_loss_gradient(theta: Sequence[float], scenarios: Sequence[ProspectiveScenario]) -> tuple[float, list[float]]:
    if not scenarios or any(record.split != "train" for record in scenarios):
        raise ValueError("decision training accepts non-empty train-group scenarios only")
    total, gradient = 0.0, [0.0, 0.0]
    for record in scenarios:
        rows = {row.route_id: row for row in record.routes}
        probabilities = {row.route_id: risk_probability(row.features, theta) for row in record.routes}
        costs = [_plan_estimated_cost(plan, rows, probabilities) for plan in record.plans]
        actual = [float(plan["realized_objective_train_only"]) for plan in record.plans]
        baseline = min(costs)
        raw = [math.exp(-(cost - baseline) / PLAN_TEMPERATURE) for cost in costs]
        norm = sum(raw)
        weights = [value / norm for value in raw]
        expected_actual = sum(w * value for w, value in zip(weights, actual))
        total += expected_actual
        dloss_dcost = [-(w * (value - expected_actual)) / PLAN_TEMPERATURE for w, value in zip(weights, actual)]
        for route in record.routes:
            contribution = 0.0
            for plan, coeff in zip(record.plans, dloss_dcost):
                if route.route_id in plan["route_ids"]:
                    contribution += coeff * route.exposure_scale
            for k, value in enumerate(_probability_gradient(route.features, theta)):
                gradient[k] += contribution * value
    n = len(scenarios)
    return total / n, [value / n for value in gradient]


def finite_difference_decision_gradient(theta: Sequence[float], scenarios: Sequence[ProspectiveScenario], h: float = 1e-5) -> list[float]:
    if h <= 0 or not math.isfinite(h):
        raise ValueError("h must be finite and positive")
    answer = []
    for k in range(2):
        plus, minus = list(theta), list(theta)
        plus[k] += h
        minus[k] -= h
        fp = _decision_loss_gradient(plus, scenarios)[0]
        fm = _decision_loss_gradient(minus, scenarios)[0]
        answer.append((fp - fm) / (2 * h))
    return answer


def _fit_decision_vqc(train: Sequence[ProspectiveScenario], steps: int = VQC_STEPS) -> tuple[list[float], list[float]]:
    theta = [0.31, -0.47]
    history = []
    for _ in range(steps):
        loss, gradient = _decision_loss_gradient(theta, train)
        history.append(loss)
        theta = [theta[k] - 0.025 * gradient[k] for k in range(2)]
    history.append(_decision_loss_gradient(theta, train)[0])
    return theta, history


def _fit_predictive_vqc(rows: Sequence[ProspectiveRoute]) -> list[float]:
    examples = [Example(row.features, int(row.train_label), "train", index) for index, row in enumerate(rows)]
    theta = [0.31, -0.47]
    for _ in range(VQC_STEPS):
        _, gradient = _loss_gradient(theta, examples, "predictive")
        theta = [theta[k] - 0.08 * gradient[k] for k in range(2)]
    return theta


def _fit_logistic(rows: Sequence[ProspectiveRoute]) -> list[float]:
    examples = [Example(row.features, int(row.train_label), "train", index) for index, row in enumerate(rows)]
    return _fit_classical(examples)


def _decision_input(record: ProspectiveScenario) -> ForecastDecisionInput:
    decision_plans = tuple({"route_ids": list(plan["route_ids"])} for plan in record.plans)
    return ForecastDecisionInput(record.scenario_id, record.forecast_multiplier,
                                 record.forecast, record.routes, decision_plans)


def _select_forecast_plan(data: ForecastDecisionInput, kind: str, model: Sequence[float]) -> tuple[dict[str, Any], dict[str, float]]:
    """Model-visible function: its input type has no realized scenario or labels."""
    rows = {row.route_id: row for row in data.routes}
    if kind == "classical":
        probabilities = {route_id: _sigmoid(model[0] + model[1] * row.features[0] + model[2] * row.features[1])
                         for route_id, row in rows.items()}
    else:
        probabilities = {route_id: risk_probability(row.features, model) for route_id, row in rows.items()}
    ranked = [(_plan_estimated_cost(plan, rows, probabilities), tuple(plan["route_ids"]), plan)
              for plan in data.plans]
    estimated_cost, _, chosen = min(ranked, key=lambda row: (row[0], row[1]))
    return {"scenario_id": data.scenario_id, "forecast_multiplier": data.forecast_multiplier,
            "route_ids": list(chosen["route_ids"]), "estimated_plan_cost": estimated_cost}, probabilities


def _evaluate_after_selection(record: ProspectiveScenario, selected: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    """Evaluation opens realized travel only after forecast-only route selection."""
    actual_routes = {_route_id(route): route for route in generate_routes(record.realized)}
    plan = _materialize_plan(selected["route_ids"], actual_routes)
    validation = validate_plan(record.realized, plan)
    if not validation["valid"]:
        raise RuntimeError(f"{record.scenario_id}: selected plan failed independent realized validator: {validation['errors']}")
    rows = {row.route_id: row for row in record.routes}
    selected_routes = [{
        "route_id": route_id,
        "features": list(rows[route_id].features),
        "forecast_travel_cost": rows[route_id].forecast_travel_cost,
        "risk_probability": probabilities[route_id],
        "estimated_exposure": probabilities[route_id] * rows[route_id].exposure_scale,
        "validator_realized_exposure": next(r["lateness_exposure"] for r in plan["routes"] if _route_id(r) == route_id),
    } for route_id in selected["route_ids"]]
    return {"scenario_id": record.scenario_id, "forecast_multiplier": record.forecast_multiplier,
            "selected_route_ids": selected["route_ids"], "selected_routes": selected_routes,
            "estimated_plan_cost": selected["estimated_plan_cost"], "estimated_exposure": sum(r["estimated_exposure"] for r in selected_routes),
            "raw_realized_plan": plan, "validation": validation}


def _metrics(outcomes: Sequence[dict[str, Any]], reference: dict[str, float]) -> dict[str, Any]:
    n = len(outcomes)
    return {"n_scenarios": n,
            "feasibility_rate": sum(row["validation"]["feasible"] for row in outcomes) / n,
            "coverage_rate": sum(row["validation"]["valid"] for row in outcomes) / n,
            "mean_validator_objective": sum(row["validation"]["recomputed"]["objective"] for row in outcomes) / n,
            "mean_validator_travel_cost": sum(row["validation"]["recomputed"]["travel_cost"] for row in outcomes) / n,
            "mean_validator_lateness_exposure": sum(row["validation"]["recomputed"]["lateness_exposure"] for row in outcomes) / n,
            "mean_model_estimated_exposure": sum(row["estimated_exposure"] for row in outcomes) / n,
            "mean_regret_vs_exact_reference": sum(row["validation"]["recomputed"]["objective"] - reference[row["scenario_id"]]
                                                  for row in outcomes) / n}


def run_prospective_study(base: dict[str, Any], protocol_sha256: str) -> dict[str, Any]:
    records = build_prospective_scenarios(base)
    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    test = [record for record in records if record.split == "test"]
    if (len(train), len(validation), len(test)) != (12, 6, 6):
        raise RuntimeError("scenario split differs from frozen protocol")
    train_rows = [route for record in train for route in record.routes]
    if not train_rows or any(route.train_label is None for route in train_rows):
        raise RuntimeError("training route labels are missing")
    if any(route.train_label is not None for record in validation + test for route in record.routes):
        raise RuntimeError("validation/test labels crossed the model boundary")
    decision_theta, loss_history = _fit_decision_vqc(train)
    predictive_theta = _fit_predictive_vqc(train_rows)
    logistic = _fit_logistic(train_rows)
    methods = {"decision_focused_vqc": ("vqc", decision_theta),
               "predictive_only_vqc": ("vqc", predictive_theta),
               "classical_logistic": ("classical", logistic)}
    evaluated: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
    exact_reference = {}
    exact_outcomes = []
    # Selection receives forecast only. Realized evaluation begins after each selection.
    for record in validation + test:
        forecast_input = _decision_input(record)
        chosen_forecasts = {}
        for name, (kind, model) in methods.items():
            chosen_forecasts[name] = _select_forecast_plan(forecast_input, kind, model)
        # All methods have committed to forecast-only plans before actuals open.
        exact = solve_exact(record.realized)
        exact_check = validate_plan(record.realized, exact)
        if not exact_check["valid"]:
            raise RuntimeError(f"{record.scenario_id}: exact reference failed validator: {exact_check['errors']}")
        exact_reference[record.scenario_id] = float(exact_check["recomputed"]["objective"])
        exact_outcomes.append({"scenario_id": record.scenario_id, "split": record.split,
                               "objective": exact_reference[record.scenario_id], "validation": exact_check})
        for name, (selection, probabilities) in chosen_forecasts.items():
            evaluated[name].append({"split": record.split,
                                    **_evaluate_after_selection(record, selection, probabilities)})
    split_details = {}
    for split, group in (("train", train), ("validation", validation), ("test", test)):
        split_details[split] = [{"scenario_id": row.scenario_id,
                                 "forecast_multiplier": row.forecast_multiplier,
                                 "paired_forecast_realized_sha256": row.data_hash,
                                 "forecast_matrix_sha256": _canonical_hash(row.forecast),
                                 "realized_matrix_sha256": _canonical_hash(row.realized),
                                 "candidate_route_count": len(row.routes),
                                 "feasible_plan_count": len(row.plans)} for row in group]
    training_scenarios = train[:2]
    analytic = _decision_loss_gradient([0.31, -0.47], training_scenarios)[1]
    numeric = finite_difference_decision_gradient([0.31, -0.47], training_scenarios)
    result = {
        "evidence_label": "synthetic prospective noisy-forecast simulation; exact local CPU statevector; no vendor service/QPU",
        "protocol_id": "prospective-dispatch-vqc-v1",
        "protocol_sha256": protocol_sha256,
        "source_fixture_sha256": _canonical_hash(base),
        "information_boundary_check": {"train_labels_available": True,
                                        "validation_labels_passed_to_models": False,
                                        "test_labels_passed_to_models": False,
                                        "forecast_and_realized_matrices_are_separately_sampled": True,
                                        "realized_plan_validation_occurs_after_forecast_plan_selection": True},
        "objective_contract": "travel duration + contract penalty rate * realized lateness minutes; units remain the tiny domain proxy, not calibrated currency",
        "feature_contract": {"x0": "clip(forecast candidate-route travel duration / 30, 0, 1)",
                             "x1": "nominal contractual window pressure from source inputs",
                             "label": "train-only indicator that independently sampled realized route has positive lateness exposure"},
        "decision_contract": "forecast-only exact-cover plan selection by forecast travel duration + predicted probability * one-minute penalty exposure; score selected plan against separately sampled realized scenario and independent validator",
        "split": split_details,
        "training": {"train_scenarios": len(train), "train_route_rows": len(train_rows), "optimizer_steps_per_model": VQC_STEPS,
                     "decision_initial_loss": loss_history[0], "decision_final_loss": loss_history[-1],
                     "decision_theta": decision_theta, "predictive_theta": predictive_theta,
                     "classical_logistic_weights": logistic,
                     "gradient_check_scenarios": [row.scenario_id for row in training_scenarios],
                     "decision_gradient_parameter_shift_chain": analytic,
                     "decision_gradient_centered_finite_difference_h_1e-5": numeric,
                     "decision_gradient_max_absolute_error": max(abs(a-b) for a, b in zip(analytic, numeric))},
        "validation_metrics": {name: _metrics([row for row in rows if row["split"] == "validation"], exact_reference)
                               for name, rows in evaluated.items()},
        "test_metrics": {name: _metrics([row for row in rows if row["split"] == "test"], exact_reference)
                         for name, rows in evaluated.items()},
        "test_outcomes": {name: [row for row in rows if row["split"] == "test"] for name, rows in evaluated.items()},
        "validation_outcomes": {name: [row for row in rows if row["split"] == "validation"] for name, rows in evaluated.items()},
        "exact_reference": exact_outcomes,
        "resources": {"qubits": 2, "statevector": "exact amplitude simulation, shots=0",
                      "scenarios": len(records), "train": len(train), "validation": len(validation), "test": len(test),
                      "max_orders": max(len(base["orders"]) for _ in records), "external_calls": 0},
    }
    return result
