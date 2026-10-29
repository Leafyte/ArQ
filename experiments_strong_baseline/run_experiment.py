"""Frozen synthetic matched comparison; use --freeze before the one evaluation run."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.domain import generate_routes, solve_exact  # noqa: E402
from backend.learning_dispatch import _exposure_scale, _feasible_plans, _nominal_pressure, _route_id  # noqa: E402
from backend.prospective_dispatch import (  # noqa: E402
    ProspectiveRoute, ProspectiveScenario,
    _evaluate_after_selection, _fit_decision_vqc, _fit_logistic,
    _fit_predictive_vqc, _plan_estimated_cost,
)
from backend.quantum_learning import risk_probability  # noqa: E402
from backend.validator import validate_plan  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
LOCK_PATH = HERE / "frozen_lock.json"
MANIFEST_PATH = HERE / "dataset_manifest.json"
OUTPUT_PATH = HERE / "results.json"


@dataclass(frozen=True)
class InferenceRoute:
    """Forecast-only route view; training labels are structurally absent."""
    route_id: str
    features: tuple[float, float]
    forecast_travel_cost: float
    exposure_scale: float


@dataclass(frozen=True)
class InferenceScenario:
    """Model-visible inputs; no realized matrix, train label, or actual objective."""
    scenario_id: str
    forecast_multiplier: float
    forecast: dict[str, Any]
    routes: tuple[InferenceRoute, ...]
    plans: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CartNode:
    positive_probability: float
    feature: int | None = None
    threshold: float | None = None
    left: "CartNode | None" = None
    right: "CartNode | None" = None


class GroupedBootstrapCartForest:
    """Small nonlinear forest with scenario-group bootstrap to limit leakage."""

    def __init__(self, n_estimators: int = 64, max_depth: int = 3,
                 min_samples_leaf: int = 2, seed: int = 1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.trees: list[CartNode] = []
        self.node_count = 0

    @staticmethod
    def _gini(rows: list[tuple[tuple[float, float], int]]) -> float:
        if not rows:
            return 0.0
        p = sum(label for _, label in rows) / len(rows)
        return 2.0 * p * (1.0 - p)

    def _fit_node(self, rows: list[tuple[tuple[float, float], int]], depth: int,
                  rng: random.Random) -> CartNode:
        positive = sum(label for _, label in rows)
        probability = positive / len(rows)
        self.node_count += 1
        leaf = CartNode(probability)
        if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf or probability in (0.0, 1.0):
            return leaf
        feature = rng.randrange(2)  # one of the two preregistered input features per node
        values = sorted({row[feature] for row, _ in rows})
        thresholds = [(a + b) / 2.0 for a, b in zip(values, values[1:])]
        base_impurity = self._gini(rows)
        best: tuple[float, float, list[tuple[tuple[float, float], int]],
                    list[tuple[tuple[float, float], int]]] | None = None
        for threshold in thresholds:
            left = [item for item in rows if item[0][feature] <= threshold]
            right = [item for item in rows if item[0][feature] > threshold]
            if min(len(left), len(right)) < self.min_samples_leaf:
                continue
            gain = (base_impurity - len(left) / len(rows) * self._gini(left)
                    - len(right) / len(rows) * self._gini(right))
            if best is None or gain > best[0] + 1e-15:
                best = (gain, threshold, left, right)
        if best is None or best[0] <= 1e-12:
            return leaf
        _, threshold, left, right = best
        return CartNode(probability, feature, threshold,
                        self._fit_node(left, depth + 1, rng),
                        self._fit_node(right, depth + 1, rng))

    def fit(self, groups: dict[str, list[tuple[tuple[float, float], int]]]) -> "GroupedBootstrapCartForest":
        if not groups or any(not rows for rows in groups.values()):
            raise ValueError("forest requires non-empty training scenario groups")
        self.trees = []
        self.node_count = 0
        group_ids = sorted(groups)
        for tree_index in range(self.n_estimators):
            rng = random.Random(self.seed + tree_index * 1_000_003)
            sampled_ids = [rng.choice(group_ids) for _ in group_ids]
            boot = [row for group_id in sampled_ids for row in groups[group_id]]
            self.trees.append(self._fit_node(boot, 0, rng))
        return self

    def predict_probability(self, features: Sequence[float]) -> float:
        if len(features) != 2 or not self.trees:
            raise ValueError("fit forest and provide two features before prediction")
        predictions = []
        for tree in self.trees:
            node = tree
            while node.feature is not None:
                if node.threshold is None or node.left is None or node.right is None:
                    raise RuntimeError("malformed CART node")
                node = node.left if float(features[node.feature]) <= node.threshold else node.right
            predictions.append(node.positive_probability)
        return sum(predictions) / len(predictions)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest().upper()


def _base() -> dict[str, Any]:
    """New synthetic geometry, independent of existing fixture files."""
    return {
        "depot": "D",
        "vehicles": [{"id": "R1", "capacity": 3}, {"id": "R2", "capacity": 2}],
        "orders": [
            {"id": "P", "demand": 1, "window": [0, 5.5], "service": 0.7, "penalty": 4.0},
            {"id": "Q", "demand": 1, "window": [0, 7.5], "service": 0.8, "penalty": 1.5},
            {"id": "R", "demand": 2, "window": [1, 10.0], "service": 0.6, "penalty": 3.0},
        ],
        "travel": {
            "D": {"D": 0.0, "P": 2.1, "Q": 1.5, "R": 2.8},
            "P": {"D": 2.0, "P": 0.0, "Q": 1.4, "R": 2.2},
            "Q": {"D": 1.7, "P": 1.2, "Q": 0.0, "R": 1.9},
            "R": {"D": 2.6, "P": 2.1, "Q": 2.0, "R": 0.0},
        },
    }


def _make_pair(base: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], dict[str, Any], float]:
    latent = rng.uniform(0.75, 2.1)
    forecast_mult = min(2.5, max(0.5, latent + rng.gauss(0.0, 0.3)))
    forecast = json.loads(json.dumps(base))
    realized = json.loads(json.dumps(base))
    for origin in sorted(base["travel"]):
        for dest in sorted(base["travel"]):
            if origin == dest:
                continue
            raw = float(base["travel"][origin][dest])
            forecast["travel"][origin][dest] = round(raw * forecast_mult * rng.uniform(0.97, 1.03), 6)
            realized["travel"][origin][dest] = round(raw * latent * rng.uniform(0.90, 1.10), 6)
    return forecast, realized, forecast_mult


def build_records() -> list[ProspectiveScenario]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cfg = protocol["scenario_family"]
    base = _base()
    nominal = {_route_id(route): route for route in generate_routes(base)}
    records: list[ProspectiveScenario] = []
    for rep in range(cfg["replicates"]):
        rng = random.Random(cfg["family_seed"] + rep * 10_007)
        scenario_ids = [f"rep{rep + 1:02d}-case{i:02d}" for i in range(cfg["scenario_groups_per_replicate"])]
        split_indices = list(range(len(scenario_ids)))
        random.Random(cfg["split_seed"] + rep * 1009).shuffle(split_indices)
        # Read explicit group counts without deriving or adapting them from outcomes.
        train_n = cfg["split_per_replicate"]["train"]
        val_n = cfg["split_per_replicate"]["validation"]
        split_map = {index: "train" for index in split_indices[:train_n]}
        split_map.update({index: "validation" for index in split_indices[train_n:train_n + val_n]})
        split_map.update({index: "test" for index in split_indices[train_n + val_n:]})
        for case_index, scenario_id in enumerate(scenario_ids):
            split = split_map[case_index]
            forecast, realized, forecast_mult = _make_pair(base, rng)
            f_routes = generate_routes(forecast)
            f_by_id = {_route_id(route): route for route in f_routes}
            r_by_id = ({_route_id(route): route for route in generate_routes(realized)} if split == "train" else {})
            forecast_plans = _feasible_plans(forecast, f_routes)
            if not forecast_plans:
                raise RuntimeError(f"{scenario_id}: no forecast-feasible plan")
            labels: dict[str, int] = {}
            if split == "train":
                labels = {route_id: int(any(float(stop["lateness"]) > 0.0 for stop in route["stops"]))
                          for route_id, route in r_by_id.items()}
            route_rows = []
            for route_id, route in f_by_id.items():
                if route_id not in nominal or (split == "train" and route_id not in r_by_id):
                    raise RuntimeError(f"{scenario_id}: route identity changed across worlds")
                features = (min(1.0, float(route["travel_cost"]) / 30.0),
                            _nominal_pressure(nominal[route_id], base))
                route_rows.append(ProspectiveRoute(route_id, features, labels.get(route_id),
                                                    float(route["travel_cost"]),
                                                    _exposure_scale(base, route)))
            plans = []
            for p in forecast_plans:
                ids = [_route_id(route) for route in p["routes"]]
                item = {"route_ids": ids, "realized_objective_train_only": None}
                if split == "train":
                    actual = _materialize(ids, r_by_id)
                    checked = validate_plan(realized, actual)
                    if not checked["valid"]:
                        raise RuntimeError(f"{scenario_id}: training plan failed validator")
                    item["realized_objective_train_only"] = checked["recomputed"]["objective"]
                plans.append(item)
            digest = _sha({"forecast": forecast, "realized": realized})
            records.append(ProspectiveScenario(scenario_id, split, forecast_mult, forecast, realized,
                                               digest, tuple(route_rows), tuple(plans)))
    return records


def _materialize(route_ids: Sequence[str], routes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = sorted((routes[key] for key in route_ids), key=lambda row: row["vehicle_id"])
    return {"status": "feasible", "routes": selected,
            "travel_cost": sum(float(row["travel_cost"]) for row in selected),
            "lateness_exposure": sum(float(row["lateness_exposure"]) for row in selected),
            "objective": sum(float(row["objective"]) for row in selected)}


def _manifest(records: list[ProspectiveScenario]) -> dict[str, Any]:
    per_rep: dict[str, Any] = {}
    for rep in range(1, 6):
        rows = [r for r in records if r.scenario_id.startswith(f"rep{rep:02d}-")]
        per_rep[f"rep{rep:02d}"] = {
            "split_ids": {split: [r.scenario_id for r in rows if r.split == split]
                          for split in ("train", "validation", "test")},
            "split_ids_sha256": {split: _sha([r.scenario_id for r in rows if r.split == split])
                                 for split in ("train", "validation", "test")},
            "scenario_hashes_sha256": {r.scenario_id: r.data_hash for r in rows},
            "candidate_routes_per_group": {r.scenario_id: len(r.routes) for r in rows},
            "feasible_plans_per_group": {r.scenario_id: len(r.plans) for r in rows},
        }
    return {"family_id": "independent-dispatch-synthetic-v1", "base_scenario_sha256": _sha(_base()),
            "scenario_group_count": len(records), "replicates": per_rep,
            "label_visibility": {r.scenario_id: ("train_only" if r.split == "train" else "unavailable_to_fit")
                                 for r in records}}


def _decision_input(record: ProspectiveScenario) -> InferenceScenario:
    routes = tuple(InferenceRoute(row.route_id, row.features, row.forecast_travel_cost,
                                  row.exposure_scale) for row in record.routes)
    plans = tuple({"route_ids": list(plan["route_ids"])} for plan in record.plans)
    return InferenceScenario(record.scenario_id, record.forecast_multiplier,
                             record.forecast, routes, plans)


def _select(data: InferenceScenario, probabilities: dict[str, float]) -> dict[str, Any]:
    rows = {row.route_id: row for row in data.routes}
    ranked = [(_plan_estimated_cost(plan, rows, probabilities), tuple(plan["route_ids"]), plan)
              for plan in data.plans]
    estimated, _, chosen = min(ranked, key=lambda row: (row[0], row[1]))
    return {"scenario_id": data.scenario_id, "forecast_multiplier": data.forecast_multiplier,
            "route_ids": list(chosen["route_ids"]), "estimated_plan_cost": estimated}


def _test_labels(record: ProspectiveScenario) -> dict[str, int]:
    routes = {_route_id(route): route for route in generate_routes(record.realized)}
    return {key: int(any(float(stop["lateness"]) > 0.0 for stop in route["stops"]))
            for key, route in routes.items()}


def _brier_by_scenario(record: ProspectiveScenario, probabilities: dict[str, float]) -> float:
    labels = _test_labels(record)
    return sum((probabilities[key] - labels[key]) ** 2 for key in probabilities) / len(probabilities)


def _fit_replicate(train: list[ProspectiveScenario], rep_index: int) -> tuple[dict[str, Any], dict[str, float]]:
    train_rows = [row for scenario in train for row in scenario.routes]
    groups = {scenario.scenario_id: [(row.features, int(row.train_label)) for row in scenario.routes]
              for scenario in train}
    start = perf_counter()
    decision_theta, decision_history = _fit_decision_vqc(train)
    decision_seconds = perf_counter() - start
    start = perf_counter()
    predictive_theta = _fit_predictive_vqc(train_rows)
    predictive_seconds = perf_counter() - start
    start = perf_counter()
    logistic = _fit_logistic(train_rows)
    logistic_seconds = perf_counter() - start
    start = perf_counter()
    cfg = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["methods_and_fixed_budgets"]["grouped_bootstrap_cart_forest"]
    forest_seed = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["scenario_family"]["family_seed"] + rep_index
    forest = GroupedBootstrapCartForest(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                        min_samples_leaf=cfg["min_samples_leaf"], seed=forest_seed).fit(groups)
    forest_seconds = perf_counter() - start
    return ({"decision_focused_vqc": ("vqc", decision_theta),
             "predictive_only_vqc": ("vqc", predictive_theta),
             "logistic_regression": ("logistic", logistic),
             "grouped_bootstrap_cart_forest": ("forest", forest)},
            {"decision_focused_vqc_fit_seconds": decision_seconds,
             "predictive_only_vqc_fit_seconds": predictive_seconds,
             "logistic_regression_fit_seconds": logistic_seconds,
             "random_forest_fit_seconds": forest_seconds,
             "decision_vqc_updates": 120, "predictive_vqc_updates": 120,
             "logistic_updates": 500, "forest_trees": cfg["n_estimators"],
             "forest_node_count": forest.node_count,
             "training_route_rows": len(train_rows), "decision_final_training_loss": decision_history[-1]})


def _probabilities(data: InferenceScenario, kind: str, model: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    if kind == "forest":
        for row in data.routes:
            output[row.route_id] = model.predict_probability(row.features)
    elif kind == "logistic":
        weights = model
        for row in data.routes:
            z = weights[0] + weights[1] * row.features[0] + weights[2] * row.features[1]
            output[row.route_id] = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))
    else:
        for row in data.routes:
            output[row.route_id] = risk_probability(row.features, model)
    return output


def _summarize(outcomes: list[dict[str, Any]]) -> dict[str, float]:
    n = len(outcomes)
    return {"n_scenarios": n,
            "mean_validator_objective": sum(x["validator_objective"] for x in outcomes) / n,
            "mean_regret_vs_exact": sum(x["regret_vs_exact"] for x in outcomes) / n,
            "valid_plan_rate": sum(x["valid"] for x in outcomes) / n,
            "mean_lateness_exposure": sum(x["lateness_exposure"] for x in outcomes) / n,
            "mean_route_brier": sum(x["route_brier"] for x in outcomes) / n}


def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    return {"mean_of_replicate_means": mean, "sample_sd_of_replicate_means": sd,
            "min_replicate_mean": min(values), "max_replicate_mean": max(values),
            "replicates": n}


def freeze_dataset() -> dict[str, Any]:
    records = build_records()
    manifest = _manifest(records)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest_path": str(MANIFEST_PATH), "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest().upper(),
            "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest().upper(),
            "scenario_groups": len(records)}


def run_experiment() -> dict[str, Any]:
    study_started = perf_counter()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    manifest_bytes = MANIFEST_PATH.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest().upper()
    manifest_obj = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(_canonical(manifest_obj)).hexdigest().upper()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    if (protocol_hash != lock["protocol_sha256"] or
            manifest_hash != lock["dataset_manifest_sha256"] or
            runner_hash != lock["runner_sha256"]):
        raise RuntimeError("frozen protocol, split manifest, or runner changed; refusing evaluation")
    if OUTPUT_PATH.exists():
        raise RuntimeError("results.json already exists; refusing to repeat or overwrite held-out evaluation")
    records = build_records()
    if _manifest(records) != manifest_obj:
        raise RuntimeError("regenerated groups/split hashes do not match frozen dataset manifest")

    all_outcomes = {name: [] for name in ("decision_focused_vqc", "predictive_only_vqc",
                                           "logistic_regression", "grouped_bootstrap_cart_forest")}
    replicate_details = []
    exact_reference_by_case = {}
    for rep in range(5):
        prefix = f"rep{rep + 1:02d}-"
        group = [r for r in records if r.scenario_id.startswith(prefix)]
        train = [r for r in group if r.split == "train"]
        validation = [r for r in group if r.split == "validation"]
        test = [r for r in group if r.split == "test"]
        if any(row.train_label is not None for scenario in validation + test for row in scenario.routes):
            raise RuntimeError("validation/test labels present at model-fitting boundary")
        methods, work = _fit_replicate(train, rep)
        # Freeze each model's forecast-only selections before opening any realized matrix.
        selected: dict[str, dict[str, tuple[dict[str, Any], dict[str, float]]]] = {name: {} for name in methods}
        for scenario in validation + test:
            visible = _decision_input(scenario)
            for name, (kind, model) in methods.items():
                probabilities = _probabilities(visible, kind, model)
                selection = _select(visible, probabilities)
                selected[name][scenario.scenario_id] = (selection, probabilities)
        # Post-selection only: compute reference and score every method on the same actuals.
        replicate_outcomes = {name: [] for name in methods}
        for scenario in validation + test:
            exact = solve_exact(scenario.realized)
            exact_validation = validate_plan(scenario.realized, exact)
            if not exact_validation["valid"]:
                raise RuntimeError(f"{scenario.scenario_id}: exact reference invalid")
            exact_value = float(exact_validation["recomputed"]["objective"])
            exact_reference_by_case[scenario.scenario_id] = {"objective": exact_value, "valid": True}
            for name in methods:
                selection, probabilities = selected[name][scenario.scenario_id]
                result = _evaluate_after_selection(scenario, selection, probabilities)
                realized_labels = _test_labels(scenario)
                brier = sum((probabilities[key] - realized_labels[key]) ** 2 for key in probabilities) / len(probabilities)
                validated_objective = float(result["validation"]["recomputed"]["objective"])
                outcome = {"scenario_id": scenario.scenario_id, "replicate": rep + 1,
                           "split": scenario.split, "selected_route_ids": selection["route_ids"],
                           "validator_objective": validated_objective,
                           "exact_reference_objective": exact_value,
                           "regret_vs_exact": validated_objective - exact_value,
                           "valid": bool(result["validation"]["valid"]),
                           "lateness_exposure": float(result["validation"]["recomputed"]["lateness_exposure"]),
                           "route_brier": brier,
                           "validation_errors": result["validation"]["errors"]}
                replicate_outcomes[name].append(outcome)
                all_outcomes[name].append(outcome)
        replicate_details.append({"replicate": rep + 1,
                                  "split_ids": {split: [s.scenario_id for s in group if s.split == split]
                                                for split in ("train", "validation", "test")},
                                  "training_work": work,
                                  "validation_metrics": {name: _summarize([x for x in rows if x["split"] == "validation"])
                                                         for name, rows in replicate_outcomes.items()},
                                  "test_metrics": {name: _summarize([x for x in rows if x["split"] == "test"])
                                                   for name, rows in replicate_outcomes.items()},
                                  "test_outcomes": {name: [x for x in rows if x["split"] == "test"]
                                                    for name, rows in replicate_outcomes.items()}})
    test_by_method = {name: [row for row in rows if row["split"] == "test"] for name, rows in all_outcomes.items()}
    rep_means = {name: [rep["test_metrics"][name]["mean_validator_objective"] for rep in replicate_details]
                 for name in test_by_method}
    paired = {}
    for name, rows in test_by_method.items():
        if name == "logistic_regression":
            continue
        logistic = {row["scenario_id"]: row["validator_objective"] for row in test_by_method["logistic_regression"]}
        deltas = [{"scenario_id": row["scenario_id"], "method_minus_logistic": row["validator_objective"] - logistic[row["scenario_id"]]}
                  for row in rows]
        paired[name] = {"per_scenario": deltas,
                        "replicate_means": [sum(x["method_minus_logistic"] for x in deltas
                                                  if x["scenario_id"].startswith(f"rep{rep:02d}-")) / 6
                                            for rep in range(1, 6)],
                        "replicate_mean_summary": _stats([sum(x["method_minus_logistic"] for x in deltas
                                                               if x["scenario_id"].startswith(f"rep{rep:02d}-")) / 6
                                                         for rep in range(1, 6)])}
    total_wall_seconds = perf_counter() - study_started
    return {
        "experiment_id": "terra-independent-matched-baseline-v1",
        "evidence_type": "one-time execution of a preregistered synthetic local comparison",
        "execution": {"python": sys.version.split()[0], "platform": platform.platform(),
                      "nonlinear_baseline": "owned dependency-free grouped-bootstrap CART forest",
                      "protocol_sha256": protocol_hash,
                      "dataset_manifest_sha256": manifest_hash, "runner_sha256": runner_hash,
                      "external_calls": 0,
                      "total_wall_seconds_including_generation_fit_validation_test": total_wall_seconds},
        "protocol": json.loads(protocol_bytes),
        "dataset_manifest": manifest_obj,
        "test_metrics_by_method": {name: _summarize(rows) for name, rows in test_by_method.items()},
        "per_scenario_paired_objective_differences_vs_logistic": paired,
        "replicate_mean_objective_summary": {name: _stats(values) for name, values in rep_means.items()},
        "replicates": replicate_details,
        "exact_reference": exact_reference_by_case,
        "limitations": json.loads(protocol_bytes)["limitations"],
    }


def main() -> None:
    if sys.argv[1:] == ["--freeze"]:
        print(json.dumps(freeze_dataset(), indent=2))
        return
    if sys.argv[1:]:
        raise SystemExit("usage: python run_experiment.py [--freeze]")
    result = run_experiment()
    if OUTPUT_PATH.exists():
        raise RuntimeError("results.json already exists; refusing to overwrite held-out evaluation")
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(OUTPUT_PATH), "test_metrics": result["test_metrics_by_method"],
                      "paired": {name: row["replicate_mean_summary"] for name, row in
                                 result["per_scenario_paired_objective_differences_vs_logistic"].items()}}, indent=2))


if __name__ == "__main__":
    main()
