"""Decision Quality Optimization (Track A - P0) Benchmark Engine.

Implements and evaluates the 7-way comparator pipeline on frozen scenarios:
Pipeline A: Classical ML + Exact selection
Pipeline B: Standard VQC (predictive-only) + Exact selection
Pipeline C: Decision-focused VQC + Exact selection
Pipeline D: Standard VQC + Standard QAOA (p=1)
Pipeline E: Decision-focused VQC + Standard QAOA (p=1)
Pipeline F: Decision-focused VQC + Warm-start QAOA (p=1)
Pipeline G: Classical ML + Warm-start QAOA (p=1)

Features:
- Controlled ablation of decision-focused temperature, learning rate, and loss formulations.
- Rigorous financial accounting: transport cost, contractual SLA penalties, total cost,
  SLA breach rate, SLA breach severity, decision regret vs realized oracle, hard-constraint violations,
  and valid order coverage.
- Leakage-safe train, validation, and held-out test splits across 5 independent replicates.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.domain import generate_routes, solve_exact, validate_scenario
from backend.financial import PENALTY_RATE_UNIT, validate_financial_plan
from backend.learning_dispatch import (
    _exposure_scale,
    _feasible_plans,
    _nominal_pressure,
    _route_id,
)
from backend.prospective_dispatch import (
    ProspectiveRoute,
    ProspectiveScenario,
    _evaluate_after_selection,
    _materialize_plan,
    _plan_estimated_cost,
    _probability_gradient,
    _sigmoid,
)
from backend.qaoa import QAOAResult, continuous_relaxation, qaoa_state, run_qaoa
from backend.quantum_learning import (
    Example,
    _fit_classical,
    _loss_gradient,
    risk_probability,
)
from backend.qubo import build_route_qubo, validate_selection
from backend.validator import validate_plan
from experiments_v1_1.quantum_optimization import execute_qaoa_benchmark

OPERATING_COST_PER_TRAVEL_MINUTE = 0.50
CURRENCY = "USD"
FEATURE_MAX_TRAVEL = 30.0


def _base_scenario() -> Dict[str, Any]:
    """Reproducible base geometry with contractual SLA terms."""
    return {
        "depot": "D",
        "vehicles": [
            {"id": "R1", "capacity": 3},
            {"id": "R2", "capacity": 2},
        ],
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


def generate_world_pair(
    base: Dict[str, Any], rng: random.Random
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    """Generate paired forecast and realized scenarios with correlated congestion."""
    latent = rng.uniform(0.75, 2.1)
    forecast_mult = min(2.5, max(0.5, latent + rng.gauss(0.0, 0.3)))
    forecast = json.loads(json.dumps(base))
    realized = json.loads(json.dumps(base))
    for u in sorted(base["travel"]):
        for v in sorted(base["travel"]):
            if u == v:
                continue
            raw = float(base["travel"][u][v])
            forecast["travel"][u][v] = round(raw * forecast_mult * rng.uniform(0.97, 1.03), 6)
            realized["travel"][u][v] = round(raw * latent * rng.uniform(0.90, 1.10), 6)
    return forecast, realized, forecast_mult, latent


def build_experiment_records(
    n_replicates: int = 5,
    cases_per_replicate: int = 30,
    family_seed: int = 2026103001,
    split_seed: int = 2026103002,
) -> List[ProspectiveScenario]:
    """Build paired training, validation and test scenario groups."""
    base = _base_scenario()
    nominal_routes = {_route_id(r): r for r in generate_routes(base)}
    records: List[ProspectiveScenario] = []

    for rep in range(n_replicates):
        rng = random.Random(family_seed + rep * 10_007)
        case_ids = [f"rep{rep + 1:02d}-case{i:02d}" for i in range(cases_per_replicate)]
        split_indices = list(range(cases_per_replicate))
        random.Random(split_seed + rep * 1009).shuffle(split_indices)

        if cases_per_replicate == 30:
            train_n = 18
            val_n = 6
        else:
            train_n = max(1, int(cases_per_replicate * 0.6))
            val_n = max(1, int(cases_per_replicate * 0.2))
        split_map = {idx: "train" for idx in split_indices[:train_n]}
        split_map.update({idx: "validation" for idx in split_indices[train_n : train_n + val_n]})
        split_map.update({idx: "test" for idx in split_indices[train_n + val_n :]})

        for case_idx, scenario_id in enumerate(case_ids):
            split = split_map[case_idx]
            forecast, realized, f_mult, r_mult = generate_world_pair(base, rng)
            f_routes = generate_routes(forecast)
            feasible_plans = _feasible_plans(forecast, f_routes)
            if not feasible_plans:
                continue

            r_routes = (
                {_route_id(r): r for r in generate_routes(realized)}
                if split == "train"
                else {}
            )
            labels = {}
            if split == "train":
                labels = {
                    r_id: int(any(float(st["lateness"]) > 0 for st in r["stops"]))
                    for r_id, r in r_routes.items()
                }

            route_rows = []
            for r in f_routes:
                rid = _route_id(r)
                if rid not in nominal_routes:
                    continue
                pressure = _nominal_pressure(nominal_routes[rid], base)
                features = (min(1.0, float(r["travel_cost"]) / FEATURE_MAX_TRAVEL), pressure)
                route_rows.append(
                    ProspectiveRoute(
                        route_id=rid,
                        features=features,
                        train_label=labels.get(rid),
                        forecast_travel_cost=float(r["travel_cost"]),
                        exposure_scale=_exposure_scale(base, r),
                    )
                )

            plans = []
            for fp in feasible_plans:
                r_ids = [_route_id(r) for r in fp["routes"]]
                item = {"route_ids": r_ids, "realized_objective_train_only": None}
                if split == "train":
                    mat = _materialize_plan(r_ids, r_routes)
                    val = validate_plan(realized, mat)
                    item["realized_objective_train_only"] = float(val["recomputed"]["objective"])
                plans.append(item)

            digest = f"{scenario_id}-{split}"
            records.append(
                ProspectiveScenario(
                    scenario_id,
                    split,
                    f_mult,
                    forecast,
                    realized,
                    digest,
                    tuple(route_rows),
                    tuple(plans),
                )
            )

    return records


# ==============================================================================
# Model Training Routines
# ==============================================================================

def train_classical_logistic(train_scenarios: List[ProspectiveScenario], steps: int = 120) -> List[float]:
    """Train classical logistic regression predictor."""
    examples = []
    idx = 0
    for sc in train_scenarios:
        for r in sc.routes:
            if r.train_label is not None:
                examples.append(Example(r.features, int(r.train_label), "train", idx))
                idx += 1
    # Run gradient descent on BCE
    w = [0.0, 0.0, 0.0]  # [w0, w1, bias]
    lr = 0.3
    n = len(examples)
    for _ in range(steps):
        grad = [0.0, 0.0, 0.0]
        for ex in examples:
            z = w[0] * ex.features[0] + w[1] * ex.features[1] + w[2]
            p = _sigmoid(z)
            err = p - ex.label
            grad[0] += err * ex.features[0]
            grad[1] += err * ex.features[1]
            grad[2] += err
        w = [w[k] - (lr / n) * grad[k] for k in range(3)]
    return w


def predict_logistic(features: Sequence[float], weights: Sequence[float]) -> float:
    z = weights[0] * features[0] + weights[1] * features[1] + weights[2]
    return _sigmoid(z)


def train_predictive_vqc(train_scenarios: List[ProspectiveScenario], steps: int = 120, lr: float = 0.08) -> List[float]:
    """Train standard VQC with BCE loss."""
    examples = []
    idx = 0
    for sc in train_scenarios:
        for r in sc.routes:
            if r.train_label is not None:
                examples.append(Example(r.features, int(r.train_label), "train", idx))
                idx += 1
    theta = [0.31, -0.47]
    for _ in range(steps):
        _, grad = _loss_gradient(theta, examples, "predictive")
        theta = [theta[k] - lr * grad[k] for k in range(2)]
    return theta


def decision_loss_and_gradient(
    theta: Sequence[float],
    scenarios: Sequence[ProspectiveScenario],
    temperature: float = 1.0,
    cost_weight: float = 1.0,
) -> Tuple[float, List[float]]:
    """Compute surrogate dispatch loss and parameter-shift chain rule gradients."""
    total_loss = 0.0
    total_grad = [0.0, 0.0]

    for record in scenarios:
        rows = {r.route_id: r for r in record.routes}
        probs = {r.route_id: risk_probability(r.features, theta) for r in record.routes}

        # Estimate dispatch costs per feasible plan
        costs = [
            sum(
                rows[rid].forecast_travel_cost * cost_weight
                + probs[rid] * rows[rid].exposure_scale
                for rid in p["route_ids"]
            )
            for p in record.plans
        ]
        actuals = [float(p["realized_objective_train_only"]) for p in record.plans]

        # Softmax surrogate selection weights
        base_c = min(costs)
        unnorm = [math.exp(-(c - base_c) / temperature) for c in costs]
        norm = sum(unnorm)
        weights = [u / norm for u in unnorm]

        expected_actual = sum(w * act for w, act in zip(weights, actuals))
        total_loss += expected_actual

        # Chain rule: dLoss / d(cost_i)
        dloss_dcost = [
            -(w * (act - expected_actual)) / temperature
            for w, act in zip(weights, actuals)
        ]

        for r in record.routes:
            dcost_dp = sum(
                coeff * r.exposure_scale
                for p, coeff in zip(record.plans, dloss_dcost)
                if r.route_id in p["route_ids"]
            )
            if dcost_dp != 0.0:
                p_grad = _probability_gradient(r.features, theta)
                total_grad[0] += dcost_dp * p_grad[0]
                total_grad[1] += dcost_dp * p_grad[1]

    n = len(scenarios)
    return total_loss / n, [g / n for g in total_grad]


def train_decision_focused_vqc(
    train_scenarios: List[ProspectiveScenario],
    steps: int = 120,
    lr: float = 0.05,
    temperature: float = 1.0,
    pretrain_predictive: bool = True,
) -> List[float]:
    """Train decision-focused VQC with optional predictive warm start and tuned temperature."""
    if pretrain_predictive:
        # Pre-train with 30 steps of BCE for stable initialization
        theta = train_predictive_vqc(train_scenarios, steps=30, lr=0.08)
    else:
        theta = [0.31, -0.47]

    for _ in range(steps):
        loss, grad = decision_loss_and_gradient(theta, train_scenarios, temperature=temperature)
        theta = [theta[k] - lr * grad[k] for k in range(2)]
    return theta


# ==============================================================================
# Downstream Selection & Independent Financial Evaluation
# ==============================================================================

def select_plan_exact(
    scenario: ProspectiveScenario,
    probabilities: Dict[str, float],
) -> Dict[str, Any]:
    """Select plan with minimum estimated cost using exact enumeration of candidate plans."""
    rows = {r.route_id: r for r in scenario.routes}
    best_plan = None
    best_est_cost = float("inf")
    for p in scenario.plans:
        est = sum(
            rows[rid].forecast_travel_cost * OPERATING_COST_PER_TRAVEL_MINUTE
            + probabilities[rid] * rows[rid].exposure_scale
            for rid in p["route_ids"]
        )
        if est < best_est_cost:
            best_est_cost = est
            best_plan = p
    assert best_plan is not None
    return best_plan


def select_plan_qaoa(
    scenario: ProspectiveScenario,
    probabilities: Dict[str, float],
    warm_start: bool = False,
    layers: int = 1,
    shots: int = 512,
    seed: int = 7,
) -> Dict[str, Any]:
    """Select plan using QAOA on the risk-conditioned route QUBO."""
    rows = {r.route_id: r for r in scenario.routes}
    active_rids = {rid for p in scenario.plans for rid in p["route_ids"]}
    f_routes = generate_routes(scenario.forecast)
    qubo_candidates = []
    for r in f_routes:
        rid = _route_id(r)
        if rid in rows and rid in active_rids:
            est_cost = (
                rows[rid].forecast_travel_cost * OPERATING_COST_PER_TRAVEL_MINUTE
                + probabilities[rid] * rows[rid].exposure_scale
            )
            qubo_candidates.append({
                "id": rid,
                "vehicle": r["vehicle_id"],
                "stops": r["order_ids"],
                "cost": float(est_cost),
            })

    if len(qubo_candidates) > 12:
        fallback = select_plan_exact(scenario, probabilities)
        return {
            "route_ids": fallback["route_ids"],
            "origin": "fallback_exact",
            "qaoa_feasible_prob": 0.0,
            "fallback_reason": "candidate pool exceeds 12-qubit limit",
        }

    qubo = build_route_qubo(qubo_candidates, [o["id"] for o in scenario.forecast["orders"]])

    res = execute_qaoa_benchmark(
        qubo, layers=layers, warm_start=warm_start, optimizer_method="grid", grid_size=6, shots=shots, seed=seed
    )

    if res["best_feasible_routes"] is not None:
        return {
            "route_ids": res["best_feasible_routes"],
            "origin": "qaoa_sample",
            "qaoa_feasible_prob": res["feasible_sample_probability"],
        }
    else:
        # Classical fallback if no feasible sample was produced by the quantum circuit
        fallback = select_plan_exact(scenario, probabilities)
        return {
            "route_ids": fallback["route_ids"],
            "origin": "fallback_exact",
            "qaoa_feasible_prob": 0.0,
        }


def evaluate_plan_financial(
    realized_scenario: Dict[str, Any],
    route_ids: Sequence[str],
) -> Dict[str, Any]:
    """Score plan on realized scenario with explicit financial metrics."""
    actual_routes = {_route_id(r): r for r in generate_routes(realized_scenario)}
    plan_obj = _materialize_plan(route_ids, actual_routes)

    # Validate plan with independent domain validator
    val = validate_plan(realized_scenario, plan_obj)
    if not val["valid"]:
        return {
            "valid": False,
            "errors": val["errors"],
            "hard_violations": 1,
            "total_financial_cost": float("inf"),
        }

    # Financial breakdown
    travel_minutes = sum(float(r["travel_cost"]) for r in plan_obj["routes"])
    transport_cost = travel_minutes * OPERATING_COST_PER_TRAVEL_MINUTE

    late_orders = 0
    total_orders = len(realized_scenario["orders"])
    late_minutes_list = []
    contractual_penalties = 0.0

    for r in plan_obj["routes"]:
        for stop in r["stops"]:
            lateness = float(stop.get("lateness", 0.0))
            if lateness > 0.0:
                late_orders += 1
                late_minutes_list.append(lateness)
                order_id = stop["order_id"]
                order_rec = next(o for o in realized_scenario["orders"] if o["id"] == order_id)
                contractual_penalties += lateness * float(order_rec["penalty"])

    total_financial_cost = transport_cost + contractual_penalties
    breach_rate = late_orders / total_orders if total_orders > 0 else 0.0
    mean_severity = (
        sum(late_minutes_list) / len(late_minutes_list) if late_minutes_list else 0.0
    )
    max_severity = max(late_minutes_list) if late_minutes_list else 0.0

    return {
        "valid": True,
        "hard_violations": 0,
        "order_coverage_rate": 1.0,
        "travel_minutes": travel_minutes,
        "transport_cost": transport_cost,
        "contractual_penalties": contractual_penalties,
        "total_financial_cost": total_financial_cost,
        "sla_breach_rate": breach_rate,
        "mean_sla_breach_severity_min": mean_severity,
        "max_sla_breach_severity_min": max_severity,
    }


def compute_oracle_cost(scenario: ProspectiveScenario) -> float:
    """Compute realized oracle cost (best feasible plan under actual realization)."""
    r_routes = {_route_id(r): r for r in generate_routes(scenario.realized)}
    best_c = float("inf")
    for p in scenario.plans:
        plan_obj = _materialize_plan(p["route_ids"], r_routes)
        val = validate_plan(scenario.realized, plan_obj)
        if val["valid"]:
            travel = sum(float(r["travel_cost"]) for r in plan_obj["routes"])
            pen = sum(
                float(st.get("lateness", 0.0))
                * next(o["penalty"] for o in scenario.realized["orders"] if o["id"] == st["order_id"])
                for r in plan_obj["routes"]
                for st in r["stops"]
            )
            cost = travel * OPERATING_COST_PER_TRAVEL_MINUTE + pen
            if cost < best_c:
                best_c = cost
    return best_c


# ==============================================================================
# Full 7-Way Comparator Benchmark Runner
# ==============================================================================

def run_comparator_benchmarks(
    records: List[ProspectiveScenario],
    df_temperature: float = 1.0,
    df_learning_rate: float = 0.05,
    qaoa_shots: int = 512,
) -> Dict[str, Any]:
    """Execute evaluation of pipelines A-G across all replicates and test scenarios."""
    replicate_groups: Dict[int, Dict[str, List[ProspectiveScenario]]] = {}
    for sc in records:
        rep_num = int(sc.scenario_id.split("-")[0].replace("rep", ""))
        replicate_groups.setdefault(rep_num, {"train": [], "validation": [], "test": []})[sc.split].append(sc)

    pipeline_names = [
        "A_classical_exact",
        "B_std_vqc_exact",
        "C_df_vqc_exact",
        "D_std_vqc_std_qaoa",
        "E_df_vqc_std_qaoa",
        "F_df_vqc_warm_qaoa",
        "G_classical_warm_qaoa",
    ]

    all_test_evals: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pipeline_names}
    calibration_data: Dict[str, List[Tuple[float, int]]] = {
        "classical": [],
        "std_vqc": [],
        "df_vqc": [],
    }

    replicate_metrics = []

    for rep_num, splits in sorted(replicate_groups.items()):
        train_set = splits["train"]
        test_set = splits["test"]

        # 1. Fit models on train split
        # Model 1: Classical Logistic
        log_weights = train_classical_logistic(train_set, steps=120)

        # Model 2: Standard VQC (predictive-only)
        vqc_pred_theta = train_predictive_vqc(train_set, steps=120, lr=0.08)

        # Model 3: Decision-Focused VQC
        vqc_df_theta = train_decision_focused_vqc(
            train_set,
            steps=120,
            lr=df_learning_rate,
            temperature=df_temperature,
            pretrain_predictive=True,
        )

        # 2. Evaluate on held-out test scenarios
        rep_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pipeline_names}

        for sc in test_set:
            oracle_cost = compute_oracle_cost(sc)

            # Get model probabilities for each route in scenario
            probs_log = {r.route_id: predict_logistic(r.features, log_weights) for r in sc.routes}
            probs_vqc_pred = {r.route_id: risk_probability(r.features, vqc_pred_theta) for r in sc.routes}
            probs_vqc_df = {r.route_id: risk_probability(r.features, vqc_df_theta) for r in sc.routes}

            # Gather calibration data on realized labels
            actual_r = {_route_id(r): r for r in generate_routes(sc.realized)}
            for r in sc.routes:
                if r.route_id in actual_r:
                    true_label = int(any(float(st["lateness"]) > 0 for st in actual_r[r.route_id]["stops"]))
                    calibration_data["classical"].append((probs_log[r.route_id], true_label))
                    calibration_data["std_vqc"].append((probs_vqc_pred[r.route_id], true_label))
                    calibration_data["df_vqc"].append((probs_vqc_df[r.route_id], true_label))

            # Execute the 7 pipelines
            # A: Classical ML + Exact
            p_a = select_plan_exact(sc, probs_log)
            res_a = evaluate_plan_financial(sc.realized, p_a["route_ids"])
            res_a["regret"] = res_a["total_financial_cost"] - oracle_cost
            rep_results["A_classical_exact"].append(res_a)

            # B: Standard VQC + Exact
            p_b = select_plan_exact(sc, probs_vqc_pred)
            res_b = evaluate_plan_financial(sc.realized, p_b["route_ids"])
            res_b["regret"] = res_b["total_financial_cost"] - oracle_cost
            rep_results["B_std_vqc_exact"].append(res_b)

            # C: Decision-focused VQC + Exact
            p_c = select_plan_exact(sc, probs_vqc_df)
            res_c = evaluate_plan_financial(sc.realized, p_c["route_ids"])
            res_c["regret"] = res_c["total_financial_cost"] - oracle_cost
            rep_results["C_df_vqc_exact"].append(res_c)

            # D: Standard VQC + Standard QAOA
            p_d = select_plan_qaoa(sc, probs_vqc_pred, warm_start=False, layers=1, shots=qaoa_shots)
            res_d = evaluate_plan_financial(sc.realized, p_d["route_ids"])
            res_d["regret"] = res_d["total_financial_cost"] - oracle_cost
            res_d["qaoa_origin"] = p_d.get("origin")
            rep_results["D_std_vqc_std_qaoa"].append(res_d)

            # E: Decision-focused VQC + Standard QAOA
            p_e = select_plan_qaoa(sc, probs_vqc_df, warm_start=False, layers=1, shots=qaoa_shots)
            res_e = evaluate_plan_financial(sc.realized, p_e["route_ids"])
            res_e["regret"] = res_e["total_financial_cost"] - oracle_cost
            res_e["qaoa_origin"] = p_e.get("origin")
            rep_results["E_df_vqc_std_qaoa"].append(res_e)

            # F: Decision-focused VQC + Warm-start QAOA
            p_f = select_plan_qaoa(sc, probs_vqc_df, warm_start=True, layers=1, shots=qaoa_shots)
            res_f = evaluate_plan_financial(sc.realized, p_f["route_ids"])
            res_f["regret"] = res_f["total_financial_cost"] - oracle_cost
            res_f["qaoa_origin"] = p_f.get("origin")
            rep_results["F_df_vqc_warm_qaoa"].append(res_f)

            # G: Classical ML + Warm-start QAOA
            p_g = select_plan_qaoa(sc, probs_log, warm_start=True, layers=1, shots=qaoa_shots)
            res_g = evaluate_plan_financial(sc.realized, p_g["route_ids"])
            res_g["regret"] = res_g["total_financial_cost"] - oracle_cost
            res_g["qaoa_origin"] = p_g.get("origin")
            rep_results["G_classical_warm_qaoa"].append(res_g)

        for p in pipeline_names:
            all_test_evals[p].extend(rep_results[p])

        replicate_metrics.append({
            "replicate": rep_num,
            "test_scenarios": len(test_set),
            "means": {
                p: {
                    "total_financial_cost": sum(r["total_financial_cost"] for r in rep_results[p]) / len(rep_results[p]),
                    "regret": sum(r["regret"] for r in rep_results[p]) / len(rep_results[p]),
                    "sla_breach_rate": sum(r["sla_breach_rate"] for r in rep_results[p]) / len(rep_results[p]),
                }
                for p in pipeline_names
            },
        })

    # Aggregate held-out statistics across all 30 test scenarios
    summary_table = {}
    for p in pipeline_names:
        rows = all_test_evals[p]
        n_sc = len(rows)
        mean_cost = sum(r["total_financial_cost"] for r in rows) / n_sc
        mean_regret = sum(r["regret"] for r in rows) / n_sc
        mean_breach = sum(r["sla_breach_rate"] for r in rows) / n_sc
        mean_severity = sum(r["mean_sla_breach_severity_min"] for r in rows) / n_sc
        valid_rate = sum(1 for r in rows if r["valid"]) / n_sc
        hard_viol = sum(r["hard_violations"] for r in rows)
        transport = sum(r["transport_cost"] for r in rows) / n_sc
        penalties = sum(r["contractual_penalties"] for r in rows) / n_sc

        # Sample standard deviation of regret across scenarios
        var_regret = sum((r["regret"] - mean_regret) ** 2 for r in rows) / max(1, n_sc - 1)
        sd_regret = math.sqrt(var_regret)

        summary_table[p] = {
            "mean_total_financial_cost_usd": round(mean_cost, 4),
            "mean_transport_cost_usd": round(transport, 4),
            "mean_contractual_penalties_usd": round(penalties, 4),
            "mean_decision_regret_usd": round(mean_regret, 4),
            "regret_sd": round(sd_regret, 4),
            "sla_breach_rate": round(mean_breach, 4),
            "mean_sla_breach_severity_min": round(mean_severity, 4),
            "valid_plan_rate": valid_rate,
            "hard_constraint_violations": hard_viol,
            "n_scenarios": n_sc,
        }

    # Calibration statistics: Brier Score and Expected Calibration Error (ECE)
    calibration_metrics = {}
    for model_name, pairs in calibration_data.items():
        brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
        # 10-bin ECE
        n_bins = 10
        bins = [[] for _ in range(n_bins)]
        for p, y in pairs:
            b_idx = min(n_bins - 1, int(p * n_bins))
            bins[b_idx].append((p, y))
        ece = 0.0
        for b in bins:
            if b:
                avg_p = sum(p for p, _ in b) / len(b)
                avg_y = sum(y for _, y in b) / len(b)
                ece += (len(b) / len(pairs)) * abs(avg_p - avg_y)
        calibration_metrics[model_name] = {
            "brier_score": round(brier, 5),
            "expected_calibration_error_ece": round(ece, 5),
            "evaluated_routes": len(pairs),
        }

    return {
        "summary_table": summary_table,
        "calibration_metrics": calibration_metrics,
        "replicates": replicate_metrics,
    }
