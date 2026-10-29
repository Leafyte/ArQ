"""Unit and regression tests for V1.1 Quantum Optimization and Decision Quality modules."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.domain import generate_routes
from backend.qubo import build_route_qubo
from experiments_v1_1.decision_quality import (
    _base_scenario,
    build_experiment_records,
    evaluate_plan_financial,
    predict_logistic,
    risk_probability,
    run_comparator_benchmarks,
    select_plan_exact,
    train_classical_logistic,
    train_decision_focused_vqc,
    train_predictive_vqc,
)
from experiments_v1_1.mathematical_verification import verify_qubo_isomorphism
from experiments_v1_1.quantum_optimization import (
    execute_qaoa_benchmark,
    get_circuit_profile,
    solve_classical_exact,
    solve_classical_heuristic_sampler,
)


@pytest.fixture
def tiny_qubo():
    candidates = [
        {"id": "r0", "vehicle": "A", "stops": ["a"], "cost": 2.0},
        {"id": "r1", "vehicle": "A", "stops": ["a", "b"], "cost": 3.0},
        {"id": "r2", "vehicle": "B", "stops": ["b"], "cost": 2.0},
        {"id": "r3", "vehicle": "C", "stops": ["a", "b"], "cost": 6.0},
    ]
    return build_route_qubo(candidates, ["a", "b"], coverage_penalty=20.0, vehicle_penalty=15.0)


@pytest.fixture
def dispatch_qubo():
    fix_path = Path(__file__).resolve().parents[1] / "experiments_qaoa" / "dispatch_fixture.json"
    scenario = json.loads(fix_path.read_text(encoding="utf-8"))
    routes = generate_routes(scenario)
    candidates = [
        {
            "id": f"{r['vehicle_id']}:{'>'.join(r['order_ids'])}",
            "vehicle": r["vehicle_id"],
            "stops": r["order_ids"],
            "cost": float(r["travel_cost"]),
        }
        for r in routes
    ]
    return build_route_qubo(candidates, [o["id"] for o in scenario["orders"]])


def test_mathematical_isomorphism_zero_discrepancy(tiny_qubo):
    res = verify_qubo_isomorphism(tiny_qubo)
    assert res["max_qubo_direct_discrepancy"] == pytest.approx(0.0)
    assert res["max_ising_direct_discrepancy"] == pytest.approx(0.0)
    assert res["ground_state_is_feasible"] is True
    assert res["penalty_margin_gap"] > 0.0
    assert res["best_feasible_bitstring"] == "0100"
    assert res["best_feasible_cost"] == 3.0


def test_circuit_profile_gate_counts(dispatch_qubo):
    p1_std = get_circuit_profile(dispatch_qubo, layers=1, warm_start=False)
    assert p1_std.qubits == 8
    assert p1_std.depth_layers == 1
    assert p1_std.single_qubit_gates > 0
    assert p1_std.two_qubit_gates > 0
    assert p1_std.total_gates == p1_std.single_qubit_gates + p1_std.two_qubit_gates

    p2_warm = get_circuit_profile(dispatch_qubo, layers=2, warm_start=True)
    assert p2_warm.qubits == 8
    assert p2_warm.depth_layers == 2
    assert p2_warm.total_gates > p1_std.total_gates


def test_classical_exact_solver(dispatch_qubo):
    exact = solve_classical_exact(dispatch_qubo)
    assert exact["feasible"] is True
    assert exact["best_objective"] == pytest.approx(4.0)
    assert exact["cpu_seconds"] >= 0.0
    assert exact["peak_memory_kb"] > 0.0


def test_qaoa_standard_and_warm_start_execution(dispatch_qubo):
    # Test standard depth 1
    std_res = execute_qaoa_benchmark(dispatch_qubo, layers=1, warm_start=False, shots=128, seed=42)
    assert std_res["shots"] == 128
    assert 0.0 <= std_res["feasible_sample_probability"] <= 1.0
    assert "preprocessing_wall_seconds" in std_res["resource_accounting"]
    assert "sampling_cpu_seconds" in std_res["resource_accounting"]

    # Test warm start depth 2
    warm_res = execute_qaoa_benchmark(dispatch_qubo, layers=2, warm_start=True, shots=128, seed=42)
    assert warm_res["layers"] == 2
    assert warm_res["warm_start"] is True
    assert warm_res["feasible_sample_probability"] > 0.0
    assert warm_res["best_feasible_objective"] == pytest.approx(4.0)


def test_decision_focused_training_and_pipeline_evaluation():
    records = build_experiment_records(n_replicates=1, cases_per_replicate=6, family_seed=42, split_seed=43)
    train_set = [r for r in records if r.split == "train"]
    test_set = [r for r in records if r.split == "test"]

    assert len(train_set) > 0
    assert len(test_set) > 0

    theta_df = train_decision_focused_vqc(train_set, steps=10, lr=0.05, temperature=1.0)
    assert len(theta_df) == 2

    sc = test_set[0]
    probs = {r.route_id: risk_probability(r.features, theta_df) for r in sc.routes}
    selected_plan = select_plan_exact(sc, probs)
    eval_res = evaluate_plan_financial(sc.realized, selected_plan["route_ids"])

    assert eval_res["valid"] is True
    assert eval_res["hard_violations"] == 0
    assert eval_res["order_coverage_rate"] == 1.0
    assert eval_res["total_financial_cost"] > 0.0
