import json
from pathlib import Path

import pytest

from backend.domain import SolveBudgetExceeded, generate_routes, solve_exact, validate_scenario
from backend.validator import validate_plan


@pytest.fixture
def scenario():
    return json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())


def test_tiny_scenario_valid_and_exact_solver_returns_independently_validated_plan(scenario):
    assert validate_scenario(scenario) == {"valid": True, "errors": []}
    routes = generate_routes(scenario)
    assert routes
    plan = solve_exact(scenario)
    report = validate_plan(scenario, plan)
    assert plan["status"] == "feasible"
    assert report["valid"]
    assert report["recomputed"]["objective"] == pytest.approx(plan["objective"])
    assert sorted(oid for r in plan["routes"] for oid in r["order_ids"]) == ["A", "B", "C"]


def test_service_waits_until_window_opens_and_lateness_is_explicit(scenario):
    # A route D->B->A->D: B starts at 1 and departs at 2; A arrives 3.
    route = next(r for r in generate_routes(scenario) if r["vehicle_id"] == "V1" and r["order_ids"] == ["B", "A"])
    assert route["stops"][1]["arrival"] == pytest.approx(3)
    assert route["stops"][1]["lateness"] == pytest.approx(0)
    assert route["lateness_exposure"] == pytest.approx(0)


def test_capacity_pruned_route_enumeration_preserves_permutation_order(scenario):
    scenario["vehicles"] = [{"id": "V", "capacity": 5}]
    routes = generate_routes(scenario)
    pairs = [tuple(route["order_ids"]) for route in routes if len(route["order_ids"]) == 2]
    assert pairs == [("A", "B"), ("A", "C"), ("B", "A"),
                     ("B", "C"), ("C", "A"), ("C", "B")]


def test_lateness_exposure_uses_per_time_unit_order_penalty(scenario):
    # Force the single order case: D->A takes 2, SLA closes at 1,
    # so lateness is 1 and exposure is 1 * 4 = 4.
    scenario["orders"] = [scenario["orders"][0]]
    scenario["vehicles"] = [scenario["vehicles"][0]]
    scenario["orders"][0]["window"] = [0, 1]
    plan = solve_exact(scenario)
    route = plan["routes"][0]
    assert route["stops"][0]["lateness"] == pytest.approx(1)
    assert route["lateness_exposure"] == pytest.approx(4)
    assert route["objective"] == pytest.approx(route["travel_cost"] + 4)
    assert validate_plan(scenario, plan)["valid"]


def test_hard_window_excludes_late_routes_and_independent_validator_rejects_them(scenario):
    scenario["orders"] = [scenario["orders"][0]]
    scenario["vehicles"] = [scenario["vehicles"][0]]
    scenario["orders"][0]["window"] = [0, 1]
    scenario["travel"] = {"D": {"D": 0, "A": 2}, "A": {"D": 0, "A": 0}}

    # Missing or explicit false preserves legacy soft-window behavior.
    soft_plan = solve_exact(scenario)
    assert soft_plan["status"] == "feasible"
    assert soft_plan["routes"][0]["stops"][0]["lateness"] == pytest.approx(1)
    assert soft_plan["routes"][0]["lateness_exposure"] == pytest.approx(4)
    assert validate_plan(scenario, soft_plan)["valid"]
    scenario["orders"][0]["hard_window"] = False
    assert generate_routes(scenario)
    assert solve_exact(scenario)["status"] == "feasible"

    scenario["orders"][0]["hard_window"] = True
    assert validate_scenario(scenario)["valid"]
    assert generate_routes(scenario) == []
    hard_plan = solve_exact(scenario)
    assert hard_plan["status"] == "infeasible"
    checked = validate_plan(scenario, soft_plan)
    assert not checked["valid"]
    assert any("hard_window" in error for error in checked["errors"])


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_hard_window_requires_a_real_boolean(scenario, value):
    scenario["orders"][0]["hard_window"] = value
    checked = validate_scenario(scenario)
    assert not checked["valid"]
    assert any("hard_window" in error for error in checked["errors"])


def test_solver_reports_infeasible_when_fleet_capacity_cannot_cover_all_orders(scenario):
    scenario["vehicles"] = [{"id": "small", "capacity": 1}]
    plan = solve_exact(scenario)
    assert plan["status"] == "infeasible"
    assert not validate_plan(scenario, plan)["feasible"]


def test_validator_rejects_duplicate_coverage_even_if_cost_fields_are_forged(scenario):
    plan = solve_exact(scenario)
    plan["routes"][0]["order_ids"].append(plan["routes"][0]["order_ids"][0])
    report = validate_plan(scenario, plan)
    assert not report["valid"]
    assert any("coverage." in error for error in report["errors"])


def test_validator_recomputes_and_rejects_forged_objective(scenario):
    plan = solve_exact(scenario)
    plan["objective"] += 100
    report = validate_plan(scenario, plan)
    assert not report["valid"]
    assert any(error.startswith("plan.objective:") for error in report["errors"])


def test_validation_rejects_missing_travel_edge_and_bad_window(scenario):
    del scenario["travel"]["A"]["C"]
    scenario["orders"][0]["window"] = [5, 2]
    report = validate_scenario(scenario)
    assert not report["valid"]
    assert any("window" in error for error in report["errors"])
    assert any("travel.A.C" in error for error in report["errors"])


def _high_branching_assignment_case():
    orders = [f"O{i}" for i in range(8)]
    locations = ["D", *orders]
    travel = {source: {target: (0 if source == target or (source != "D" and target != "D") else 1)
                       for target in locations} for source in locations}
    return {"depot": "D",
            "vehicles": [{"id": f"V{i}", "capacity": 1} for i in range(8)],
            "orders": [{"id": order, "demand": 1, "window": [0, 100], "service": 0, "penalty": 0}
                       for order in orders],
            "travel": travel}


def test_high_branching_exact_cover_exhausts_work_budget_without_partial_plan():
    scenario = _high_branching_assignment_case()
    assert validate_scenario(scenario)["valid"]
    # 8! distinct vehicle/order assignments have equal objective; the search
    # cannot prove optimality before visiting the deterministic work cap.
    with pytest.raises(SolveBudgetExceeded) as raised:
        solve_exact(scenario)
    assert raised.value.code == "solver_budget_exceeded"
