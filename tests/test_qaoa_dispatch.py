import json
from pathlib import Path

import pytest

import backend.qaoa_dispatch as dispatch_module
from backend.qaoa import QAOAResult
from backend.qubo import validate_selection
from backend.qaoa_dispatch import solve_qaoa_dispatch


@pytest.fixture
def small_scenario():
    return json.loads((Path(__file__).parents[1] / "experiments_qaoa" / "dispatch_fixture.json").read_text())


def test_qaoa_candidates_are_decoded_and_domain_validated(small_scenario):
    result = solve_qaoa_dispatch(small_scenario, seed=23, shots=512, angle_grid=5)
    assert result["status"] == "success"
    assert result["decision"]["validation"]["valid"]
    raw = result["optimization"]["raw_result"]
    assert raw is not None
    assert sum(raw["raw_counts"].values()) == 512
    assert raw["backend"] == "local_python_statevector_simulator"
    assert all("selected_route_ids" in sample["decoded"] for sample in raw["raw_samples"])
    assert all("independent_plan_validation" in sample for sample in raw["raw_samples"])
    fallback = result["fallback"]["used"]
    assert (result["decision"]["origin"] == "classical_exact_fallback") == fallback
    if not fallback:
        assert result["decision"]["origin"] == "qaoa_raw_sample"
        assert result["optimization"]["selected_raw_bits"] is not None


def test_unsupported_candidate_pool_uses_explicit_validated_classical_fallback():
    scenario = json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())
    scenario["vehicles"].append({"id": "V3", "capacity": 3})
    result = solve_qaoa_dispatch(scenario, seed=23, shots=64, angle_grid=4)
    assert result["status"] == "success"
    assert result["decision"]["origin"] == "classical_exact_fallback"
    assert result["fallback"]["used"]
    assert result["optimization"]["qaoa_failure"]["code"] == "UNSUPPORTED_CAPABILITY"
    assert result["decision"]["validation"]["valid"]


def test_invalid_qaoa_budgets_are_boundary_errors(small_scenario):
    result = solve_qaoa_dispatch(small_scenario, shots=10_001)
    assert result["status"] == "invalid_input"
    assert not result["fallback"]["used"]


def test_no_feasible_raw_samples_returns_classical_label_and_preserves_attempt(small_scenario, monkeypatch):
    def no_feasible_samples(qubo, **kwargs):
        bits = "0" * qubo.n
        vector = [0] * qubo.n
        return QAOAResult(
            mode="warm_start", backend="local_python_statevector_simulator", probabilities=[0.5] * qubo.n,
            angles={"gamma": [0.0], "beta": [0.0]}, expected_qubo_energy=qubo.energy(vector),
            raw_counts={bits: kwargs["shots"]},
            raw_samples=[{"bits": bits, "energy": qubo.energy(vector),
                          "validation": validate_selection(qubo, vector), "repair": None}],
            feasible_samples=0, shots=kwargs["shots"], repair_edits=[],
            relaxation={"method": "projected_gradient_continuous_box_relaxation", "vector": None},
        )

    monkeypatch.setattr(dispatch_module, "run_qaoa", no_feasible_samples)
    result = solve_qaoa_dispatch(small_scenario, shots=8, angle_grid=2)
    assert result["status"] == "success"
    assert result["execution_type"] == "classical"
    assert result["optimization"]["execution_type"] == "local_python_statevector_simulator"
    assert result["optimization"]["raw_result"]["feasible_samples"] == 0
    assert result["decision"]["origin"] == "classical_exact_fallback"
    assert result["fallback"]["used"]
    assert result["decision"]["validation"]["valid"]
