import math

import pytest

from backend.qubo import build_route_qubo, decode_bitstring, enumerate_qubo, ising_energy, qubo_to_ising, validate_selection
from backend.qaoa import check_ising_equivalence, continuous_relaxation, qaoa_state, run_qaoa
from backend.qpiai_adapter import run_qpiai_warm_start


@pytest.fixture
def tiny():
    # Two complete choices; r0/r1 conflict on vehicle A and overlap stop a.
    candidates = [
        {"id": "r0", "vehicle": "A", "stops": ["a"], "cost": 2.0},
        {"id": "r1", "vehicle": "A", "stops": ["a", "b"], "cost": 3.0},
        {"id": "r2", "vehicle": "B", "stops": ["b"], "cost": 2.0},
        {"id": "r3", "vehicle": "C", "stops": ["a", "b"], "cost": 6.0},
    ]
    return build_route_qubo(candidates, ["a", "b"], coverage_penalty=20, vehicle_penalty=15)


def test_exhaustive_qubo_matches_direct_business_penalties_and_finds_optimum(tiny):
    rows = enumerate_qubo(tiny)
    assert len(rows) == 16
    best = min((r for r in rows if r["feasible"]), key=lambda r: r["business_cost"])
    assert best["selected_route_ids"] == ["r1"]
    assert best["business_cost"] == 3


def test_ising_offset_equivalence_for_every_assignment(tiny):
    offset, h, j = qubo_to_ising(tiny)
    assert math.isfinite(offset) and len(h) == tiny.n
    assert check_ising_equivalence(tiny)
    for row in enumerate_qubo(tiny):
        bits = [int(b) for b in row["bits"]]
        assert ising_energy(tiny, bits) == pytest.approx(row["energy"])


def test_validator_rejects_duplicate_vehicle_even_with_full_coverage(tiny):
    duplicate_fixture = build_route_qubo([
        {"id": "a", "vehicle": "A", "stops": ["a"], "cost": 1},
        {"id": "b", "vehicle": "A", "stops": ["b"], "cost": 1},
    ], ["a", "b"])
    report = validate_selection(duplicate_fixture, [1, 1])
    assert report["coverage_ok"]
    assert not report["vehicle_ok"]
    assert report["duplicate_vehicles"] == ["A"]


def test_decode_preserves_variable_order_and_rejects_malformed_bits(tiny):
    decoded = decode_bitstring(tiny, "0100")
    assert decoded["variable_order"] == ["r0", "r1", "r2", "r3"]
    assert decoded["validation"]["selected_route_ids"] == ["r1"]
    with pytest.raises(ValueError):
        decode_bitstring(tiny, "01x0")


def test_relaxation_and_qaoa_preserve_counts_and_raw_repair_separation(tiny):
    relaxed = continuous_relaxation(tiny)
    assert len(relaxed) == tiny.n and all(0 <= p <= 1 for p in relaxed)
    standard = run_qaoa(tiny, seed=91, shots=64, warm_start=False, angle_grid=4)
    warm = run_qaoa(tiny, seed=91, shots=64, warm_start=True, relaxation=relaxed, angle_grid=4)
    assert standard.backend == warm.backend == "local_python_statevector_simulator"
    assert standard.shots == warm.shots == 64
    assert sum(standard.raw_counts.values()) == sum(warm.raw_counts.values()) == 64
    assert len(standard.raw_samples) == 64
    assert all("validation" in row and "repair" in row for row in warm.raw_samples)
    assert warm.relaxation["vector"] == relaxed
    assert warm.raw_samples[0]["bits"] in warm.raw_counts


def test_statevector_is_normalized_and_invalid_input_fails(tiny):
    state = qaoa_state(tiny, [0.4], [0.7])
    assert sum(abs(a) ** 2 for a in state) == pytest.approx(1)
    with pytest.raises(ValueError):
        run_qaoa(tiny, shots=0)
    with pytest.raises(ValueError):
        build_route_qubo([{"id": "x", "vehicle": "v", "stops": ["unknown"], "cost": 1}], ["a"])


def test_tilted_mixer_keeps_its_relaxation_state_when_cost_angle_is_zero():
    single = build_route_qubo([{"id": "x", "vehicle": "v", "stops": ["a"], "cost": 1}], ["a"])
    probability, beta = 0.25, 0.37
    state = qaoa_state(single, [0.0], [beta], [probability])
    assert abs(state[0]) ** 2 == pytest.approx(1 - probability)
    assert abs(state[1]) ** 2 == pytest.approx(probability)
    # <psi| (2sqrt(p(1-p)) X + (1-2p) Z) |psi> = 1 analytically.
    mixer_expectation = 4 * probability * (1-probability) + (1-2*probability) ** 2
    assert mixer_expectation == pytest.approx(1.0)


def test_cost_phase_uses_complete_ising_offset_at_amplitude_level():
    from cmath import exp

    one = build_route_qubo([{"id": "x", "vehicle": "v", "stops": ["a"], "cost": 2}], ["a"])
    gamma = 0.41
    state = qaoa_state(one, [gamma], [0.0])
    expected = [2 ** -0.5 * exp(-1j * gamma * one.energy([bit])) for bit in (0, 1)]
    assert state == pytest.approx(expected)


def test_fractional_values_are_not_truncated_to_binary(tiny):
    with pytest.raises(ValueError):
        validate_selection(tiny, [0.5, 0, 0, 0])
    with pytest.raises(ValueError):
        tiny.energy([0.5, 0, 0, 0])


def test_qaoa_resource_budgets_are_bounded(tiny):
    with pytest.raises(ValueError):
        run_qaoa(tiny, shots=10_001)
    with pytest.raises(ValueError):
        run_qaoa(tiny, angle_grid=17)


def test_qpiai_adapter_refuses_unverified_variable_order_above_eight():
    q = build_route_qubo([{"id": f"r{i}", "vehicle": f"v{i}", "stops": ["a"], "cost": 1}
                          for i in range(9)], ["a"])
    result = run_qpiai_warm_start(q, [0.5] * q.n)
    assert result["status"] == "unsupported"
    assert result["error"]["code"] == "UNSUPPORTED_CAPABILITY"
    assert result["error"]["verified_variable_limit"] == 8
