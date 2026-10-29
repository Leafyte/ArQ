"""Dependency-free tiny statevector QAOA with standard and tilted mixers.

This is a local reference simulator, not a QpiAI adapter or QPU call. Keep
instances small (normally <= 8 qubits); the hard safety cap is 12 qubits.
"""
from __future__ import annotations

from cmath import exp
from dataclasses import dataclass
from itertools import product
from math import cos, pi, sin, sqrt
from random import Random
from typing import Sequence

from .qubo import Qubo, enumerate_qubo, ising_energy, qubo_to_ising, validate_selection


def continuous_relaxation(qubo: Qubo, *, steps: int = 4000, step_size: float = 0.001) -> list[float]:
    """Projected gradient descent on the QUBO polynomial over the unit box."""
    n = qubo.n
    if n > 24:
        raise ValueError("reference continuous relaxation is bounded to 24 variables")
    x = [0.5] * n
    for _ in range(steps):
        grad = list(qubo.linear)
        for (i, j), value in qubo.quadratic.items():
            grad[i] += value * x[j]
            grad[j] += value * x[i]
        nxt = [min(1.0, max(0.0, x[i] - step_size * grad[i])) for i in range(n)]
        if max(abs(nxt[i] - x[i]) for i in range(n)) < 1e-10:
            x = nxt
            break
        x = nxt
    return x


def _energies(qubo: Qubo) -> list[float]:
    return [qubo.energy(bits) for bits in product((0, 1), repeat=qubo.n)]


def _apply_single(state: list[complex], gate: tuple[tuple[complex, complex], tuple[complex, complex]], qubit: int, n: int) -> list[complex]:
    stride = 1 << (n - 1 - qubit)
    output = state.copy()
    for base in range(0, len(state), 2 * stride):
        for offset in range(stride):
            i, j = base + offset, base + offset + stride
            a, b = state[i], state[j]
            output[i] = gate[0][0] * a + gate[0][1] * b
            output[j] = gate[1][0] * a + gate[1][1] * b
    return output


def qaoa_state(qubo: Qubo, gammas: Sequence[float], betas: Sequence[float],
               probabilities: Sequence[float] | None = None) -> list[complex]:
    """Simulate p-layer QAOA using full QUBO Ising energy including offset."""
    n, p = qubo.n, len(gammas)
    if n > 12:
        raise ValueError("reference statevector is bounded to 12 qubits")
    if p == 0 or len(betas) != p:
        raise ValueError("gammas and betas must have the same positive length")
    warm = probabilities is not None
    probs = [0.5] * n if probabilities is None else [float(v) for v in probabilities]
    if len(probs) != n or any(not 0 <= v <= 1 for v in probs):
        raise ValueError("warm-start probability vector must match variables and lie in [0,1]")
    state = [1.0 + 0.0j] * (1 << n)
    for q, probability in enumerate(probs):
        state = _apply_single(state, ((sqrt(1-probability), 0j), (0j, sqrt(probability))), q, n)
    ising_offset, h, j = qubo_to_ising(qubo)
    ising = []
    for basis in range(1 << n):
        z = [1 - 2 * ((basis >> (n-1-q)) & 1) for q in range(n)]
        ising.append(ising_offset + sum(h[q] * z[q] for q in range(n)) +
                     sum(v * z[i] * z[k] for (i, k), v in j.items()))
    for gamma, beta in zip(gammas, betas):
        state = [amp * exp(-1j * gamma * energy) for amp, energy in zip(state, ising)]
        for q, probability in enumerate(probs):
            if warm:
                # The prepared one-qubit state is the +1 eigenstate of this axis.
                nx, nz = 2 * sqrt(probability * (1-probability)), 1 - 2 * probability
            else:
                nx, nz = 1.0, 0.0
            axis = ((nz, nx), (nx, -nz))
            mixer = ((cos(beta) - 1j*sin(beta)*axis[0][0], -1j*sin(beta)*axis[0][1]),
                     (-1j*sin(beta)*axis[1][0], cos(beta) - 1j*sin(beta)*axis[1][1]))
            state = _apply_single(state, mixer, q, n)
    return state


def _expectation(qubo: Qubo, state: Sequence[complex]) -> float:
    return sum((amp.real**2 + amp.imag**2) * energy for amp, energy in zip(state, _energies(qubo)))


@dataclass
class QAOAResult:
    mode: str
    backend: str
    probabilities: list[float]
    angles: dict[str, list[float]]
    expected_qubo_energy: float
    raw_counts: dict[str, int]
    raw_samples: list[dict]
    feasible_samples: int
    shots: int
    repair_edits: list[dict]
    relaxation: dict


def _greedy_repair(qubo: Qubo, bits: list[int]) -> tuple[list[int], list[dict]]:
    """Deterministic repair; raw samples and edits stay separately recorded."""
    chosen = {i for i, bit in enumerate(bits) if bit}
    edits: list[dict] = []
    for vehicle in sorted({c["vehicle"] for c in qubo.candidates}):
        group = [i for i in chosen if qubo.candidates[i]["vehicle"] == vehicle]
        if len(group) > 1:
            keep = min(group, key=lambda i: (qubo.candidates[i]["cost"], i))
            for i in group:
                if i != keep:
                    chosen.remove(i)
                    edits.append({"index": i, "from": 1, "to": 0, "reason": "duplicate_vehicle"})
    for stop in qubo.required_stops:
        if any(stop in qubo.candidates[i]["stops"] for i in chosen):
            continue
        occupied = {qubo.candidates[k]["vehicle"] for k in chosen}
        available = [i for i, c in enumerate(qubo.candidates) if stop in c["stops"] and c["vehicle"] not in occupied]
        if available:
            i = min(available, key=lambda k: (qubo.candidates[k]["cost"], k))
            chosen.add(i)
            edits.append({"index": i, "from": 0, "to": 1, "reason": "uncovered_stop"})
    return [int(i in chosen) for i in range(qubo.n)], edits


def run_qaoa(qubo: Qubo, *, seed: int = 7, shots: int = 512, layers: int = 1,
             warm_start: bool = False, relaxation: Sequence[float] | None = None,
             angle_grid: int = 10, repair: bool = True) -> QAOAResult:
    """Run matched angle-grid search and finite-shot sampling locally."""
    if (isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 10_000 or
            isinstance(angle_grid, bool) or not isinstance(angle_grid, int) or not 2 <= angle_grid <= 16):
        raise ValueError("shots must be in [1, 10000] and angle_grid in [2, 16]")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers != 1:
        raise ValueError("the bounded reference currently supports one QAOA layer")
    if warm_start:
        if relaxation is None:
            relaxation = continuous_relaxation(qubo)
        if len(relaxation) != qubo.n:
            raise ValueError("relaxation vector must match the QUBO variable count")
        probabilities = [min(0.95, max(0.05, float(v))) for v in relaxation]
    else:
        probabilities = [0.5] * qubo.n
    angles = [2*pi*i/angle_grid for i in range(angle_grid)]
    best: tuple[float, float, float] | None = None
    for gamma in angles:
        for beta in angles:
            state = qaoa_state(qubo, [gamma], [beta], probabilities if warm_start else None)
            item = (_expectation(qubo, state), gamma, beta)
            if best is None or item[0] < best[0]:
                best = item
    assert best is not None
    expected, gamma, beta = best
    state = qaoa_state(qubo, [gamma], [beta], probabilities if warm_start else None)
    rng = Random(seed)
    cumulative: list[float] = []
    total = 0.0
    for amp in state:
        total += amp.real**2 + amp.imag**2
        cumulative.append(total)
    sampled = []
    for _ in range(shots):
        draw = rng.random()
        sampled.append(next((i for i, cutoff in enumerate(cumulative) if draw < cutoff), len(cumulative)-1))
    counts: dict[str, int] = {}
    samples: list[dict] = []
    feasible = 0
    edits_for_best: list[dict] = []
    for index in sampled:
        bits = [int(v) for v in format(index, f"0{qubo.n}b")]
        key = "".join(map(str, bits))
        counts[key] = counts.get(key, 0) + 1
        validation = validate_selection(qubo, bits)
        feasible += int(validation["feasible"])
        row = {"bits": key, "energy": qubo.energy(bits), "validation": validation}
        if repair:
            fixed, edits = _greedy_repair(qubo, bits)
            row["repair"] = {"bits": "".join(map(str, fixed)), "edits": edits,
                             "validation": validate_selection(qubo, fixed)}
            if not edits_for_best and edits:
                edits_for_best = edits
        else:
            row["repair"] = None
        samples.append(row)
    return QAOAResult(
        mode="warm_start" if warm_start else "standard", backend="local_python_statevector_simulator",
        probabilities=probabilities, angles={"gamma": [gamma], "beta": [beta]},
        expected_qubo_energy=expected, raw_counts=counts, raw_samples=samples,
        feasible_samples=feasible, shots=shots, repair_edits=edits_for_best,
        relaxation={"method": "projected_gradient_continuous_box_relaxation" if warm_start else "not_used",
                    "vector": list(relaxation) if warm_start and relaxation is not None else None,
                    "clamping": "[0.05, 0.95]" if warm_start else "not_used"},
    )


def check_ising_equivalence(qubo: Qubo) -> bool:
    return all(abs(qubo.energy(bits) - ising_energy(qubo, bits)) <= 1e-8
               for bits in product((0, 1), repeat=qubo.n))


def exhaustive_optimum(qubo: Qubo) -> dict:
    feasible = [r for r in enumerate_qubo(qubo) if r["feasible"]]
    return min(feasible, key=lambda r: r["business_cost"]) if feasible else {"feasible": False}
