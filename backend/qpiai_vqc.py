"""QpiAI Quantum local-statevector execution for the tiny decision VQC.

This module deliberately uses only QpiAI Quantum's local StatevectorSimulator.
It does not create a QpiAI Cloud client or submit a remote job.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import math
from time import perf_counter
from typing import Sequence


def _validate(features: Sequence[float], theta: Sequence[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(features) != 2 or len(theta) != 2:
        raise ValueError("expected two features and two trainable angles")
    x = tuple(float(v) for v in features)
    angles = tuple(float(v) for v in theta)
    if any(not math.isfinite(v) for v in (*x, *angles)):
        raise ValueError("circuit inputs must be finite")
    if any(v < 0.0 or v > 1.0 for v in x):
        raise ValueError("features must lie in [0, 1]")
    return x, angles  # type: ignore[return-value]


def sdk_statevector(features: Sequence[float], theta: Sequence[float]) -> tuple[complex, ...]:
    """Run the identical 2-qubit circuit and return SDK statevector amplitudes."""
    x, angles = _validate(features, theta)
    try:
        from qpiai_quantum import Circuit
        from qpiai_quantum.simulator import StatevectorSimulator
    except ImportError as exc:
        raise RuntimeError("QpiAI Quantum SDK is unavailable in this environment") from exc

    circuit = Circuit(2, name="DebugLeafDecisionVQC")
    circuit.ry(0, math.pi * x[0])
    circuit.ry(1, math.pi * x[1])
    circuit.cx(0, 1)
    circuit.ry(0, angles[0])
    circuit.ry(1, angles[1])
    circuit.cx(1, 0)
    result = StatevectorSimulator().run(circuit, shots=1, seed=0)
    raw = getattr(result, "statevector", None)
    if raw is None or len(raw) != 4:
        raise RuntimeError("QpiAI StatevectorSimulator did not return the expected four amplitudes")
    state = tuple(complex(value) for value in raw)
    norm = sum(abs(value) ** 2 for value in state)
    if abs(norm - 1.0) > 1e-10:
        raise RuntimeError("QpiAI statevector is not normalized")
    return state


def sdk_expectation_z0(features: Sequence[float], theta: Sequence[float]) -> float:
    """Read <Z0> from the SDK statevector (SDK wire 0 is the low bit)."""
    state = sdk_statevector(features, theta)
    value = sum((1.0 if index & 1 == 0 else -1.0) * abs(amplitude) ** 2
                for index, amplitude in enumerate(state))
    return float(value)


def sdk_risk_probability(features: Sequence[float], theta: Sequence[float]) -> float:
    z0 = sdk_expectation_z0(features, theta)
    argument = -2.0 * z0
    if argument >= 0:
        return 1.0 / (1.0 + math.exp(-min(argument, 700.0)))
    e = math.exp(max(argument, -700.0))
    return e / (1.0 + e)


def sdk_versions() -> dict[str, str]:
    try:
        return {"qpiai-quantum": version("qpiai-quantum")}
    except PackageNotFoundError:
        return {"qpiai-quantum": "unavailable"}


def timed_sdk_expectation(features: Sequence[float], theta: Sequence[float]) -> tuple[float, float]:
    started = perf_counter()
    result = sdk_expectation_z0(features, theta)
    return result, perf_counter() - started
