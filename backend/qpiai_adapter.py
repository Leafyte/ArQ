"""Credential-free local QpiAI Opt warm-start circuit execution.

This adapter only models a supplied QUBO, builds one documented QpiAI warm
start circuit, and invokes QpiAI Quantum's local StatevectorSimulator. It never
creates a Client, reads an API key, submits cloud work, or submits a QPU job.
"""
from __future__ import annotations

from importlib.metadata import version
from itertools import product
from math import isfinite
from typing import Any, Sequence

from .qubo import Qubo

MAX_VERIFIED_QPIAI_VARIABLES = 8


def run_qpiai_warm_start(qubo: Qubo, relaxed_solution: Sequence[float], *,
                         gamma: float = 0.15, beta: float = 0.35,
                         shots: int = 128, seed: int = 7) -> dict[str, Any]:
    """Build and sample QpiAI Opt warm-start QAOA for an existing route QUBO.

    The returned `raw_counts` are vendor simulator output. This adapter does
    not decode assignments or assert feasibility; callers must decode using
    the known variable order and run the independent domain validator.
    """
    if qubo.n < 1 or qubo.n > MAX_VERIFIED_QPIAI_VARIABLES:
        return {"status": "unsupported", "execution_type": "not_run",
                "error": {"code": "UNSUPPORTED_CAPABILITY",
                          "reason": "QpiAI variable order and compiled QUBO exhaustively verified only through 8 variables",
                          "verified_variable_limit": MAX_VERIFIED_QPIAI_VARIABLES}}
    if len(relaxed_solution) != qubo.n or any(not isfinite(float(p)) or not 0 <= float(p) <= 1
                                               for p in relaxed_solution):
        return {"status": "invalid_input", "execution_type": "not_run",
                "error": {"code": "INVALID_RELAXATION", "reason": "vector must match QUBO and lie in [0,1]"}}
    if isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 10_000:
        return {"status": "invalid_input", "execution_type": "not_run",
                "error": {"code": "INVALID_BUDGET", "reason": "shots must be in [1,10000]"}}
    try:
        from qpiai_opt.opt_modeller import Model
        from qpiai_opt.opt_modeller.variables import Binary
        from qpiai_opt.opt_solvers.qubo.models import QUBOProblem
        from qpiai_opt.quantum import QAOA, WarmStart, build_circuit
        from qpiai_quantum.simulator import StatevectorSimulator
    except ImportError as exc:
        return {"status": "unsupported", "execution_type": "not_run",
                "error": {"code": "SDK_NOT_INSTALLED", "error_type": type(exc).__name__}}

    try:
        model = Model("DebugLeafRouteSelection")
        variables = [Binary(f"x{i}") for i in range(qubo.n)]
        objective = qubo.offset
        for i, coefficient in enumerate(qubo.linear):
            objective = objective + coefficient * variables[i]
        for (i, j), coefficient in qubo.quadratic.items():
            objective = objective + coefficient * variables[i] * variables[j]
        model.minimize(objective)
        sdk_terms, sdk_offset = model.to_qubo_dict(matrix_format="upper")
        sdk_problem = QUBOProblem.from_model(model)
        if sdk_problem.num_variables != qubo.n:
            return {"status": "failed", "execution_type": "not_run",
                    "error": {"code": "QUBO_VARIABLE_COUNT_MISMATCH", "error_type": "ValueError"},
                    "api_calls": 0, "cloud_jobs": 0, "qpu_jobs": 0}
        for bits in product((0, 1), repeat=qubo.n):
            sdk_energy = float(sdk_offset) + sum(float(value) * bits[int(i)] * bits[int(j)]
                                                  for (i, j), value in sdk_terms.items())
            if abs(sdk_energy - qubo.energy(bits)) > 1e-8:
                return {"status": "failed", "execution_type": "not_run",
                        "error": {"code": "QUBO_MAPPING_MISMATCH", "error_type": "ValueError"},
                        "api_calls": 0, "cloud_jobs": 0, "qpu_jobs": 0}
        warm = WarmStart(epsilon=0.25, relaxed_solution=[float(x) for x in relaxed_solution])
        algorithm = QAOA(reps=1, warm_start=warm)
        circuit = build_circuit(sdk_problem, algorithm=algorithm,
                                parameters=[float(gamma), float(beta)], measurements=False)
        circuit.measure_all()
        simulator = StatevectorSimulator()
        result = simulator.run(circuit, shots=shots, seed=seed)
        raw_counts = getattr(result, "counts", None)
        if callable(raw_counts):
            raw_counts = raw_counts()
        elif raw_counts is None and hasattr(result, "get"):
            data = result.get()
            raw_counts = data.get("counts", {}) if isinstance(data, dict) else {}
        elif raw_counts is None:
            raw_counts = getattr(result, "get_counts", lambda: {})()
        return {
            "status": "success", "execution_type": "local_simulator",
            "sdk_versions": {"qpiai-opt": version("qpiai-opt"), "qpiai-quantum": version("qpiai-quantum")},
            "backend": "QpiAI Quantum StatevectorSimulator",
            "variable_order": list(qubo.variable_ids), "qubits": sdk_problem.num_variables,
            "variable_names_in_sdk_model_order": [f"x{i}" for i in range(qubo.n)],
            "compiled_qubo_validation": {"status": "exhaustive_match",
                                          "bitstrings_checked": 1 << qubo.n,
                                          "sdk_offset": float(sdk_offset),
                                          "sdk_upper_terms": {f"{i},{j}": float(v) for (i, j), v in sdk_terms.items()}},
            "relaxed_solution": [float(x) for x in relaxed_solution],
            "warm_start_epsilon": 0.25,
            "parameters": {"gamma": float(gamma), "beta": float(beta), "reps": 1},
            "shots": shots, "seed": seed,
            "raw_counts": dict(raw_counts or {}),
            "api_calls": 0, "cloud_jobs": 0, "qpu_jobs": 0,
        }
    except Exception as exc:
        # Deliberately omit exception strings; they can contain environment details.
        return {"status": "failed", "execution_type": "local_simulator_attempt",
                "error": {"code": "SDK_LOCAL_EXECUTION_FAILED", "error_type": type(exc).__name__},
                "api_calls": 0, "cloud_jobs": 0, "qpu_jobs": 0}
