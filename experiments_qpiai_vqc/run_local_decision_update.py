"""Run a tiny SDK-backed VQC decision-gradient check and one train update."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.quantum_learning import (  # noqa: E402
    Example, _decision_loss_and_dp, _make_examples, decision_loss_gradient,
    finite_difference_gradient, risk_probability,
)
from backend.qpiai_vqc import (  # noqa: E402
    sdk_expectation_z0, sdk_risk_probability, sdk_statevector, sdk_versions,
)


def sdk_decision_loss(theta: list[float] | tuple[float, float], rows: list[Example]) -> float:
    if not rows:
        raise ValueError("training rows cannot be empty")
    return sum(_decision_loss_and_dp(sdk_risk_probability(row.features, theta), row.label)[0]
               for row in rows) / len(rows)


def sdk_decision_gradient(theta: list[float] | tuple[float, float], rows: list[Example]) -> list[float]:
    """Parameter-shift <Z0>, then chain through sigmoid and decision loss."""
    if not rows:
        raise ValueError("training rows cannot be empty")
    gradient = [0.0, 0.0]
    for row in rows:
        p = sdk_risk_probability(row.features, theta)
        _, dloss_dp = _decision_loss_and_dp(p, row.label)
        for k in range(2):
            plus, minus = list(theta), list(theta)
            plus[k] += 3.141592653589793 / 2
            minus[k] -= 3.141592653589793 / 2
            dz = (sdk_expectation_z0(row.features, plus) - sdk_expectation_z0(row.features, minus)) / 2
            dp = -2.0 * p * (1.0 - p) * dz
            gradient[k] += dloss_dp * dp
    return [value / len(rows) for value in gradient]


def sdk_finite_difference(theta: list[float] | tuple[float, float], rows: list[Example], h: float = 1e-5) -> list[float]:
    if h <= 0:
        raise ValueError("h must be positive")
    answer = []
    for k in range(2):
        plus, minus = list(theta), list(theta)
        plus[k] += h
        minus[k] -= h
        answer.append((sdk_decision_loss(plus, rows) - sdk_decision_loss(minus, rows)) / (2 * h))
    return answer


def run_experiment() -> dict:
    all_rows, _ = _make_examples()
    rows = [row for row in all_rows if row.split == "train"][:4]
    theta = [0.31, -0.47]
    started = perf_counter()
    sample = rows[0]
    sample_state = sdk_statevector(sample.features, theta)
    sample_z = sdk_expectation_z0(sample.features, theta)
    sample_p = sdk_risk_probability(sample.features, theta)
    reference_p = risk_probability(sample.features, theta)
    sdk_gradient = sdk_decision_gradient(theta, rows)
    custom_gradient = decision_loss_gradient(theta, rows)
    finite_difference = sdk_finite_difference(theta, rows, 1e-5)
    initial_loss = sdk_decision_loss(theta, rows)
    learning_rate = 0.08
    updated_theta = [theta[k] - learning_rate * sdk_gradient[k] for k in range(2)]
    updated_loss = sdk_decision_loss(updated_theta, rows)
    elapsed = perf_counter() - started
    errors = {
        "sdk_vs_custom_gradient": [abs(a - b) for a, b in zip(sdk_gradient, custom_gradient)],
        "sdk_vs_finite_difference": [abs(a - b) for a, b in zip(sdk_gradient, finite_difference)],
        "sdk_vs_custom_sample_probability": abs(sample_p - reference_p),
    }
    return {
        "experiment_id": "qpiai-quantum-local-vqc-decision-update-20261028",
        "execution_type": "QpiAI Quantum local StatevectorSimulator",
        "remote_calls": 0,
        "cloud_jobs": 0,
        "qpu_jobs": 0,
        "versions": {"python": sys.version.split()[0], **sdk_versions()},
        "runtime": {"platform": platform.platform(), "wall_seconds": elapsed},
        "circuit": {
            "qubits": 2,
            "gates": ["RY(q0, pi*x0)", "RY(q1, pi*x1)", "CX(q0,q1)",
                      "RY(q0,theta0)", "RY(q1,theta1)", "CX(q1,q0)"],
            "readout": "Z0; SDK statevector index bit 0 is q0",
            "statevector": "exact SDK amplitudes; no sampling shots",
        },
        "training_data": [{"features": list(row.features), "label": row.label,
                            "split": row.split, "source_index": row.index} for row in rows],
        "sample": {"features": list(sample.features), "theta": theta,
                   "sdk_statevector": [{"real": value.real, "imag": value.imag}
                                       for value in sample_state],
                   "sdk_z0": sample_z, "sdk_probability": sample_p,
                   "custom_probability": reference_p},
        "gradient_check": {
            "theta": theta,
            "sdk_parameter_shift_decision_gradient": sdk_gradient,
            "custom_reference_gradient": custom_gradient,
            "sdk_centered_finite_difference_h_1e-5": finite_difference,
            "absolute_errors": errors,
            "tolerance": 2e-7,
            "rows": len(rows),
        },
        "single_training_update": {"optimizer": "one deterministic gradient descent update",
                                   "learning_rate": learning_rate,
                                   "initial_loss": initial_loss,
                                   "initial_theta": theta,
                                   "sdk_gradient": sdk_gradient,
                                   "updated_theta": updated_theta,
                                   "updated_loss": updated_loss,
                                   "training_rows_only": True},
        "limitations": ["synthetic deterministic training rows only; no held-out evaluation",
                         "exact local simulator, not QpiAI Cloud or QPU",
                         "no operational data, calibration, advantage, or deployment claim"],
    }


def main() -> None:
    result = run_experiment()
    out = Path(__file__).with_name("local_decision_update_results.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(out), "sdk": result["versions"],
                      "gradient": result["gradient_check"],
                      "update": result["single_training_update"],
                      "wall_seconds": result["runtime"]["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
