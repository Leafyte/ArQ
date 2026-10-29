"""Small, exact statevector VQC and decision-focused toy dispatch study.

This module intentionally uses only the Python standard library. Its statevector
simulator is an actual gate-by-gate 2-qubit circuit implementation, not an
SDK call or a proxy. It is CPU-only and suitable for deterministic tiny studies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Example:
    features: tuple[float, float]
    label: int
    split: str
    index: int


@dataclass(frozen=True)
class CircuitMetadata:
    qubits: int = 2
    depth: int = 4
    feature_map: str = "RY(pi*x0) on q0; RY(pi*x1) on q1"
    ansatz: str = "CNOT(0,1); RY(theta0) q0; RY(theta1) q1; CNOT(1,0)"
    observable: str = "Z on q0; risk=sigmoid(-2*<Z0>)"
    gate_counts: tuple[tuple[str, int], ...] = (("RY", 4), ("CNOT", 2))
    initialization: str = "theta=(0.31,-0.47) radians"
    simulator: str = "custom exact statevector, complex128 semantics, local CPU"


def _ry(state: list[complex], wire: int, angle: float) -> None:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    mask = 1 << wire
    for i in range(4):
        if i & mask == 0:
            j = i | mask
            a, b = state[i], state[j]
            state[i], state[j] = c * a - s * b, s * a + c * b


def _cnot(state: list[complex], control: int, target: int) -> None:
    c, t = 1 << control, 1 << target
    for i in range(4):
        if i & c and not i & t:
            j = i | t
            state[i], state[j] = state[j], state[i]


def expectation_z0(features: Sequence[float], theta: Sequence[float]) -> float:
    """Return <Z0> after a feature-map + trainable entangling circuit."""
    if len(features) != 2 or len(theta) != 2:
        raise ValueError("expected exactly two features and two trainable angles")
    if any(not math.isfinite(float(v)) for v in (*features, *theta)):
        raise ValueError("circuit inputs must be finite")
    state = [1 + 0j, 0j, 0j, 0j]
    _ry(state, 0, math.pi * float(features[0]))
    _ry(state, 1, math.pi * float(features[1]))
    _cnot(state, 0, 1)
    _ry(state, 0, float(theta[0]))
    _ry(state, 1, float(theta[1]))
    _cnot(state, 1, 0)
    return sum((1 if i & 1 == 0 else -1) * abs(state[i]) ** 2 for i in range(4))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 700)))
    e = math.exp(max(x, -700))
    return e / (1.0 + e)


def risk_probability(features: Sequence[float], theta: Sequence[float]) -> float:
    return _sigmoid(-2.0 * expectation_z0(features, theta))


def _probability_gradient(features: Sequence[float], theta: Sequence[float]) -> list[float]:
    """Chain rule: analytic parameter-shift for <Z>, then sigmoid derivative."""
    z = expectation_z0(features, theta)
    p = _sigmoid(-2.0 * z)
    grad = []
    for k in range(2):
        plus, minus = list(theta), list(theta)
        plus[k] += math.pi / 2
        minus[k] -= math.pi / 2
        dz = (expectation_z0(features, plus) - expectation_z0(features, minus)) / 2
        grad.append(-2.0 * p * (1.0 - p) * dz)
    return grad


def _softmax_route0(p: float, temperature: float = 1.0) -> float:
    # Costs encode a compact dispatch tradeoff: route A is cheaper but riskier.
    c0, c1 = 2.0 + 4.0 * p, 3.1 + 4.0 * (1.0 - p)
    return _sigmoid((c1 - c0) / temperature)


def realized_dispatch_cost(p: float, label: int) -> float:
    """Soft route selection cost; labels are the synthetic SLA event outcomes."""
    w0 = _softmax_route0(p)
    actual0, actual1 = 2.0 + 4.0 * label, 3.1 + 4.0 * (1 - label)
    return w0 * actual0 + (1.0 - w0) * actual1


def _decision_loss_and_dp(p: float, label: int) -> tuple[float, float]:
    w0 = _softmax_route0(p)
    # dw0/dp = w0*(1-w0)*(-8); realized costs differ by label.
    a0, a1 = 2.0 + 4.0 * label, 3.1 + 4.0 * (1 - label)
    return w0 * a0 + (1 - w0) * a1, -8.0 * w0 * (1.0 - w0) * (a0 - a1)


def _make_examples() -> tuple[list[Example], dict[str, str]]:
    """Freeze deterministic synthetic records and contiguous index splits."""
    rows: list[Example] = []
    for i in range(64):
        x0 = ((i * 17 + 11) % 64) / 63
        x1 = ((i * 29 + 7) % 64) / 63
        # Synthetic rule is fixed before model training; no operational data.
        y = int(0.62 * x0 + 0.38 * x1 + 0.12 * math.sin(2 * math.pi * x0) > 0.55)
        split = "train" if i < 40 else "validation" if i < 52 else "test"
        rows.append(Example((x0, x1), y, split, i))
    hashes = {}
    for split in ("train", "validation", "test"):
        payload = [asdict(row) for row in rows if row.split == split]
        hashes[split] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return rows, hashes


def _loss_gradient(theta: Sequence[float], rows: Iterable[Example], objective: str) -> tuple[float, list[float]]:
    subset = list(rows)
    if not subset:
        raise ValueError("training data cannot be empty")
    total, gradient = 0.0, [0.0, 0.0]
    for row in subset:
        p = risk_probability(row.features, theta)
        dp = _probability_gradient(row.features, theta)
        if objective == "decision":
            loss, dloss_dp = _decision_loss_and_dp(p, row.label)
        elif objective == "predictive":
            eps = 1e-12
            loss = -(row.label * math.log(max(p, eps)) + (1 - row.label) * math.log(max(1 - p, eps)))
            dloss_dp = (p - row.label) / max(p * (1 - p), eps)
        else:
            raise ValueError("objective must be 'decision' or 'predictive'")
        total += loss
        for k in range(2):
            gradient[k] += dloss_dp * dp[k]
    n = len(subset)
    return total / n, [g / n for g in gradient]


def decision_loss_gradient(theta: Sequence[float], rows: Sequence[Example]) -> list[float]:
    return _loss_gradient(theta, rows, "decision")[1]


def finite_difference_gradient(theta: Sequence[float], rows: Sequence[Example], h: float = 1e-5) -> list[float]:
    if h <= 0 or not math.isfinite(h):
        raise ValueError("h must be finite and positive")
    answer = []
    for k in range(2):
        plus, minus = list(theta), list(theta)
        plus[k] += h
        minus[k] -= h
        fp = _loss_gradient(plus, rows, "decision")[0]
        fm = _loss_gradient(minus, rows, "decision")[0]
        answer.append((fp - fm) / (2 * h))
    return answer


def _fit_vqc(rows: Sequence[Example], objective: str, steps: int = 120, rate: float = 0.08) -> tuple[list[float], list[float]]:
    theta = [0.31, -0.47]
    history = []
    for _ in range(steps):
        loss, gradient = _loss_gradient(theta, rows, objective)
        history.append(loss)
        theta = [theta[k] - rate * gradient[k] for k in range(2)]
    history.append(_loss_gradient(theta, rows, objective)[0])
    return theta, history


def _fit_classical(rows: Sequence[Example], steps: int = 500, rate: float = 0.3) -> list[float]:
    weights = [0.0, 0.0, 0.0]
    for _ in range(steps):
        grad = [0.0, 0.0, 0.0]
        for row in rows:
            p = _sigmoid(weights[0] + weights[1] * row.features[0] + weights[2] * row.features[1])
            err = p - row.label
            for j, v in enumerate((1.0, *row.features)):
                grad[j] += err * v / len(rows)
        weights = [w - rate * g for w, g in zip(weights, grad)]
    return weights


def run_study(steps: int = 120) -> dict:
    if not 1 <= steps <= 2000:
        raise ValueError("steps must be between 1 and 2000")
    rows, split_hashes = _make_examples()
    train = [r for r in rows if r.split == "train"]
    validation = [r for r in rows if r.split == "validation"]
    test = [r for r in rows if r.split == "test"]
    theta_decision, hist_decision = _fit_vqc(train, "decision", steps)
    theta_predictive, hist_predictive = _fit_vqc(train, "predictive", steps)
    classical = _fit_classical(train)

    def metrics(split: Sequence[Example], theta: Sequence[float], kind: str) -> dict:
        costs, brier, correct = [], [], []
        for row in split:
            p = risk_probability(row.features, theta) if kind == "vqc" else _sigmoid(classical[0] + classical[1] * row.features[0] + classical[2] * row.features[1])
            costs.append(realized_dispatch_cost(p, row.label))
            brier.append((p - row.label) ** 2)
            correct.append(int((p >= 0.5) == bool(row.label)))
        return {"n": len(split), "mean_dispatch_cost": sum(costs) / len(costs), "brier": sum(brier) / len(brier), "accuracy": sum(correct) / len(correct)}

    test_scores = {
        "decision_focused_vqc": metrics(test, theta_decision, "vqc"),
        "predictive_only_vqc": metrics(test, theta_predictive, "vqc"),
        "classical_logistic": metrics(test, (), "classical"),
    }
    # Exact per-row reference knows the synthetic label and chooses lower realized cost.
    oracle = sum(min(2.0 + 4.0 * r.label, 3.1 + 4.0 * (1 - r.label)) for r in test) / len(test)
    for result in test_scores.values():
        result["regret_vs_synthetic_oracle"] = result["mean_dispatch_cost"] - oracle
    check_analytic = decision_loss_gradient([0.31, -0.47], train[:8])
    check_finite = finite_difference_gradient([0.31, -0.47], train[:8])
    gradient_abs_errors = [abs(a - b) for a, b in zip(check_analytic, check_finite)]
    sample_features = (0.2, 0.8)
    sample_p = risk_probability(sample_features, [0.31, -0.47])
    sample_w0 = _softmax_route0(sample_p)
    return {
        "evidence_label": "synthetic deterministic data; exact local statevector; no QCloud/QPU call",
        "data": {"n": len(rows), "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)}, "split_hashes_sha256": split_hashes, "rule": "y=1[0.62*x0+0.38*x1+0.12*sin(2*pi*x0)>0.55]", "seed": "not applicable; arithmetic fixture"},
        "circuit": asdict(CircuitMetadata()),
        "gradient_check": {"analytic_parameter_shift_chain_rule": check_analytic, "centered_finite_difference_h_1e-5": check_finite, "absolute_errors": gradient_abs_errors, "max_absolute_error": max(gradient_abs_errors), "tolerance_absolute": 2e-7, "training_examples": 8},
        "tiny_example": {"features": sample_features, "trainable_angles": [0.31, -0.47], "readout_z0": expectation_z0(sample_features, [0.31, -0.47]), "risk_probability": sample_p, "soft_route_a_share": sample_w0, "soft_route_b_share": 1.0 - sample_w0, "realized_cost_if_sla_event": realized_dispatch_cost(sample_p, 1)},
        "training": {"steps": steps, "learning_rate": 0.08, "decision_initial_loss": hist_decision[0], "decision_final_loss": hist_decision[-1], "predictive_initial_bce": hist_predictive[0], "predictive_final_bce": hist_predictive[-1], "decision_theta": theta_decision, "predictive_theta": theta_predictive, "classical_logistic_weights": classical},
        "test_metrics": test_scores,
        "synthetic_oracle_test_cost": oracle,
        "validation_metrics": {"decision_focused_vqc": metrics(validation, theta_decision, "vqc"), "predictive_only_vqc": metrics(validation, theta_predictive, "vqc"), "classical_logistic": metrics(validation, (), "classical")},
        "resource_budget": {"qubits": 2, "shots": "exact statevector (no sampling)", "optimizer_steps_per_model": steps, "training_rows": len(train), "test_rows": len(test), "external_services": 0},
    }


if __name__ == "__main__":
    print(json.dumps(run_study(), indent=2, sort_keys=True))
