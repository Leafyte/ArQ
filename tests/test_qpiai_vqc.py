"""Integration checks against the installed QpiAI local statevector SDK."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("qpiai_quantum")

from backend.quantum_learning import (  # noqa: E402
    _make_examples, decision_loss_gradient, expectation_z0,
)
from backend.qpiai_vqc import sdk_expectation_z0, sdk_statevector, sdk_risk_probability  # noqa: E402
from experiments_qpiai_vqc.run_local_decision_update import (  # noqa: E402
    run_experiment, sdk_decision_gradient, sdk_finite_difference,
)


def test_qpiai_statevector_and_z_readout_match_reference_circuit():
    features, theta = (0.2, 0.8), (0.31, -0.47)
    state = sdk_statevector(features, theta)
    assert math.isclose(sum(abs(v) ** 2 for v in state), 1.0, abs_tol=1e-10)
    assert sdk_expectation_z0(features, theta) == pytest.approx(
        expectation_z0(features, theta), abs=1e-12)
    assert sdk_risk_probability(features, theta) == pytest.approx(0.7551089048198733, abs=1e-12)


def test_sdk_parameter_shift_decision_gradient_matches_custom_and_sdk_fd():
    all_rows, _ = _make_examples()
    train_rows = [row for row in all_rows if row.split == "train"][:4]
    theta = (0.31, -0.47)
    sdk = sdk_decision_gradient(theta, train_rows)
    reference = decision_loss_gradient(theta, train_rows)
    finite = sdk_finite_difference(theta, train_rows, 1e-5)
    for value, expected in zip(sdk, reference):
        assert value == pytest.approx(expected, abs=2e-7)
    for value, expected in zip(sdk, finite):
        assert value == pytest.approx(expected, abs=2e-7)
    assert all(abs(value) > 1e-5 for value in sdk)


def test_one_real_sdk_gradient_update_lowers_tiny_training_loss():
    result = run_experiment()
    update = result["single_training_update"]
    assert update["training_rows_only"] is True
    assert update["updated_theta"] != update["initial_theta"]
    assert update["updated_loss"] < update["initial_loss"]
    assert result["remote_calls"] == result["cloud_jobs"] == result["qpu_jobs"] == 0
    assert result["gradient_check"]["rows"] == 4


@pytest.mark.parametrize("features,theta", [((0.2,), (0.1, 0.2)), ((1.1, 0.2), (0.1, 0.2)),
                                               ((0.2, 0.3), (float("nan"), 0.1))])
def test_qpiai_adapter_rejects_invalid_inputs_before_sdk_execution(features, theta):
    with pytest.raises(ValueError):
        sdk_expectation_z0(features, theta)
