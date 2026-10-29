# QpiAI local VQC decision-gradient check

This bounded experiment sends a real two-qubit circuit through QpiAI Quantum's
`StatevectorSimulator`. It uses the same feature encoding, entanglers, trainable
angles, fixed `Z0` readout, sigmoid risk mapping, and soft-route decision loss
as `backend/quantum_learning.py`.

The simulator returns exact state amplitudes. There is no measurement sampling,
remote service, credential, cloud job, or QPU call. SDK statevector index bit 0
was checked against the custom reference circuit. The four deterministic rows
are the first four training records from the synthetic arithmetic fixture; no
validation or test rows participate.

## Reproduce

From the repository root in PowerShell:

```powershell
.\.venv-sdk\Scripts\python.exe experiments_qpiai_vqc\run_local_decision_update.py
.\.venv-sdk\Scripts\python.exe -m pytest tests/test_qpiai_vqc.py -q
```

The runner rewrites `local_decision_update_results.json`. The observed run used
Python 3.12.14 and QpiAI Quantum 0.3.0. It compared SDK parameter-shift
decision gradients with both the custom simulator and centered finite
differences (`h=1e-5`, tolerance `2e-7`), then applied one training-only
gradient descent update at learning rate 0.08. See the JSON artifact for the
numeric gradients, losses, timing, exact row features/labels, versions and
execution limits.

This is a local SDK circuit execution check, not evidence of calibrated SLA
risk, operational improvement, QpiAI Cloud capability, or quantum advantage.
