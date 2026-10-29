# Decision-focused VQC experiment

**Execution:** 2026-10-28 (Asia/Kolkata), CPython 3.12 local CPU exact statevector.  
**Status:** executed synthetic experiment; not an operational dataset, QCloud call, or QPU run.  
**Entry point:** `python -m experiments.run_quantum_learning`; JSON is in `experiments/artifacts/quantum-learning-results.json`.

## Mechanism and objective

`backend/quantum_learning.py` implements a two-qubit statevector circuit directly. The feature map applies `RY(pi*x0)` and `RY(pi*x1)`, then `CNOT(0,1)`, trainable `RY(theta0)` and `RY(theta1)`, and `CNOT(1,0)`. The observable is `Z` on q0 and predicted SLA risk is `sigmoid(-2 <Z0>)`. Exact simulator expectations are calculated from all four state amplitudes; there are no finite-shot counts. Metadata records the 2-qubit, 4-layer circuit and gate counts.

The decision loss models two soft route choices. Route A has estimated cost `2 + 4p`; route B has estimated cost `3.1 + 4(1-p)`. A temperature-1 softmax converts these estimates to route shares. The training loss is realized cost over the fixed synthetic SLA label: `share_A * (2+4y) + share_B * (3.1+4(1-y))`. This is an explicitly smooth surrogate for discrete dispatch, not a validated fleet objective. The predictive-only VQC instead minimizes binary cross-entropy; the classical baseline is logistic regression trained on the same 40 rows. Evaluation on validation and test uses the same 12-row splits and exact same per-row costs. The oracle uses the synthetic label and chooses the lower of those two costs, so it is a reference bound for this toy objective only.

For a concrete forward calculation, the frozen artifact records `x=(0.2,0.8)`, `theta=(0.31,-0.47)`, resulting `<Z0>`, risk probability, soft route shares, and realized cost under an SLA event. The gradient is chain rule: parameter-shift estimates each circuit observable derivative, then classical derivatives propagate through the sigmoid and soft decision cost. At the frozen initial point, both decision-loss gradient coordinates are nonzero; the reported maximum absolute error against centered finite difference at `h=1e-5` is approximately `5.31e-11` (acceptance tolerance `2e-7`).

## Frozen synthetic data and measured results

The fixture contains 64 deterministic arithmetic-generated examples. Its label rule is fixed in code before fitting and explicitly synthetic. Splits are contiguous fixed indices: train 0-39, validation 40-51, test 52-63. SHA-256 hashes for all split payloads are embedded in the raw JSON. No test rows are passed to either optimizer. All models share the same objective evaluation rows, and the split is independent of training outcomes. The fixture is not sourced from real fleet operations.

Test-set results (`n=12`, single deterministic run):

| Model | Mean dispatch cost | Regret vs synthetic oracle | Brier | Accuracy |
|---|---:|---:|---:|---:|
| Decision-focused VQC | 3.2664 | 0.8081 | 0.1183 | 0.8333 |
| Predictive-only VQC | 3.1622 | 0.7039 | 0.1009 | 0.9167 |
| Classical logistic | 3.1779 | 0.7196 | 0.1073 | 0.8333 |

This tiny test set has no uncertainty estimate or meaningful calibration claim. The decision-focused VQC is worse than both comparators on test cost and Brier score in this run. It is evidence that the code path and comparison executed, not evidence of a benefit or quantum advantage. No ablation of the discrete-choice surrogate was run; predictive-only training is the loss ablation.

## Initial VQC prototype refinement record

| Substantive cycle | Failure hypothesis and evidence | Material change | Retest and result |
|---|---|---|---|
| Circuit coupling | One ansatz parameter could fail to affect the measured decision score. Initial gradient output showed `theta1` at `5.46e-17` for an 8-row decision-loss check. | Reversed the final CNOT from `0->1` to `1->0`, coupling the second rotation to measured q0. | The decision-loss parameter-shift gradient matched centered differences on both coordinates. |
| Matched toy evaluation | Predictive loss could be mistaken for decision quality, and unmatched data would confound the comparison. The initial workspace had no VQC, split fixture, or classical baseline. | Added frozen synthetic train/validation/test rows, split hashes, decision and predictive VQCs, train-only logistic fit, shared cost evaluation, and synthetic-oracle regret. | On the 12-row held-out split, decision VQC cost was `3.2664`, predictive-only `3.1622`, logistic `3.1779`. The decision VQC underperformed; this result motivated an integration with the actual dispatcher contract. |

The later finite-difference, determinism, malformed-input, and training-boundary additions are **verification hardening**, not a third substantive refinement cycle. The dispatcher integration is reported separately in [dispatch-integrated-vqc.md](dispatch-integrated-vqc.md), with its own substantive changes and actual domain/validator outcomes.

## Limits and next work

- Local exact statevector only; no QpiAI package invocation, authenticated API, shots, QCloud, Opt, or QPU evidence.
- Dataset is synthetic and tiny; a single held-out split does not establish generalization or calibration.
- The soft route share is a differentiable training surrogate. A production decision requires operational route constraints and an independent validator from the dispatch backend.
- Decision regret here is relative only to the explicit two-route synthetic oracle, not a fleet-wide optimum.
- Classical logistic is a minimal matched reference, not a broad classical benchmark. No repeated seeds or uncertainty intervals were computed.
- Split hashes and raw values are preserved in the adjacent JSON; dependency versions are unavailable for this dependency-free module except Python runtime/platform.
