# V1 Acceptance Targets

**Status:** owner-approved targets for the V1 implementation window. This document records what must be demonstrated; it does not report completed implementation or results. Record each gate as **PASS**, **PARTIAL**, **FAIL**, or **NOT RUN**, with artifact/path and exact command or procedure.

## Product boundary

V1 is a decision-focused fleet-dispatch backend centered on a variational quantum circuit (VQC), with classical warm-start information and warm-start QAOA as a supporting path. The product goal is a defensible dispatch decision under SLA/financial risk, not predictive accuracy alone. Use tiny deterministic local CPU instances by default (typically 4–8 qubits); document actual resource budgets. QPU use and paid credits are out of scope absent specific authorization.

## Measurable gates

### 1. VQC mechanism and gradient evidence

- The implemented circuit must encode actual instance features through documented rotation gates, include entanglement, and expose trainable observable/readout parameters. Record qubits, depth, gate counts, feature map, observable, initialization, and SDK/version when available.
- Provide a deterministic tiny input example showing feature values -> encoded circuit parameters/states -> measured or simulator observables -> prediction/decision score.
- Check at least one nontrivial trainable parameter gradient against centered finite differences on a tiny deterministic circuit, with stated step size, absolute/relative tolerance, and observed errors.
- For the real decision-focused loss, show a **nonzero gradient path through the actual circuit** and identify the mechanism (parameter-shift or verified autodiff). Include a numeric gradient check for at least one circuit parameter on a non-degenerate instance. A gradient through a classical proxy alone does not pass this gate.
- State any surrogate, estimator variance, or approximation explicitly; report if the check fails or the circuit gradient is zero.

### 2. Decision-focused objective and matched comparisons

- Define the downstream dispatch objective in computable terms (e.g. expected operating cost plus SLA penalty/risk term over scenarios), and show how its value depends on decoded decisions and circuit output. Provide a tiny hand-computable example.
- Compare at minimum: predictive-only training, a classical model/baseline, and decision-focused VQC training. Use identical leakage-safe train/validation/test splits, scenario sets, common random numbers (CRN), and evaluation budgets. Freeze tuning before blind held-out testing.
- Report decision regret relative to an exact or best-known tiny-instance reference when available, plus feasibility rate, coverage, expected/realized operating cost, SLA exposure, and predictive/calibration metrics where meaningful. Give uncertainty or replicate spread and sample counts.
- Keep source data provenance and split hashes. No held-out fitting, post-hoc favorable scenario selection, random or fabricated labels, or synthetic results presented as operational evidence. Label synthetic data clearly.
- Include a decision-quality ablation that removes the decision-focused term, and explain whether its gradient contribution is measurably nonzero.

### 3. Tiny QUBO and warm-start QAOA correctness

- Construct at least one tiny route/dispatch QUBO with explicit variable order, objective coefficients, constraint penalties, and offset. Exhaustively enumerate every bitstring and independently recompute both business objective/constraints and QUBO energy.
- Verify QUBO-to-Ising conversion including the constant offset for every assignment on the same tiny instance.
- Record the classical relaxation or other source of warm-start probabilities, the complete vector, and its mapping to initial state and tilted mixer. Compare ordinary and warm-start QAOA on matched instances/seeds/budgets when executable.
- Check tilted-mixer expectation/energy on a tiny analytically tractable or exhaustive example against the implementation. Verify measurement decoding separately from circuit energy.
- Validate feasibility independently on raw samples and on any repaired candidate. Preserve raw bitstrings, decoded plans, repair edits, and validator output with provenance. Any fallback to classical output must be labelled explicitly and cannot be counted as a quantum-derived feasible sample.
- If the pinned SDK's warm-start path cannot be executed locally, report the exact unsupported/blocking call and leave the gate PARTIAL/NOT RUN; do not substitute invented SDK APIs.

### 4. Backend/API, persistence, and failure recovery

- Provide a durable API job lifecycle for submission, polling, terminal success/failure/timeout/cancellation states, result retrieval, validation, and restart recovery. Persist job ID, request/instance hash, seed, settings, package versions, backend/execution type, timestamps, status, outputs, and available circuit metadata.
- Validate malformed/invalid data with stable field-level errors. Distinguish unsupported capability, infeasibility, solver/job failure, timeout, cancellation, and validation failure.
- A failed, timed-out, cancelled, or infeasible quantum path must not crash the service or replace the last validated decision; any classical fallback must be surfaced as fallback with reason and provenance.
- Demonstrate secret redaction in returned API payloads and captured logs using non-secret sentinel values. Never read or print actual key contents. SDK authentication may be exercised only through configured environment variables; report whether the result was mocked or a real authenticated call.
- Run bounded stress checks for invalid data, concurrent jobs, timeout, cancellation, process restart/recovery, and secret redaction. Record request count/concurrency, wall time, memory/CPU budget when available, observed failures, and cleanup. Avoid unbounded load or heavy workloads.

### 5. QpiAI adapter evidence

- Include meaningful QpiAI Quantum and QpiAI Opt adapters or explicit capability boundaries. At least one real decision-focused local quantum computation must run; mocked counters or fabricated results cannot satisfy this minimum.
- Identify each adapter check as source-inspected, import-only, local simulator, mocked, authenticated cloud simulator, Opt service, or QPU. Preserve actual backend/job IDs and response metadata where those exist; mark unavailable fields unavailable.
- An authenticated API call or QPU run is reported only if actually performed. Do not spend QPU credits without specific authorization. Do not claim quantum advantage, novelty, or superiority from a working circuit or small example.

### 6. Three substantive refinement cycles

Record at least three completed cycles in a table with: cycle ID; specific failure hypothesis; evidence establishing the failure; material code/config change; retest command and result; artifact/commit or file pointer; and remaining concern. Each cycle must change or challenge a substantive hypothesis and rerun a relevant check. Renaming milestones, prose-only cleanup, or cosmetic changes do not count.

## Required evidence summary

For each gate list status, exact evidence, limitations, and next action. Include the final matched comparison table and execution labels (classical, local simulator, mocked, cloud simulator, QPU). Separate targets from results throughout. If a gate cannot be completed within the time/resource budget, report it as open with the blocker; do not relax the criterion after observing results.
