# Architecture Readiness — Provisional

**Status:** review framework only. This document identifies decisions and evidence gates; it does not freeze the architecture or algorithm. The selected B+C direction remains a research hypothesis, with novelty, differentiability and decision-quality gains unproven.

## Verified boundary

- Official QML-01 calls for fleet dispatch under capacity, delivery-window, contract, traffic, and service-time uncertainty. Outputs include exact vehicle/route/ordered-stop assignments, ETAs and risk estimates, and financial exposure.
- Evaluation must cover SLA violations, valid coverage, cost, calibration, solver quality and runtime.
- The official QpiAI technology rubric has four five-point components: meaningful integration, QCloud correctness, Opt correctness, and working reproducible evidence. Citation and source extraction: `sources/problem-extract.txt`.
- SDK source/package versions and claims are captured in `SDK-EVIDENCE.md`. QAOA warm start and VQC/autodiff details remain unknown; do not freeze interfaces against assumed SDK APIs.

## Proposed decision sequence (owner review required)

1. **Problem contract:** specify operational feasibility, cost/exposure definitions, contractual violation event, uncertainty horizon, and which input fields are actually available. Settle source-data provenance before modelling.
2. **Data and prediction contract (M2):** compare target candidates, including whether a single risk probability is enough or an ETA/delay distribution is required. Record event definition, prediction horizon, calibration metadata, censoring, dependence/correlation assumptions, and leakage-safe split keys.
3. **Candidate route boundary (M1/M4):** classical route generation may produce individually feasible candidates, but the fleet-level selector and independent validator must still check global vehicle, coverage, time, capacity, driver, and commitment constraints. Do not treat candidate feasibility as global feasibility.
4. **Decision/risk interface (proposed only):** route candidate identity, vehicle, ordered stops, travel/service scenario summary, cost and feasibility evidence; risk outputs identify event, horizon, probability/distribution, calibration metadata, and source model version. Exposure should account for expected penalties with explicit joint-tail/correlation/censoring assumptions.
5. **Optimization/validation boundary (M1/M3/M4):** route-selection QUBO candidate may use selection/coverage penalties, but encoding, penalty magnitudes and repair behavior are undecided. Preserve raw quantum/solver bitstring, objective energy, decoded assignment, and any post-repair output as distinct fields. Independent validator recomputes constraints and business metrics from base inputs.
6. **Warm-start feasibility gate:** specify which classical relaxation supplies initial information, how it maps into the state/circuit, and whether the backend/API supports that parameterization. Compare against ordinary QAOA. Exact enumeration on tiny cases must validate QUBO energy/feasibility and decoding before any quantum comparison.
7. **Experiment contract:** compare exact exhaustive solutions on tiny instances, an appropriate MILP/OR-Tools formulation, classical heuristics, ordinary QAOA, standard VQC, and strong classical ML. Measure decision quality as well as predictive metrics; report infeasible rate, solution gap, SLA exposure, calibration, runtime, shot/call counts, and cost where known.
8. **Execution boundary:** development checkout authors portable code; Legion handles later heavy runs after inventory. Record immutable experiment identifiers, seeds, split/data hashes, dependency versions, backend, shots, counts, wall time, solver outputs and logs. Simulator evidence, authenticated QCloud/Opt calls, and QPU evidence are separate categories.

## M1/M2 first; M3/M4 contract handoff

- **M1 first:** approve official problem interpretation, objective/constraint definitions, and candidate route/decision boundaries. Write a list of undecided assumptions rather than silently setting values.
- **M2 first:** establish data provenance and a leakage-safe split before fitting. Select prediction target alternatives and classical baselines; preserve calibration and decision-metric reporting.
- **M3/M4 later:** start from frozen interface proposals and deterministic tiny fixtures. Their QUBO, QAOA, route validator, and benchmark scaffolding must not wait for a trained VQC. Freeze interfaces only after owner review.

## Explicit open gates

- Is the dispatch objective expected realized cost, risk-adjusted cost, or both as separately reported metrics? Define contractual penalties, probability events and dependence structure.
- What real or permitted synthetic data exists, with enough route/time features and outcomes to estimate ETA distributions and calibrate SLA risk?
- Can QpiAI Quantum and Opt versions be resolved compatibly? What exact circuit, solver, backend and job APIs exist at pinned releases?
- Is a differentiable decision surrogate valid when the dispatch action is discrete? Compare against decision-focused methods with exact/relaxed/black-box downstream optimization; choose only after mathematical and empirical review.
- Does classical relaxation warm-start preserve meaningful feasibility/quality after mapping, and does the QCloud backend support it? Unknown.
- What target Legion OS, architecture, Python build, memory, storage, GPU/VRAM, and connectivity are available? Unknown until remote inventory.

## Not claimed

No final algorithm, novelty, QAOA warm-start compatibility, SDK autodiff, optimized route plan, calibrated model, quantum advantage, benchmark, hardware capability, or remote setup completion is claimed here.
