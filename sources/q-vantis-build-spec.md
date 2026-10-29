You are a senior full-stack + optimization engineer. Build the project "Q-VANTIS": a risk-aware fleet dispatch and disruption-recovery system for contract logistics, using a proposed algorithm called RACE (Risk-Aware Contractual Execution). Work in the repo root `q-vantis/`. Follow the spec below exactly, build in the milestone order given, and keep the whole thing runnable at every milestone.

## 0. Ground rules (non-negotiable)
1. Thin vertical slice first. Get scenario -> routes -> risk -> selection -> validation -> API -> basic UI working end to end on a tiny instance, THEN add quantum features and polish.
2. Never assume QpiAI SDK APIs. Before writing adapters, run a compatibility spike (M0): install the packages, inspect their real public API (`help()`, `dir()`, README, docs), and write minimal executable examples. Record findings in `docs/sdk_findings.md`. If something is unsupported (e.g. native HUBO), document it and implement a clearly labelled alternative. Do not invent method names.
   - QpiAI Quantum SDK: https://github.com/qpiai/quantum-sdk | docs: https://docs.qcloud.qpiai.tech/
   - QpiAI Opt: https://github.com/qpiai/qpiai-opt | portal: https://opt.qpiai.tech/
3. Honest reporting everywhere. No fabricated metrics. Never imply QPU use when a simulator ran. Never claim quantum advantage or novelty. Every result is labelled one of: classical | local_simulator | cloud_simulator | QPU. Show "unavailable" instead of made-up values.
4. Every plan shown as feasible must pass an independent validator that recomputes coverage, duplicates, capacity, time windows and the objective from scratch (never trust solver-reported energy as business cost).
5. Credentials live only in backend env vars (`.env`, gitignored). Never bundled into web/mobile. Never logged or returned by the API.
6. Classical fallback must always work if a quantum job fails, times out, or returns no feasible plan. Failed quantum jobs must never crash the app.
7. Quantum SDKs are never called from the mobile app. Mobile and web only talk to the backend REST API.
8. Everything seeded and reproducible. Store seed, scenario ID, solver settings, SDK versions, backend metadata for every run.

## 1. Tech stack
- Backend: Python 3.10+ (check compatibility with QpiAI SDKs), FastAPI, Pydantic v2, SQLAlchemy + SQLite, NumPy/pandas/SciPy, Google OR-Tools (classical routing baseline), pytest.
- Web: React + TypeScript + Vite (dispatcher + experiment reviewer dashboard).
- Mobile: React Native + TypeScript + Expo Go (driver app; no native modules that need a custom dev build, or document them if unavoidable).
- Risk prediction: classical baselines (logistic regression / tree models) with calibration metrics. A quantum-kernel predictor is optional, only after everything else works, and must use identical train/test splits vs classical.

## 2. Repo layout
q-vantis/
  apps/web/            # React+TS+Vite
  apps/mobile/         # RN + Expo Go
  backend/
    api/ domain/ routing/ risk/ race/
    optimization/{qubo,ising,hubo}/
    quantum/{qcloud,adapters,qpiai_opt}/
    validation/ benchmarks/ persistence/
    tests/{unit,integration,fixtures}/
  data/{scenarios,benchmark_instances}/
  docs/
  .env.example
  README.md

## 3. Domain model (Pydantic + DB)
Scenario(id, name, provenance[synthetic|public_benchmark|user_provided|operational], seed, horizon, schema_version), DeliveryOrder(id, destination, demand, service_duration, time_window, sla_ref, priority), Vehicle(id, capacity, start/end, availability_window, cost params, restrictions), Contract/SLA(id, deadline policy, grace, penalty function, penalty cap, currency), RouteCandidate(id, vehicle, ordered stops, predicted arrivals, served_orders, feasibility, cost features), DisruptionEvent(id, type, timestamp, affected entity, magnitude, status), Plan(id, scenario, selected routes, feasibility, objective breakdown, approval_status, parent_plan_id), ExperimentRun(id, formulation, solver, sdk_versions, settings, seeds, backend, execution_type, runtime, status, objective, metrics, artifacts), DriverEvent(id, driver/vehicle/stop refs, type, client_ts, server_ts, ack, notes).
Rules: ISO 8601 timestamps with a documented timezone policy; money always has currency; missing travel-time/penalty data must fail validation or use a documented default, never silently zero; minimize driver personal data.

## 4. Core functionality

### 4.1 Scenario import + validation (FR-01..05)
CSV/JSON import templates + export. Validate fields, types, IDs, time-window consistency, capacity, contract params. Return field-level errors; never silently drop records. Include a seeded synthetic scenario generator and a Solomon VRPTW benchmark loader (adapt fields to our schema).

### 4.2 Route generation (FR-06..10)
OR-Tools VRPTW-based candidate route pool generator (multiple diverse candidates per vehicle: vary seeds/metaheuristics/perturbations). Expose count, settings, runtime. Independent validator for coverage, duplicates, capacity, time windows.

### 4.3 Risk + contract exposure (FR-11..16)
Seeded travel/service-time scenario sampling (common scenario set across all methods). Per route: expected operating cost, expected SLA penalty, optional CVaR_alpha with configurable alpha and lambda. Route cost: C_r = C_operating + E[P_r] + lambda * CVaR_alpha(P_r). Document estimation, avoid double counting penalties. Per-stop ETA distribution and delay-risk probability. Delay-risk predictor: classical baseline with held-out metrics + calibration (Brier, reliability curve/ECE).

### 4.4 RACE (FR-17..24)
Modular pipeline: route portfolio generation -> anticipatory risk evaluation -> contract-aware route selection -> event-triggered recovery. Inputs: candidate pool, scenario set, cost rules, constraints. Outputs: selected routes, objective breakdown, feasibility, risk metrics, config. Selection model:
  min sum_r C_r x_r  s.t. sum_r a_ir x_r = 1 for all i; x_r + x_s <= 1 for incompatible (r,s) (same vehicle/driver conflicts).
Record all coefficients, penalty weights, route IDs, decoding rules for reproducibility. Keep the RACE name and novelty claim provisional in UI/reports.

### 4.5 QUBO / Ising / HUBO (FR-25..30)
- QUBO: H(x) = sum C_r x_r + A * sum_i (sum_r a_ir x_r - 1)^2 + B * sum_{(r,s) in K} x_r x_s. Calibrate A and B (document the method) and test with exhaustive enumeration on small instances.
- Ising: derive from QUBO and test energy correspondence for ALL assignments on small instances (offset handled).
- HUBO: implement a genuinely higher-order term only where meaningful (e.g. 3-way interaction like shared-resource/shift conflicts or SLA-tier cascades), and only run natively if the installed Opt supports it. Otherwise record the limitation and offer a separately labelled quadratization/alternate solver path.
- Report per formulation: binary variable count, polynomial order, nonzero coefficients, penalty weights, solver interface.
- Compare formulations without assuming any is universally better.

### 4.6 QpiAI Quantum / QCloud (FR-31..36)
Adapter with: SDK import check, auth check, backend discovery, circuit construction, async job submission with bounded timeouts, status states (queued/running/completed/failed/timed_out), and persistence of circuit config, counts, shots, backend identity, job ID, timestamps. Implement at least one meaningful circuit experiment (e.g. a small QAOA-style circuit for a tiny selection instance). A circuit measurement is NEVER presented as proof the fleet problem was solved. Persist qubit count, depth, gate counts when available.

### 4.7 Disruption recovery (FR-37..41)
Disruption types: travel-time increase, vehicle unavailable, service delay. Recovery: freeze completed + locked stops, update scenario, re-plan remaining work, add configurable penalty for changing communicated assignments. Compare old vs new plan (cost delta, SLA exposure, affected deliveries, route changes). Keep original plan, event, revised plan, validation result as an audit trail. On failure keep the last valid plan and state that no replacement is approved. Test recovery against a static-plan baseline; don't reward unnecessary conservatism.

### 4.8 REST API (versioned, /api/v1)
POST /scenarios, GET /scenarios/{id}, POST /routes/generate, POST /risk/evaluate, POST /optimization/qubo, POST /optimization/ising, POST /optimization/hubo (clear capability error if unsupported), POST /quantum/circuits, GET /jobs/{id}, POST /plans/{id}/validate, POST /recovery/plan, GET /experiments/{id}, POST /driver/events.
Long jobs return a job ID and support polling. Errors have a stable code, message, correlation/job ID. Distinguish: job failure, timeout, unsupported capability, infeasible candidate, validation failure. Auth/authorization hooks (simple token auth for the prototype with dispatcher/driver/researcher/admin roles). Never expose secrets.

### 4.9 Web dashboard (FR-42..47)
Scenario import + validation summary; route assignments with ordered stops, ETAs, time windows, SLA exposure, cost breakdown; plan vs baseline comparison with feasibility + objective; experiment view (formulation, solver/backend, status, runtime, metadata, execution_type badge); recovery diff view; per-commitment ETA/risk and total financial exposure; clear labels for synthetic/benchmark/operational data; show solver failures and unavailable metrics honestly; accessible (labels, keyboard nav, non-colour-only status).

### 4.10 Mobile app (FR-48..53)
Driver view: assigned vehicle + ordered manifest; each stop shows location label, delivery ID, time window, status; submit arrived/completed/disruption events; pull approved route updates and flag changed manifests; show network/API errors; show "pending/unsent" and only confirm after backend ack. Must run in Expo Go.

### 4.11 Experiments + reporting (FR-54..59)
Unique run ID + stored manifest (seeds, scenario IDs, solver settings, SDK versions, backend metadata). Report: feasibility, coverage, operating cost, SLA penalty, tail risk, runtime, optimality gap (only if a reference optimum is known), quantum resources (shots/depth/gates/qubits, only fields that actually exist). Export JSON/CSV + human-readable summary. Label each result classical / local_simulator / cloud_simulator / QPU.

## 5. Required benchmarks
Distance-only baseline; risk-aware classical selection; exact enumeration or MILP on small instances; QpiAI Opt on the verified formulation; quantum circuit experiment on an identified backend; optional quantum-kernel vs classical predictor on identical splits. Use the same instances, candidate routes, costs, scenarios and stopping budgets.

## 6. Tests
Unit (objectives, penalties, sampling, decoders, constraints, validation); mathematical (exhaustive QUBO<->Ising equivalence, HUBO vs direct evaluation); integration (SDK import/auth/backend discovery/circuit + Opt submission, mocked when no credentials); end-to-end (import -> plan -> validate -> dashboard -> mobile manifest); recovery tests (inject disruption, completed work frozen, audit trail); validator must catch deliberately injected violations.

## 7. Milestones (build in order, commit at each, tests green before moving on)
M0 SDK compatibility spike + docs/sdk_findings.md
M1 Data schema, validation, candidate-route generation, independent validator
M2 RACE prototype: risk/contract cost + selection on small exhaustively-checked instances
M3 QUBO / Ising / HUBO (HUBO implemented or limitation documented)
M4 Quantum circuit experiment with persisted metadata
M5 Recovery workflow
M6 Web + mobile integration against real endpoints
M7 Benchmarks, report, README with setup instructions, limitations section

## 8. Acceptance criteria
1. Scenario imports, validates, and plans end to end (or infeasibility is clearly explained).
2. Validator catches intentionally introduced violations.
3. Small QUBO built and solved via the verified Opt path.
4. QUBO<->Ising equivalence passes for all assignments on a small instance.
5. A measured circuit runs through QpiAI Quantum on a recorded backend.
6. HUBO works natively or limitation + explicit alternative are documented.
7. Disruption triggers a revised plan without altering completed/locked work.
8. Web and mobile show real backend data and handle failures honestly.
9. Reproducible report compares RACE against a classical baseline.

## 9. Deliverables
Working code, `.env.example` (QPIAI_QUANTUM_API_KEY, QPIAI_OPT_API_KEY, QPIAI_QUANTUM_BACKEND, DATABASE_URL, LOG_LEVEL, APP_ENV; align names with the SDK's real auth mechanism), a README with one-command setup for backend, web and mobile, seeded demo scenario + demo script, docs/sdk_findings.md, docs/formulation.md (math for QUBO/Ising/HUBO + penalty calibration), docs/limitations.md.

## How to work
Start by printing a short plan, then execute M0 immediately. After each milestone, summarize what works, what's mocked, what's unverified, and any assumption you had to make. Ask me only when blocked by missing credentials or a genuine spec ambiguity; otherwise make the most conservative assumption, document it, and keep going.