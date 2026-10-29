# SDK Interface Requirements for Codex Adapters

**Date:** 2026-10-28 (Asia/Kolkata)  
**Author:** Antigravity (Implementation Support Agent)  
**Target:** Codex (Architecture & Implementation Lead) and Luna  
**Status:** PROPOSED INTERFACE SPECIFICATION (Awaiting Owner & Codex Freeze)

---

## 1. Overview and Architecture Context

DebugLeaf investigates a hybrid quantum machine learning and combinatorial optimization pipeline for **QML-01 (Delay-Risk-Aware Fleet Dispatch)**. The overall system flow comprises four distinct processing stages:

```
[Problem Inputs & Stops]
       │
       ├──► [Classical Route Generator] ──► {Candidate Routes}
       │                                            │
       └──► [VQC Delay-Risk Predictor]  ──► {SLA Risk Estimates}
                                                    │
                                           [QUBO Formulation]
                                                    │
                                     [Continuous Relaxation (LP)]
                                                    │
                                                    ▼
                                         [Warm-Start QAOA Solver]
                                                    │
                                                    ▼
                                      [Independent Dispatch Validator]
```

To preserve clean boundaries, testability, and portability, Codex's implementation must decouple core mathematical algorithms from external SDK specifics using **four dedicated adapters**:
1. `RouteGeneratorAdapter` (Google OR-Tools / heuristic)
2. `VQCPredictorAdapter` (QpiAI Quantum parameterized circuits)
3. `RouteQUBOAdapter` (QpiAI Opt algebraic modeling and QUBO compilation)
4. `QAOASolverAdapter` (QpiAI Opt Quantum / QpiAI Quantum QAOA execution)

Below are the exact SDK interfaces, data contracts, and unresolved decisions.

---

## 2. Adapter 1: Classical Route Generator (`RouteGeneratorAdapter`)

### 2.1 Role & SDK Target
Generates a pool of operationally feasible route candidates using Google OR-Tools (`ortools.constraint_solver.pywrapcp`) or exact MILP. Each candidate route represents a single vehicle traveling from the depot through a sequence of customer stops and returning to the depot, respecting vehicle capacity and nominal time windows.

### 2.2 Proposed Interface Specification
```python
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass(frozen=True)
class CustomerStop:
    stop_id: str
    lat: float
    lon: float
    demand: float
    earliest_time: float      # Minutes from dispatch epoch
    latest_time: float        # Contractual SLA deadline
    service_duration: float   # Expected unload/service time

@dataclass(frozen=True)
class CandidateRoute:
    route_id: str
    vehicle_id: str
    vehicle_capacity: float
    ordered_stop_ids: Tuple[str, ...]
    nominal_distance_km: float
    nominal_transit_time_min: float
    nominal_arrival_times: Tuple[float, ...]
    operating_cost: float     # Fuel/labor cost

class RouteGeneratorAdapter:
    def generate_candidate_routes(
        self,
        depot: Tuple[float, float],
        stops: List[CustomerStop],
        vehicle_fleet: Dict[str, float],   # vehicle_id -> capacity
        distance_matrix: List[List[float]],
        time_matrix: List[List[float]],
        max_routes_per_vehicle: int = 5,
        seed: int = 42,
    ) -> List[CandidateRoute]:
        """
        Uses OR-Tools RoutingModel or diverse routing strategies
        to produce a pool of valid candidate routes.
        """
        ...
```

### 2.3 Verified SDK Backing
- Uses `ortools.constraint_solver.pywrapcp.RoutingIndexManager` and `RoutingModel`.
- Solver options: `FirstSolutionStrategy.PATH_CHEAPEST_ARC` or `PARALLEL_CHEAPEST_INSERTION` with local search metaheuristics (`LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH`).
- 100% offline, deterministic, zero-credential execution.

---

## 3. Adapter 2: VQC Delay-Risk Predictor (`VQCPredictorAdapter`)

### 3.1 Role & SDK Target
Constructs and evaluates a Parameterized Quantum Circuit (PQC) using `qpiai_quantum.Circuit` to predict the delay probability or expected SLA violation exposure for each candidate route.

### 3.2 Proposed Interface Specification
```python
@dataclass
class RiskPrediction:
    route_id: str
    expected_delay_min: float
    sla_violation_probability: float  # P(arrival_time > deadline)
    expected_penalty_cost: float      # Probability * contractual SLA penalty
    confidence_interval: Tuple[float, float]

class VQCPredictorAdapter:
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend: str = "statevector",  # 'statevector' locally, 'QCloud' remotely
    ):
        ...

    def build_feature_circuit(
        self,
        features: List[float],          # [route_length, traffic_factor, buffer_slack, demand_ratio]
        parameters: List[float],        # Variational angles
    ) -> Any:  # qpiai_quantum.Circuit
        """
        Builds a parameter-bound qpiai_quantum.Circuit using angle encoding
        (e.g., Ry/Rz rotations) and entangling layers (CX/CZ).
        """
        ...

    def predict_risk(
        self,
        routes: List[CandidateRoute],
        parameters: List[float],
        weather_traffic_context: Dict[str, float],
    ) -> List[RiskPrediction]:
        """
        Executes local StatevectorSimulator to compute expectation values
        Z_0 or computational basis probabilities, mapped to SLA risk.
        """
        ...
```

### 3.3 Verified SDK Backing
- `qpiai_quantum.Circuit(n_qubits)` with `.h()`, `.ry()`, `.rz()`, `.cx()`.
- Local execution via `qpiai_quantum.simulator.StatevectorSimulator().run(qc, shots=shots)`.
- Verified in `tools/sdk_probes/probe_qpiai_quantum.py`.

---

## 4. Adapter 3: Risk-Conditioned Route QUBO (`RouteQUBOAdapter`)

### 4.1 Role & SDK Target
Formulates the binary route-selection decision into an algebraic optimization model using `qpiai_opt.opt_modeller.Model`, applies penalty terms for order coverage and vehicle assignment, and exports to a platform-validated `QUBOProblem`.

### 4.2 Proposed Mathematical Formulation
Let binary variable $x_r \in \{0, 1\}$ indicate selection of candidate route $r \in \mathcal{R}$.
$$\min_{x} \sum_{r \in \mathcal{R}} \left( \text{Cost}_r + \text{RiskPenalty}_r \right) x_r + \lambda_{\text{cov}} \sum_{i \in \text{Stops}} \left( \sum_{r: i \in r} x_r - 1 \right)^2 + \lambda_{\text{veh}} \sum_{v \in \text{Vehicles}} \left( \sum_{r \in \mathcal{R}_v} x_r - 1 \right)^2$$

### 4.3 Proposed Interface Specification
```python
from qpiai_opt.opt_modeller import Model
from qpiai_opt.opt_solvers.qubo.models import QUBOProblem

@dataclass
class QUBOCompilationResult:
    model: Model
    qubo_problem: QUBOProblem
    qubo_dict: Dict[Tuple[int, int], float]
    offset: float
    variable_index_to_route_id: Dict[int, str]
    route_id_to_variable_index: Dict[str, int]

class RouteQUBOAdapter:
    def __init__(self, penalty_coverage: float = 100.0, penalty_vehicle: float = 100.0):
        self.penalty_cov = penalty_coverage
        self.penalty_veh = penalty_vehicle

    def build_qubo(
        self,
        candidates: List[CandidateRoute],
        risks: List[RiskPrediction],
        required_stops: List[str],
        fleet_vehicles: List[str],
    ) -> QUBOCompilationResult:
        """
        Builds algebraic model with Binary variables, adds quadratic objective and penalties,
        and compiles to symmetric/upper QUBO dictionary and QUBOProblem.
        """
        ...

    def decode_solution(
        self,
        bitstring: str,
        compilation: QUBOCompilationResult,
    ) -> Dict[str, Any]:
        """
        Decodes binary bitstring into selected route IDs, evaluates total cost,
        checks order coverage feasibility, and reports violations.
        """
        ...
```

### 4.4 Verified SDK Backing
- `from qpiai_opt.opt_modeller import Model, Binary`
- `m.to_qubo_dict(matrix_format="upper")`
- `QUBOProblem.from_model(m)`
- `m.evaluate_objective(values)` and `m.is_feasible(values)`
- Verified in `tools/sdk_probes/probe_qpiai_opt.py`.

### 4.5 Critical Indexing Invariant
- **Strict Contiguity ($0 \le i < N$):** In `qpiai_opt`, `_canonical_qubo` infers problem qubit count via $\max(\text{index}) + 1$. Non-contiguous index mappings (e.g., indices $\{0, 4\}$ for 2 variables) will cause `QUBOProblem` validation to fail or allocate 5 qubits instead of 2. `RouteQUBOAdapter` **must** guarantee dense, zero-indexed contiguity $0, 1, \dots, N-1$ across all compiled candidate routes.

---

## 5. Adapter 4: Warm-Start QAOA Solver (`QAOASolverAdapter`)

### 5.1 Role & SDK Target
Performs quantum combinatorial optimization using `qpiai_opt.quantum.QAOA`. Initializes initial state and tilted mixer from continuous LP relaxation probabilities using `qpiai_opt.quantum.WarmStart`.

### 5.2 Proposed Interface Specification
```python
@dataclass
class QAOASolverConfig:
    reps: int = 1
    optimizer: str = "cobyla"
    max_iterations: int = 100
    epsilon: float = 0.25                # Warm-start tolerance parameter in [0, 0.5]
    shots: int = 1024
    use_warm_start: bool = True

@dataclass
class QAOASolveOutput:
    optimal_parameters: List[float]
    optimal_energy: float
    best_bitstring: str
    counts: Dict[str, int]
    is_feasible: bool
    decoded_routes: List[str]
    runtime_sec: float

class QAOASolverAdapter:
    def __init__(self, config: QAOASolverConfig):
        self.config = config

    def solve(
        self,
        qubo_compilation: QUBOCompilationResult,
        relaxed_solution: Optional[List[float]] = None,
    ) -> QAOASolveOutput:
        """
        Constructs WarmStart(epsilon=epsilon, relaxed_solution=relaxed_solution).
        Configures QAOA(reps=reps, warm_start=warm).
        Executes local StatevectorSimulator or QpiAI Platform backend.
        Decodes best sampled bitstring.
        """
        ...
```

### 5.3 Verified SDK Backing
- `from qpiai_opt.quantum import QAOA, WarmStart, CVaR, build_circuit`
- `WarmStart(epsilon=0.25, relaxed_solution=relaxed_vector)`
- Parameterized circuit construction verified in `tools/sdk_probes/probe_qpiai_opt.py`.

### 5.4 Verified Mathematical & Relaxation Behaviors
- **Probability Clamping:** In `qpiai_opt.quantum._shared._warm_start_probabilities`, relaxed solution vectors $x^* \in [0, 1]^N$ are automatically clamped to $[\epsilon, 1-\epsilon]$: `clipped = [min(1.0 - epsilon, max(epsilon, float(v))) for v in values]`. This prevents zero-division or degenerate mixer angles for extreme 0 or 1 continuous relaxations.
- **Automatic Fallback Box Relaxation:** If `relaxed_solution is None`, `WarmStart` natively executes an internal L-BFGS-B box relaxation on the continuous Ising objective, providing an offline fallback without external solvers.
- **Standard Parameter Layout:** Under `parameterization="standard"`, the parameter vector expects $2 \times \text{reps}$ entries: $[\gamma_0, \dots, \gamma_{p-1}, \beta_0, \dots, \beta_{p-1}]$.

---

## 6. Unresolved Interface Decisions (Awaiting Codex & Owner Review)

The following design decisions are documented as **unresolved** and must not be unilaterally frozen without coordinator review:

| Gate ID | Open Question | Alternatives | Trade-offs & Recommendation |
|---|---|---|---|
| **GATE-01** | Risk Prediction Target | (A) Single binary probability $P(\text{delay} > \text{slack})$; (B) Continuous expected delay in minutes; (C) Two-parameter distribution (mean, variance). | Recommendation: Start with (A) for simplest circuit mapping; transition to (B) if SLA penalty is piece-wise linear. |
| **GATE-02** | Decision-Focused Gradient Propagation | (A) Analytic gradients through a differentiable surrogate solver; (B) Policy gradient / REINFORCE on sampled routes; (C) Two-stage decoupled training (fit VQC to historical delays first, then optimize dispatch). | Recommendation: Two-stage decoupled training first as a strict baseline; only attempt differentiable surrogate if baseline achieves calibrated risk scores. |
| **GATE-03** | Continuous Relaxation Source for Warm-Start | (A) HiGHS LP solver inside `qpiai-opt`; (B) Google OR-Tools GLOP linear solver; (C) Classical quadratic relaxation via SciPy. | Recommendation: Use HiGHS (`qpiai_opt` native) or OR-Tools GLOP to generate the continuous relaxed vector $x^* \in [0, 1]^N$. |
| **GATE-04** | Penalty Weight Scaling Strategy | (A) Fixed hand-tuned penalties $\lambda_{\text{cov}}, \lambda_{\text{veh}}$; (B) Adaptive spectral scaling based on largest eigenvalue of cost Hamiltonian. | Recommendation: Implement spectral scaling to ensure constraint violation penalties strictly exceed any potential route cost savings. |
| **GATE-05** | Solution Infeasibility Repair Mechanism | (A) Reject infeasible bitstrings; (B) Deterministic greedy repair heuristic; (C) Fall back to classical OR-Tools solution if QAOA samples zero feasible bitstrings. | Recommendation: (C) Fall back to classical solution with explicit reporting of quantum feasibility rate. |
