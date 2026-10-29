# SDK Evidence Audit

**Checked:** 2026-10-28 (Asia/Kolkata)  
**Scope:** Public first-party SDK repositories, official documentation links, PyPI JSON metadata, and wheel archive symbol inspections.

**Pinned repository refs:** `qpiai/quantum-sdk` `db93cfff54182ac0c5fc5016de26d46b4eed5151` (`main`); `qpiai/qpiai-opt` `b13785a00314da8bfb450f182b2e97b3012bd9ca` (`main`); `google/or-tools` `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5` (`stable`). These are read-only `git ls-remote` observations on the check date.

**Published metadata (Verified 2026-10-28):**
- `qpiai-quantum`: `0.7.5` (Released 2026-10-13, `Requires-Python >=3.10`, pure Python wheel `py3-none-any.whl`)
- `qpiai-opt`: `0.0.9` (Released 2026-10-26, `Requires-Python <3.15,>=3.10`, binary wheels for Windows, Linux, macOS on CPython 3.10–3.14)
- `ortools`: `9.15.6755` (Released 2026-09-24, `Requires-Python >=3.9`, binary wheels for Windows and Linux on CPython 3.10–3.14)
- `highspy`: `1.15.1` (Released 2026-08-20, `Requires-Python >=3.9`, binary wheels available for CPython 3.10–3.14)

---

## Verified from primary sources & wheel archives

| Component | Source evidence | What it establishes |
|---|---|---|
| QpiAI Quantum SDK | [PyPI JSON](https://pypi.org/pypi/qpiai-quantum/json), wheel archive inspection (`0.7.5` and `0.3.0`) | Repository install instruction is `pip install qpiai-quantum`. Exports top-level `Circuit`, `QpiAIQuantumAuth`, and `StatevectorSimulator`. Contains `qpiai_quantum.algorithms.opt.solvers.qaoa.QAOASolver` with ansatz (`standard`, `hardware_efficient`), mixer (`x`, `xy`), and optimizers (`adam`, `spsa`, `cobyla`). Contains `qpiai_quantum.applications.matter.models.QubitHamiltonian`. Pure Python wheel runs across platforms. |
| QpiAI Opt SDK | [PyPI JSON](https://pypi.org/pypi/qpiai-opt/json), wheel archive inspection (`0.0.9`) | Distribution `qpiai-opt`, import `qpiai_opt`. Provides algebraic modeling (`opt_modeller.Model`), binary variables (`Binary`), and QUBO compilation (`to_qubo_dict`). Contains native `qpiai_opt.quantum` module providing `QAOA`, `WarmStart`, `ADAPTQAOA`, `RecursiveQAOA`, `CVaR`, `XYMixer`, and `build_circuit`. Direct byte inspection reveals `WarmStart` takes continuous relaxation probabilities (`relaxed_solution`) and `epsilon` tolerance to configure tilted mixers and initial states. Local solving through `Client` requires online key verification. |
| Google OR-Tools | [PyPI JSON](https://pypi.org/pypi/ortools/json), wheel archive inspection (`9.15.6755`) | Provides CP-SAT (`ortools.sat.python.cp_model`) and Vehicle Routing (`ortools.constraint_solver.pywrapcp`). Fully self-contained, 100% offline local solver; executes without internet or API credentials. |

---

## Resolved Version Discrepancies

- **The `qpiai-opt[quantum]` vs `qpiai-quantum 0.7.5` Skew:**  
  `qpiai-opt 0.0.9` declared `Requires-Dist: qpiai-quantum==0.3.0; extra == "quantum"` in its package metadata. However, wheel inspection reveals that `qpiai_opt` base has **no** mandatory dependency on `qpiai-quantum`; `qpiai_opt.quantum` dynamically imports `Circuit`, `QpiAIQuantumAuth`, and `QubitHamiltonian` only when quantum routines are invoked. All required symbols exist and are backwards-compatible in `qpiai-quantum 0.7.5`.
- **Installation Resolution:** Installing `qpiai-opt==0.0.9` and `qpiai-quantum==0.7.5` directly (without requesting `[quantum]`) avoids pip resolution failure and operates correctly. Alternatively, `qpiai-opt[quantum]==0.0.9` can be used to pin strictly to `0.3.0`. Full audit is recorded in `bootstrap/SDK-DEPENDENCY-ANALYSIS.md`.
- **Target Python Version:** CPython 3.12 (64-bit) is the primary recommended runtime. While `qpiai-opt` and `ortools` have Python 3.14 wheels, transitive scientific dependencies of `qpiai-quantum` (`cvxpy 1.9.3` and `numba 0.68.1`) currently lack Python 3.14 binary wheels.

---

## Verified Capabilities vs. Remaining Unknowns

### Verified Capabilities
- **QAOA in QpiAI SDKs:** Confirmed present in both `qpiai_quantum.algorithms.opt.solvers.qaoa` and `qpiai_opt.quantum.QAOA`.
- **Warm-Start QAOA:** Confirmed present in `qpiai_opt.quantum.WarmStart`. Accepts continuous relaxation vectors $x^* \in [0, 1]^N$ and generates tilted mixers for QAOA circuits.
- **Local Simulation:** `qpiai_quantum.simulator.StatevectorSimulator` executes locally using NumPy and multi-core Numba acceleration without authentication or internet access.
- **QUBO / Model Compilation:** `qpiai_opt.opt_modeller.Model` compiles algebraic formulations to symmetric and upper-triangular QUBO dictionaries and `QUBOProblem` platform objects offline.

### Remaining Unknowns / Blockers
- **Supervised VQC Autodiff:** `qpiai_quantum` provides optimizers and parameter-shift simulation, but no scikit-learn compatible estimator or automatic reverse-mode differentiation framework. Gradients for decision-focused learning must be handled via parameter-shift or external surrogate.
- **QCloud Execution:** Indus-1 QPU execution, credit estimation, and queue latency remain unverified and require authenticated access.
- **QpiAI Opt Local License:** QpiAI Opt solver execution (`Client.solve`) requires online license/API key verification. Offline solving is limited to OR-Tools and local quantum simulation.

---

## Primary references

- https://github.com/qpiai/quantum-sdk
- https://docs.qcloud.qpiai.tech/
- https://github.com/qpiai/qpiai-opt
- https://opt-docs.qpiai.tech/
- https://github.com/google/or-tools
- https://developers.google.com/optimization/install/python
- `bootstrap/SDK-DEPENDENCY-ANALYSIS.md`
- `docs/SDK-INTERFACE-REQUIREMENTS.md`
