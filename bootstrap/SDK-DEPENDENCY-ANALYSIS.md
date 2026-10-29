# SDK Dependency and Version Compatibility Analysis

**Audit Date:** 2026-10-28 (Asia/Kolkata)  
**Author:** Antigravity (Implementation Support Agent)  
**Status:** Verified via direct PyPI JSON metadata inspection and wheel archive byte-level symbol analysis.

---

## 1. Executive Summary

The DebugLeaf project requires three core software development kits:
1. **QpiAI Opt SDK** (`qpiai-opt`) — for mathematical modeling, QUBO/HUBO compilation, and classical/quantum solver orchestration.
2. **QpiAI Quantum SDK** (`qpiai-quantum`) — for circuit construction, local simulation (`StatevectorSimulator`), and QCloud execution.
3. **Google OR-Tools** (`ortools`) — for classical vehicle routing (VRPTW) baselines, candidate route generation, and MILP validation.

A critical version discrepancy was previously documented in `bootstrap/SDK-EVIDENCE.md`:
> `qpiai-opt 0.0.9`'s optional extra `[quantum]` explicitly pins `qpiai-quantum==0.3.0`, whereas the latest standalone release of `qpiai-quantum` on PyPI is `0.7.5`.

This analysis reports the exact technical root cause of this discrepancy, wheel availability across platforms, runtime symbol compatibility, and provides two verified dependency specifications.

---

## 2. Published Package Metadata (PyPI Inspection)

All observations were fetched and verified from PyPI on 2026-10-28:

| Package | Latest Release | Release Date | Requires-Python | Available Distributions |
|---|---|---|---|---|
| `qpiai-quantum` | `0.7.5` | 2026-10-13 | `>=3.10` | Pure Python wheel (`py3-none-any.whl`) + sdist (`.tar.gz`) |
| `qpiai-quantum` | `0.3.0` | 2026-07-29 | `>=3.10` | Pure Python wheel (`py3-none-any.whl`) + sdist (`.tar.gz`) |
| `qpiai-opt` | `0.0.9` | 2026-10-26 | `<3.15,>=3.10` | 25 binary platform wheels (No sdist available) |
| `ortools` | `9.15.6755` | 2026-09-24 | `>=3.9` | 34 binary platform wheels (No sdist available) |
| `highspy` | `1.15.1` | 2026-08-20 | `>=3.9` | Binary platform wheels |

### Wheel Matrix Details for `qpiai-opt 0.0.9`
`qpiai-opt` publishes pre-compiled binary extensions (`_local_runtime.pyd` on Windows, `.so` on Linux/macOS) for:
- **Python Versions:** 3.10, 3.11, 3.12, 3.13, 3.14
- **Platforms:** `win_amd64`, `manylinux_2_26_x86_64`, `manylinux_2_26_aarch64`, `macosx_13_0_x86_64`, `macosx_13_0_arm64`

### Transitive Wheel Constraints
While `qpiai-opt` supports Python 3.10 through 3.14, its required quantum and optimization dependencies impose narrower constraints:
- `cvxpy==1.9.3` (required by `qpiai-quantum`): Binary wheels exist for Python 3.10, 3.11, and 3.12. Python 3.14 currently lacks pre-built wheels and fails to compile without C/C++ build tools.
- `numba==0.68.1` (required by `qpiai-quantum`): Binary wheels exist for Python 3.10–3.12.
- **Conclusion:** **Python 3.12 (CPython 3.12.x 64-bit)** is the only verified, mutually compatible interpreter version for all packages across both Windows and Linux.

---

## 3. Discrepancy Analysis: `qpiai-opt[quantum]` vs `qpiai-quantum 0.7.5`

### 3.1 Root Cause in Wheel Metadata
Inspection of `qpiai_opt-0.0.9.dist-info/METADATA` reveals:
```ini
Requires-Dist: numpy>=1.20.0
Requires-Dist: highspy>=1.15.1
Requires-Dist: pydantic>=2.0.0
Requires-Dist: requests>=2.30.0
Requires-Dist: scipy>=1.9.0
Provides-Extra: ommx
Requires-Dist: ommx<3,>=2.6; extra == "ommx"
Provides-Extra: quantum
Requires-Dist: qpiai-quantum==0.3.0; extra == "quantum"
Provides-Extra: all
Requires-Dist: ommx<3,>=2.6; extra == "all"
Requires-Dist: qpiai-quantum==0.3.0; extra == "all"
```
The `pyproject.toml` of `qpiai-opt` pinned `qpiai-quantum==0.3.0` during development in July 2026. When `qpiai-opt 0.0.9` was released on 2026-10-26, the `[quantum]` extra pin was not bumped to match the newly released `qpiai-quantum 0.7.5` (2026-10-13).

### 3.2 Symbol-Level Runtime Audit
Direct inspection of the compiled wheel code of `qpiai_opt` demonstrates where and how `qpiai_quantum` is invoked:
1. `qpiai_opt/quantum/__init__.py`:
   ```python
   try:
       from qpiai_quantum import Circuit, QpiAIQuantumAuth
       from qpiai_quantum.applications.matter.models import QubitHamiltonian
   except ImportError: ...
   ```
2. `qpiai_opt/quantum/_shared.py`:
   ```python
   from qpiai_quantum import Circuit, QpiAIQuantumAuth
   from qpiai_quantum.applications.matter.models import QubitHamiltonian
   from qpiai_quantum.simulator import StatevectorSimulator
   from qpiai_quantum.algorithms.utils.optimizers import ...
   ```
3. `qpiai_opt/quantum/algorithms.py`:
   ```python
   from qpiai_quantum.algorithms.opt.solvers import VQESolver
   ```

Inspection of `qpiai-quantum 0.7.5` confirms that:
- `qpiai_quantum.Circuit` is exported.
- `qpiai_quantum.QpiAIQuantumAuth` is exported.
- `qpiai_quantum.applications.matter.models.QubitHamiltonian` is present.
- `qpiai_quantum.simulator.StatevectorSimulator` is present.
- `qpiai_quantum.algorithms.opt.solvers.VQESolver` is present.
- `qpiai_quantum.algorithms.opt.solvers.qaoa.QAOASolver` is present.
- All additions in 0.7.5 (`creditestimator`, `icr/decomposition.py`, `jobmanager/result_contract.py`) are strictly additive.

### 3.3 Resolution Strategy
There are two viable installation strategies:

1. **Option 1 (Recommended): Unpinned Extra Installation**
   Install `qpiai-opt==0.0.9` and `qpiai-quantum==0.7.5` directly without requesting the `[quantum]` extra token:
   ```bash
   pip install qpiai-opt==0.0.9 qpiai-quantum==0.7.5 ortools==9.15.6755
   ```
   - **Why this works:** `qpiai-opt` core does not declare a mandatory dependency on `qpiai-quantum`. Omitting `[quantum]` from the pip invocation avoids triggering the strict `==0.3.0` metadata constraint in pip's dependency resolver.
   - **Runtime behavior:** At runtime, `import qpiai_opt.quantum` succeeds completely because all required classes exist in `0.7.5`.
   - **Specification file:** `bootstrap/requirements-sdk.txt`.

2. **Option 2 (Strict Vendor Extra): Strict 0.3.0 Pinning**
   Install with `[quantum]` extra, allowing pip to install `qpiai-quantum==0.3.0`:
   ```bash
   pip install "qpiai-opt[quantum]==0.0.9" ortools==9.15.6755
   ```
   - **Why this works:** Satisfies 100% of vendor metadata declarations.
   - **Limitation:** Misses bug fixes and `creditestimator` features present in 0.7.5.
   - **Specification file:** `bootstrap/requirements-sdk-strict.txt`.

---

## 4. Portability to Lenovo Legion

To ensure seamless reproduction on the remote Legion:
1. **Interpreter:** Require CPython 3.12 (64-bit). Both Windows (`win_amd64`) and Linux (`manylinux_2_26_x86_64`) have full binary wheel support for all three SDKs.
2. **Environment Isolation:** Use a project-local virtual environment (`.venv/`) or uv-managed environment.
3. **Execution Policy (Windows):** If Legion runs Windows with Application Control (WDAC), run scripts via the registered python executable rather than invoking user-directory binaries directly.
4. **Offline / CPU Default:** All local capability probes (`run_all_probes.py`) execute in CPU fallback mode without requesting credentials.
