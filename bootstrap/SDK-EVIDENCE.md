# SDK Evidence Audit

**Checked:** 2026-10-28 (Asia/Kolkata)
**Scope:** Public first-party SDK repositories and official documentation links. This is a documentation/source audit, not an installation or execution result. Repository `main` is not assumed to equal the latest published wheel.

**Pinned repository refs:** `qpiai/quantum-sdk` `db93cfff54182ac0c5fc5016de26d46b4eed5151` (`main`); `qpiai/qpiai-opt` `b13785a00314da8bfb450f182b2e97b3012bd9ca` (`main`); `google/or-tools` `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5` (`stable`). These are read-only `git ls-remote` observations on the check date, not installed source revisions. The web-extracted source snapshots below were fetched from moving branch URLs and were not independently matched to these pinned SHAs.

**Published metadata:** Read-only [PyPI JSON](https://pypi.org/pypi/qpiai-quantum/json), [PyPI JSON](https://pypi.org/pypi/qpiai-opt/json), and [PyPI JSON](https://pypi.org/pypi/ortools/json) observations on 2026-10-28 (Asia/Kolkata): `qpiai-quantum` 0.7.5 (`Requires-Python >=3.10`); `qpiai-opt` 0.0.9 (`>=3.10,<3.15`); `ortools` 9.15.6755 (`>=3.9`). Versions are release metadata, not proof of successful installation or target-platform wheel availability.

## Verified from primary sources

| Component | Source evidence | What it establishes |
|---|---|---|
| QpiAI Quantum SDK | [Repository README](https://github.com/qpiai/quantum-sdk), [pyproject.toml](https://raw.githubusercontent.com/qpiai/quantum-sdk/main/pyproject.toml), [package exports](https://raw.githubusercontent.com/qpiai/quantum-sdk/main/qpiai_quantum/__init__.py), [algorithm exports](https://raw.githubusercontent.com/qpiai/quantum-sdk/main/qpiai_quantum/algorithms/__init__.py) | Repository install instruction is `pip install qpiai-quantum`; README says Python 3.10+ and describes local statevector, density-matrix, tensor-network simulators, QCloud and Indus access, `JobManager`, and circuit execution/results. The retrieved moving-branch pyproject snapshot declares package version `0.1.41`, `requires-python >=3.10` and classifiers 3.10–3.12. The separately pinned `main` SHA above is not claimed to be the source of this snapshot. The retrieved public package exports include `Circuit`, `Backend`, `JobManager`, `JobResult`, `QpiAIQuantumAuth` and `VQESolver`. The listed algorithms include VQE, but no QAOA. |
| QpiAI Opt SDK | [Repository README](https://github.com/qpiai/qpiai-opt), [Opt documentation](https://opt-docs.qpiai.tech/) | README says distribution `qpiai-opt`, import namespace `qpiai_opt`, Python 3.10–3.14 on Windows/macOS/Linux, and install `python -m pip install qpiai-opt`. It describes local LP/MILP/QP/QUBO/HUBO/Ising solving, with a native binary local runtime; the `[quantum]` extra pins `qpiai-quantum==0.3.0`. Opt local solve through its `Client` performs online Opt-key verification. Gate-based optimization requires QCloud authentication. |
| Google OR-Tools | [Repository](https://github.com/google/or-tools), [Python installation guide](https://developers.google.com/optimization/install/python) | Official README lists CP-SAT, LP, VRP/TSP, graph algorithms and Python wrappers. Official Python installation guide recommends published pip packages for typical use; platform-specific install pages are linked there. |

## Version and compatibility cautions

- The retrieved Quantum SDK moving-branch pyproject snapshot (`0.1.41`) differs from current PyPI JSON metadata (`0.7.5`); the source file was not fetched by immutable SHA, so this difference is a snapshot/release comparison, not a pinned-commit finding.
- The Opt README's optional `[quantum]` dependency specifies `0.3.0`, while PyPI JSON reported Quantum `0.7.5` and Opt `0.0.9`. These observations do not establish whether the pin is compatible with the standalone package. Resolve the pair in the isolated target environment before relying on either.
- Opt source README's `[quantum]` extra pins `qpiai-quantum==0.3.0`, while current PyPI JSON reports Opt `0.0.9` and Quantum `0.7.5`. This is a concrete compatibility question; investigate package metadata and integration API before choosing a pair.
- OR-Tools current PyPI metadata was observed as `9.15.6755`, but wheel availability for the target Python/OS/architecture remains unverified.

## Not established / blocked

- **QAOA in QpiAI Quantum:** no QAOA is shown in the repository README's algorithm inventory or current public top-level algorithm exports inspected here. That does not prove no implementation exists elsewhere in the source tree or through an Opt integration.
- **Warm-start QAOA / warm-start inputs:** not established by the inspected Quantum README or Opt README. No warm-start support claim should be made until the exact current SDK API and backend support are checked.
- **VQC and autodifferentiation:** VQE is exported, but this alone says nothing about supervised VQC or gradients. The inspected docs and public exports do not establish parameter-shift, autodiff framework integration, gradient availability on simulators, or gradient availability on QCloud/QPU. Keep all such capabilities **unknown** pending API and execution evidence.
- **Backend names/capabilities, shot/noise controls, QAOA variants, and credential-free smoke execution:** not fully checked. QCloud documentation homepage extraction returned no usable content; Opt docs homepage and linked manuals were inaccessible to the research browser. No credentials were accessed.
- **Artifact hashes and wheel availability:** PyPI project JSON versions were checked, but per-file hashes and the target Legion's compatible wheel tags have not been recorded. Resolve these in the isolated target environment.
- **Execution:** no package installation, SDK import, local solver call, cloud request, simulator run, QPU run, or benchmark was performed.

## Reproducibility / next evidence needed

1. On the Legion, inventory OS, architecture, Python interpreters, GPU and available disk before choosing environment markers or optional acceleration.
2. In an isolated project environment, resolve and record exact versions/hashes for QpiAI Quantum, QpiAI Opt and OR-Tools. Investigate the Opt `[quantum]` version pin explicitly; do not repair incompatibilities by silently upgrading/downgrading.
3. Inspect source and official API reference at pinned commits/releases for QAOA, initialization/warm-start controls, differentiation and backend execution settings. Keep documentary capability claims separate from smoke-test evidence.
4. Any later smoke should be a tiny local circuit/import/model validation where supported. QCloud/Opt execution remains outside this bootstrap stage and needs configured access; any later smoke evidence must be recorded separately.

## Primary references

- https://github.com/qpiai/quantum-sdk
- https://docs.qcloud.qpiai.tech/
- https://github.com/qpiai/qpiai-opt
- https://opt-docs.qpiai.tech/
- https://github.com/google/or-tools
- https://developers.google.com/optimization/install/python
