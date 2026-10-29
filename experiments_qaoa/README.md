# Tiny QUBO and QAOA slice

Run from the repository root with `python experiments_qaoa/run_tiny.py`. The
script uses a four-route synthetic fixture, checks all 16 assignments, then runs
standard and continuous-relaxation warm-start one-layer QAOA with the same seed,
512 shots, and 10x10 angle grid. Exact statevector evolution is a pure-Python CPU
simulation. This experiment does not invoke QpiAI Quantum, QpiAI Opt, QCloud,
an authenticated API, or QPU hardware.

Mechanism:

- [backend/qubo.py](../backend/qubo.py): bit `x_i` selects candidate route `i`
  in listed variable order. For each stop, `P_cov (sum x_i - 1)^2` enforces
  exactly one covering route. For each vehicle, `P_veh sum_{i<j} x_i x_j`
  forbids selecting two routes on the same vehicle while allowing no route for
  a vehicle. QUBO energy includes route cost, penalties, and explicit constant
  offset. `validate_selection` recomputes stop counts, vehicle duplication,
  and base route cost without trusting energy.
- [backend/qaoa.py](../backend/qaoa.py): box-relaxes the same polynomial with
  projected gradient descent. Its probabilities initialize product amplitudes
  `sqrt(1-p_i)|0> + sqrt(p_i)|1>`. Warm mixer at variable `i` uses
  `(2 sqrt(p_i(1-p_i))) X + (1-2p_i) Z`, whose +1 eigenstate is that initial
  state. For example `p=0.25` gives mixer axis `(sqrt(3)/2) X + 0.5 Z`.
  The standard path starts uniformly and uses X mixers. Both paths phase by the
  same QUBO/Ising energy, use an equal angle grid and shared sampling seed, and
  retain each raw sample, independent validation, repair edits, and repaired
  validation separately.
- [backend/qaoa_dispatch.py](../backend/qaoa_dispatch.py) connects generated
  route candidates to the selector QUBO. It validates distinct raw and repaired
  bitstrings with `backend.validator.validate_plan`. Only raw samples that pass
  both QUBO checks and domain validation can be selected. Otherwise it reports
  exact classical fallback or infeasibility explicitly.
- [backend/qpiai_adapter.py](../backend/qpiai_adapter.py) builds the route QUBO
  through QpiAI Opt `Model`/`QUBOProblem`, passes the relaxation to `WarmStart`,
  builds a QAOA circuit, and invokes QpiAI Quantum's local simulator. It does
  not read credentials or create a remote client.

## Evidence status

`tiny_results.json` is generated output, not prefilled metrics. The result
records exact variable order, QUBO/Ising coefficients and offset, all sampled
bitstrings/counts, relaxation probabilities, circuit angles, and raw versus
repaired feasibility/cost. Synthetic results illustrate mechanics only and do
not establish fleet feasibility, QpiAI integration, calibrated risk, novelty,
or quantum advantage.

`qpiai_warm_start_results.json` records an actual 8-variable QpiAI local
simulator run on `dispatch_fixture.json`, using the dispatch selector's route
pool and QUBO, continuous-relaxation vector, and angle pair. The output-bit
direction was checked with distinct per-variable warm probabilities and zero
cost/mixer angles (2,048 shots). Its 128 raw measurement shots were decoded;
each distinct outcome has separate QUBO and independent domain validation. The
best feasible measured sample was the raw `10000000` outcome, decoded as
variable-order `00000001` (route `V2:7:B,A`), observed twice; independent
recomputed objective was 4.0. QpiAI Opt's online
`Client` was not invoked; no API, cloud simulator, or QPU job was submitted.
The generated model's upper-triangle coefficients and offset matched the
source QUBO exactly for all 256 assignments. The adapter now refuses more than
8 QpiAI variables: both the compiled coefficient mapping and printed-bit order
were verified only at this 8-variable size.

Test discovery/evidence: the first unconfigured repository-wide `pytest -q`
attempted to collect preserved vendored tests under `_archive/` and failed on
15 import/path collection errors. `_archive/` was left untouched. A scoped run
of `tests/test_domain.py`, `tests/test_qubo.py`, and `tests/test_qaoa_dispatch.py`
passed 22 tests. Root `pytest.ini` now limits discovery to `tests/`, excludes
archives/caches/environments, and points temporary files into an ignored
workspace runtime directory because the sandbox blocked pytest's default
Windows Temp path. After that change, repository-wide `pytest -q` passed 70
tests.

## Refinement record (required cycles)

| Cycle | Failure hypothesis | Material change and retest | Evidence pointer | Remaining concern |
|---|---|---|---|---|
| Q1 | An exactly-one-per-vehicle penalty wrongly requires every fleet vehicle to be assigned; a weak check could miss duplicate vehicle selection. | Encode pairwise at-most-one conflicts and correct the validator fixture so two same-vehicle routes really cover the required stops; rerun direct objective checks over all 16 bitstrings. | `backend/qubo.py`, corresponding exhaustive and duplicate-vehicle tests; `python -m pytest tests/test_qubo.py -q` → 5 passed after the first correction; final suite → 7 passed. | Route-level capacity/time-window validation is outside this selector contract. |
| Q2 | A QUBO-to-Ising conversion could preserve rankings but lose its constant offset or bit order. | Return `c,h,J`; enumerate direct energy equality on every bitstring and add a strict route-order decoder. | `backend/qubo.py::qubo_to_ising`, `decode_bitstring`, `backend/qaoa.py::check_ising_equivalence`; `python -m pytest tests/test_qubo.py -q` → final suite 7 passed. | No SDK Hamiltonian adapter is asserted. |
| Q3 | Warm-start probabilities could be decorative and fail to alter the circuit state/mixer; direct script execution could fail to reproduce the experiment. | Use continuous relaxation values in product-state amplitudes and probability-dependent Pauli-axis rotations, check the mixer eigenstate analytically, and insert repository root for script execution. Paired standard/warm sampling used identical seed and shot count. | `backend/qaoa.py::qaoa_state`, `run_qaoa`; `experiments_qaoa/run_tiny.py`; initial tests passed 5/5, later owned suite 14/14; local matched run returned 231/512 vs 384/512 feasible samples and cost 4.0 for both. | Small synthetic single-layer statevector evidence only. |
| Q4 | Using the QUBO offset instead of the converted Ising constant changes simulator amplitudes by a global phase; integer coercion could silently truncate fractional decisions. | Use returned Ising `c` in phase evolution, compare basis amplitudes against `exp(-i gamma E(x))`, reject fractional bits, and enforce shot/angle-grid bounds. | `backend/qaoa.py::qaoa_state`, `backend/qubo.py::validate_selection`; offset, fractional-bit and budget tests; owned suite 14 passed. | Statevector cap is 12 variables; experiment remains synthetic. |
| Q5 | A QUBO-feasible sample might not be an operationally feasible dispatch, and QpiAI count strings might be reversed relative to route variable order. | Independently validate raw/repaired route sets; use explicit exact fallback; verify output direction with asymmetric relaxation probabilities before decoding. | `backend/qaoa_dispatch.py`, `backend/qpiai_adapter.py`, `tests/test_qaoa_dispatch.py`; SDK run returned 11/128 independently valid raw sample shots, including the selected 4.0 route; bit order probe reversed MAE 0.00635 vs 0.28142 direct. | Tiny synthetic fixture; no production fleet validation or risk calibration. |
| Q6 | QpiAI Opt could reorder candidate variables or compile a different QUBO even if a circuit builds; variable names `x0` through `x12` are not guaranteed lexicographic/index aligned. | Limit adapter to 8 variables, compare the SDK compiled objective terms plus offset to the source QUBO for all 256 assignments, and test the refusal boundary above 8. | `backend/qpiai_adapter.py::run_qpiai_warm_start`; `tests/test_qubo.py::test_qpiai_adapter_refuses_unverified_variable_order_above_eight`; `qpiai_warm_start_results.json` reports `exhaustive_match`, 256 assignments, 8 variables. | No SDK mapping claim beyond the tested 8-variable model. |

## QpiAI boundary

The reviewed `bootstrap/SDK-EVIDENCE.md` documents QpiAI Quantum `0.7.5` and
QpiAI Opt `0.0.9`; the isolated SDK runtime used QpiAI Quantum `0.3.0` with Opt
`0.0.9` to match the Opt quantum extra. QpiAI Opt `Client.solve` needs online
key verification. This work executed local circuit construction and
`StatevectorSimulator`; SDK QUBO coefficients and constant offset were
exhaustively matched to the source route QUBO. The adapter hard-limits itself
to this verified 8-variable range; cloud and QPU paths were not called.
