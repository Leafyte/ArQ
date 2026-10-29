# V1 local runbook

This runbook describes the current Windows local prototype and the commands used for its evidence. It is not a deployment guide. Keep the API on localhost, keep QpiAI SDK calls in the backend, and do not infer cloud/QPU execution from simulator results.

## Runtime environments

Observed interpreter: CPython **3.12.14**, Windows x86-64. The default WindowsApps `python.exe` alias did not start; use the project virtual environments below. Current runtime versions were read from package metadata:

| Environment | Observed packages |
|---|---|
| `.venv` core/API and tests | FastAPI 0.162.0; Pydantic 2.14.1; Uvicorn 0.54.0; HTTPX 0.28.1; pytest 9.2.0; NumPy 2.5.4; SciPy 1.18.1 |
| `.venv-sdk` vendor experiment | QpiAI Opt 0.0.9; QpiAI Quantum 0.3.0; OR-Tools 9.15.6755; NumPy 2.5.4; SciPy 1.18.1; pytest 9.2.0 |

The SDK pin is intentionally the strict `qpiai-opt[quantum]` pairing: Opt 0.0.9 declares Quantum 0.3.0 for its `quantum` extra. The locally executed adapter uses `StatevectorSimulator`; Opt's online `Client.solve` is not used.

### Create the isolated environments

From the repository root in PowerShell, with `uv` available:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe `
  "fastapi==0.162.0" "pydantic==2.14.1" "uvicorn==0.54.0" `
  "httpx==0.28.1" "pytest==9.2.0" "numpy==2.5.4" "scipy==1.18.1"

uv venv --python 3.12 .venv-sdk
uv pip install --python .venv-sdk/Scripts/python.exe `
  -r bootstrap/requirements-sdk-strict.txt
```

These commands describe the observed package set and reviewed strict SDK file. They have not been rerun as a clean-room install for this report. Standard-library VQC/QAOA code does not call NumPy; the API/tests use the `.venv` environment, and the vendor experiment uses `.venv-sdk`.

## Run checks and experiments

Run from the repository root. The following checks do not overwrite the frozen
prospective artifact:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m experiments.run_quantum_learning
.\.venv\Scripts\python.exe -m experiments.run_dispatch_vqc
.\.venv\Scripts\python.exe experiments_qaoa\run_tiny.py
.\.venv-sdk\Scripts\python.exe experiments_qaoa\run_qpiai_warm_start.py
.\.venv-sdk\Scripts\python.exe -m pytest tests/test_qpiai_vqc.py -q --basetemp=.runtime\pytest-sdk-vqc
.\.venv\Scripts\python.exe -m pytest --basetemp=.runtime\pytest-risk tests/test_domain.py tests/test_api.py tests/test_recovery.py tests/test_risk.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_strong_baseline.py -q
```

`pytest.ini` limits discovery to `tests/`, excludes archives, environments and caches, and sets a workspace temp directory because the sandbox blocked the Windows default. Latest full command `.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime\pytest-final-parent-6` passed **151 tests, skipped 1, and passed 18 subtests in 28.51 seconds** (checkpoint 19:21 UTC). SDK checks were separate: QPiAI VQC 6 passed; SDK inference 8 passed and 3 subtests passed. Older full-suite snapshots are superseded.

The prospective run overwrites `experiments/artifacts/prospective-dispatch-results.json`. That frozen result uses protocol SHA `60CDC7A0B6E676B2BE9FFA7928BC0BE505BBF9870B9D65CF48D833F935AD86F8` and was generated before the zero-penalty label correction; do **not** rerun into the same path or alter the held-out result. The correction now labels any positive lateness, including zero-penalty lateness, and `tests/test_quantum_learning.py::test_prospective_sla_label_uses_lateness_even_when_penalty_is_zero` passes. The historical artifact does not evaluate corrected labels. The separate preregistered V2 comparison uses penalty-independent lateness labels and was run once; its metrics and limitations are in [`experiments_strong_baseline/README.md`](../experiments_strong_baseline/README.md) and the report. Do not tune from or overwrite `experiments_strong_baseline/results.json`. The scikit-learn V1 comparator and a production-grade or compute-matched stronger comparator are **NOT RUN**.

V2 protocol SHA is `18D870457CE46F17FD2D132C5902B09EFFAB3134633C4CB8FDDA0CE013F1051B`. On its 30 held-out groups, mean regret was 0.15796 (decision VQC), 0.14065 (predictive VQC), 0.14084 (logistic), and 0.22329 (grouped-bootstrap CART). Across five replicate mean regrets, sample SD/range were 0.12976 / 0.05716–0.37390, 0.09483 / 0.05716–0.28736, 0.12644 / 0.04980–0.35898, and 0.31896 / 0.04980–0.79085, respectively. VQC used 120 updates, logistic 500, and the fixed local forest 64 depth-3 trees; budgets differ and the forest is not production-validated. All methods had valid plans; no superiority claim follows.

The two QAOA experiments have separate evidence labels:

- `run_tiny.py`: custom pure-Python exact statevector, synthetic 4-route fixture, exhaustive 16 assignments, standard and warm-start with seed `20261028`, 512 shots per mode, and 10x10 angle grid. It does not import QpiAI.
- `run_qpiai_warm_start.py`: QpiAI Opt 0.0.9 / Quantum 0.3.0 local simulator on an 8-variable route QUBO, 128 measurement shots plus a 2,048-shot bit-order probe. It exhaustively compares SDK objective coefficients/offset over all 256 assignments and independently validates decoded route samples. Adapter mapping is capped at 8 variables.

The decision-focused QpiAI VQC path also has an executed SDK gradient/update check:
6 tests passed, gradient `[0.5992162396800853,0.4446328968053965]`, centered
finite-difference maximum error `8.21e-11`, and one 0.08-rate update reduced
the synthetic training loss from `3.2409607395` to `3.1948195973`. This is an
exact local statevector test on four synthetic training rows, not held-out or
finite-shot learning evidence. Do not attribute it to Opt online solve.

Raw results are in [`experiments/artifacts/`](../experiments/artifacts/) and [`experiments_qaoa/`](../experiments_qaoa/). They are synthetic examples and should not be presented as operating results.

## Start the local API

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Use a second PowerShell window for local requests. For example:

```powershell
$scenario = Get-Content fixtures/dispatch_tiny.json -Raw | ConvertFrom-Json
$body = $scenario | ConvertTo-Json -Depth 30
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/scenarios/plan `
  -ContentType 'application/json' -Body $body
```

The caller-supplied VQC path is `POST /api/v1/scenarios/vqc-plan` with `forecast_scenario`, `nominal_scenario`, `theta`, and `model_version`. Optional `execution_backend` is `reference` by default or `qpiai_local`; a requested QPiAI path does not silently fall back. Inference neither trains nor loads a registered model; version and parameters are caller supplied and provenance is unverified. The synchronous QAOA endpoint is `POST /api/v1/scenarios/qaoa`; exact fallback is labeled. Async QAOA jobs use `POST /api/v1/jobs/qaoa` and poll with `GET /api/v1/jobs/{id}`. One coordinator-observed job on port 18378 succeeded using the local Python statevector simulator, seed 7, 64 shots, warm start and angle grid 4; its raw-sample decision passed independent validation and no fallback occurred. This does not establish QpiAI execution or QAOA-specific restart recovery. Generic `POST /api/v1/jobs` classical job records default to `.runtime/debugleaf.sqlite3`.

`POST /api/v1/scenarios/financial-plan` requires `scenario`, `currency`,
`operating_cost_per_travel_minute`, and
`penalty_rate_unit: "currency_per_late_minute"`. Example terms are
`{"currency":"USD","operating_cost_per_travel_minute":0.5,"penalty_rate_unit":"currency_per_late_minute"}`;
the rate and currency must come from the caller, with no defaults. The
response's `financial_objective` is distinct from the domain proxy objective
and is independently recomputed. Financial mode is bounded to 6 orders, 8
vehicles, 3,000 route upper bound, and 5,000 work units. HTTP 409 is proven
infeasibility; HTTP 503 is budget exhaustion with no plan. Currency syntax is
checked but not registry-verified; rates and penalty units are not calibrated
or claimed to be financially realistic. VQC also accepts optional
`financial_terms`; it reports an assumed one-minute expected lateness and
unverified parameter/rate suitability. If financial inference fails, there is
no proxy-only fallback.

Coordinator HTTP proof: `.venv-sdk` Uvicorn at `127.0.0.1:18377` returned a
feasible financial plan for explicit USD rate 2 and total 20 with independent
validation true. QpiAI-local VQC with the same terms returned
`financial_validation.valid: true`, forecast financial total 20, and the
requested backend. The VQC expected-exposure estimate was not captured. The
temporary server was stopped. USD/rates were caller-supplied illustrative
terms, not calibrated financial inputs.

`POST /api/v1/recovery/plan` independently validates inputs and result, then
stores an append-only SQLite audit keyed to original scenario/plan scope.
Latest success survives a failed attempt for that same scope and persists
across service restart. Recovery is a depot/time-zero replay with route-candidate
cap 2,000 and search-node cap 10,000. It does not consume live vehicle state.

`POST /api/v1/risk/evaluate` accepts a fixed validated plan and 1–100 explicit
travel-matrix scenarios, optionally `cvar_alpha`. Each replay is independently
validated. The localhost proof at port 18376 used two repeated matrices with
distinct IDs: both validations true, expected proxy objective 10, late-stop
frequency 0, and CVaR at 0.75 equal to 10. This is a sample estimate, not
calibrated operational risk.

## Limits and safe interpretation

- Domain exact solver: at most 8 orders, 16 vehicles, 10,000 capacity-feasible route candidates, and 100,000 deterministic exact-cover work units (recursive nodes plus route-option visits). Exhaustion raises `solver_budget_exceeded` and publishes no incumbent. Async jobs report this as `failed`, distinct from proven `infeasible`. Route generation prunes over-capacity prefixes while retaining stable permutation order. Delivery close times are soft SLA deadlines by default; optional `hard_window=true` rejects service after a close time.
- Recovery independently has limits of 2,000 generated routes and 10,000 recursive nodes.
- QAOA API: at most 12 route variables, 512 shots and angle grid 10. QpiAI adapter: at most 8 variables because only that range has been exhaustively mapped and measured.
- Local compute gate admits two synchronous compute calls. SQLite jobs default to two workers and 30-second timeout. Cancellation/timeout suppresses publication; Python cannot stop an already-running thread, which retains its worker slot until completion.
- Default `local` mode has no authentication. Optional token-role mode passed a coordinator-observed localhost sentinel smoke: missing/invalid tokens returned 401, dispatcher plan succeeded, researcher financial access and admin unknown-route access returned 403. The temporary server was stopped; this is not production security evidence. Keep the prototype bound to `127.0.0.1`; TLS, token rotation/expiry, identity attribution, rate limiting, secret-manager integration, and production review remain open. See [`API-RUNTIME.md`](API-RUNTIME.md). One worker process per SQLite database.
- Latest direct stress run: 0.779 seconds, maximum two solver threads, two simultaneous jobs, three capacity rejections, one cancellation, one timeout, one restart recovery, one invalid request and one sentinel-redaction case. Latest HTTP stress: 14 requests, four concurrent validations, two jobs and successful poll in 0.243 seconds. The cancellation request found the job already succeeded; in-flight HTTP cancellation is **NOT RUN**. Memory was not measured. Separate actual localhost checks validated QPiAI inference/recovery audit restart at port 18375 and risk scoring at 18376; both temporary services were stopped.
- Latest independently observed full suite: **153 passed, 1 skipped, 18 subtests in 24.79 s** (after hard-window risk replay fix); earlier counts are superseded. SDK-specific suites are separately recorded in `TERRA-CHECKPOINT.md`.
- Job metadata includes ID/status, canonical scenario/result or stable error, submitted/start/completion/deadline timestamps, solver/execution labels, request hash, settings, and Python/platform/FastAPI versions. The deterministic classical solver seed is marked not applicable. This is not a full environment lock or end-to-end reproducibility guarantee. One local Python QAOA job completed and was polled by ID; QAOA-specific restart recovery and durable QpiAI QAOA execution are unverified. VQC inference remains synchronous.
- No API key was used in local SDK or localhost checks. Do not add credentials to client requests, test output, or artifacts. Authenticated Opt/QpiAI Cloud, cloud simulator and QPU execution remain **NOT RUN**.
- Financial tests cover explicit units/rates, alternative route choices, independent validation, malformed data, infeasibility, and work-budget exhaustion. The localhost USD sample is caller-defined and uncalibrated; it does not establish cost savings or financial value. Risk estimates remain uncalibrated scenario statistics. Optional hard-window behavior is covered in core domain/API/validator paths; VQC now accepts the flag and has 14 focused inference tests passing in both environments plus localhost `qpiai_local` feasible (200/valid) and impossible (409) checks. Risk replay reports empirical hard-window violation frequency and costs while explicitly marking violating scenarios infeasible and preserving their failed validations; performance evidence for hard-window cases is absent. Live vehicle state, production authentication/authorization, durable QpiAI QAOA, and a production-grade or compute-matched comparator remain **NOT RUN/UNVERIFIED**. Frozen experiments have not been rerun after the domain change and their fixtures omit `hard_window`. The zero-penalty label test passes; the original six-case prospective artifact predates the fix and stays frozen. V2 separately evaluates penalty-independent lateness labels. See [`TERRA-V1-REPORT.md`](TERRA-V1-REPORT.md).

For gate-by-gate status, measured outputs, refinement history, and open work, see [`TERRA-V1-REPORT.md`](TERRA-V1-REPORT.md).
