# Q-VANTIS V1 preflight

**Checked:** 2026-10-28 (Asia/Kolkata)  
**Scope:** Read-only inventory and bounded validation of the current workspace before the authorized V1 implementation. This report does not describe implemented product behavior.

## Verified inventory

- Checkout is `main` at `cfeda5534d511e3d18f518ff8fe8dc07aaf7306e`. Existing worktree changes at inspection time are the SDK evidence/dependency notes, SDK requirements, interface requirements, and `tests/` plus `tools/` probes/inspector. No `q-vantis/`, backend application, frontend, mobile app, scenario implementation, or solver implementation appears in the tracked or untracked inventory.
- Existing tools are `tools/inspect_environment.py` and three SDK probes plus `run_all_probes.py`. The inspector reads OS/CPU/memory/disk, installed package metadata, Python interpreter candidates, and installed GPU utilities; it does not inventory Legion because it runs on this host. The probes attempt imports, small construction/local checks, and report credential *presence* as a boolean. They do not submit QCloud jobs.
- Existing `tests/test_inspection_and_probes.py` is a `unittest` suite for inspector output, probe structure, mocked SDK paths, credential-value non-disclosure, and CLI JSON. It contains no fleet/risk/route/validation/QUBO/Ising/VQC/API/UI/recovery tests. Several mocked tests assert behavior of test doubles rather than installed SDK behavior.
- The QpiAI Quantum probe constructs and simulates a two-qubit Bell circuit, not a fleet-selection circuit. The Opt probe attempts a small model/QUBO construction and can stop before solving when no key is configured. Neither is evidence of an authenticated Opt solve, QCloud submission, or QPU execution. The local bootstrap evidence documents SDK symbols/capabilities by source/wheel inspection, separately from execution.

## Validation attempted

- `python -m unittest discover -s tests -v` — **not run**: `python.exe` resolves to the WindowsApps app-execution alias and could not start (`ResourceUnavailable`).
- `py -3 --version` — **not run**: the `py.exe` WindowsApps alias likewise could not start.
- No dependencies were installed; no cloud calls or heavy workloads were run. No tests passed or failed because no Python interpreter was available.
- The project-local `python-testing-patterns` skill was read. Its AAA structure and isolation guidance informed this review; its examples do not add verification evidence.

## Top implementation gates / defects

1. **Runtime gate:** locate/provision the explicitly authorized isolated interpreter and pin dependencies before claiming any tests or app startup. This host currently exposes only non-starting WindowsApps aliases.
2. **No vertical slice exists yet:** implement the smallest deterministic scenario → routes → risk → selection → independent validation → API path first, then quantum work. Keep the current bootstrap probes distinct from product modules.
3. **QpiAI V1 claims need executed evidence:** current probes cannot meet the spec's meaningful circuit experiment, persisted job metadata, authenticated QCloud, or verified Opt solve criteria. Keep those states unavailable until each is actually exercised; a local Bell simulation is only a local simulator smoke.
4. **Credential contract mismatch:** probes read `QPIAI_QCLOUD_API_KEY`, while the attached spec lists `QPIAI_QUANTUM_API_KEY`; settle this against the SDK's real auth API and document one backend-only name. Current probes expose only configured/not-configured, but this mismatch can produce a false “missing” result.
5. **Current automated checks are narrow and may overstate readiness:** `run_all_probes.py` returns its report without failing its process for unsuccessful capability stages, and probe tests are largely structural/mocked. Make capability status explicit in docs/CLI and add real domain mathematical/validator tests as implementation arrives.

## Suggested first commands

After selecting an installed interpreter and creating the isolated environment, start with the current lightweight checks (do not install from global Python):

```powershell
<venv-python> -m unittest discover -s tests -v
<venv-python> tools/inspect_environment.py
<venv-python> tools/sdk_probes/run_all_probes.py --json
```

Then build the V1 vertical slice in the attached milestone order, beginning with a seeded tiny fixture and an independent validator. Add QUBO/Ising exhaustive equivalence and decision-focused VQC/warm-start experiments only behind clearly separated, reproducible interfaces. This is a proposed implementation sequence, not an assertion that those modules exist.

## Evidence boundary

The workspace sources and bootstrap docs establish the challenge requirements and documented/source-inspected SDK facts. The existing tools establish only their executed checks once a working interpreter is available. **No Q-VANTIS backend, VQC, decision-focused learning, warm-start QAOA, API, web/mobile client, optimizer result, benchmark, authenticated SDK run, or QPU run is implemented or evidenced by this preflight.**
