# DebugLeaf Agent Operating Rules

## Authority and stage

- Codex coordinates architecture, reviews evidence, and prepares bounded handoffs. Luna executes implementation and tests; Antigravity executes only assigned work. The project owner retains product decisions.
- The owner has opened the **V1 implementation stage** for the bounded 5-hour Terra build. Build a full backend centered on a decision-focused VQC, with classical warm starts and warm-start QAOA as supporting paths. Keep the delivery decision-focused and runnable locally.
- Before work, read this file, `README.md`, the relevant acceptance targets in `docs/V1-ACCEPTANCE.md`, and source/evidence documents linked from them. Use only reviewed project-local skills in `.agents/skills/`; `bootstrap/SKILLS-REVIEW.md` records provenance and limitations.
- The attached legacy build specification is preserved verbatim at `sources/q-vantis-build-spec.md` as reference. It does not expand this V1 scope by itself. `_archive/` remains preserved, inert source material; do not restore, overwrite, or delete it without owner direction. Vendored repositories under `bootstrap/skill-sources/` are review evidence, not active skills.

## V1 scope and execution

- Authorized now: implement the backend and decision-focused quantum learning path, bounded local CPU experiments/training on tiny instances, tests, stress checks, localhost API integration, and scoped isolated runtime/package installs. Keep the runtime isolated; do not install globally or modify global PATH, administrator settings, or CUDA drivers.
- Default to bounded local CPU and genuine small quantum circuits, typically 4–8 qubits and tiny deterministic instances. Do not assume GPU, QPU, Legion availability, or a working accelerator. Use Legion only if reachable and after recording OS, architecture, Python, memory/disk, GPU/VRAM, and accelerator inventory.
- No paid QPU jobs or credit spending without specific owner authorization. Credential-free CPU simulation and local work are authorized. Configured SDK authentication may use standard environment variables when required; never read or print secret values, inspect browser-saved passwords, or put secrets in files, screenshots, command history, logs, or API responses.
- Keep changes scoped to assigned deliverables and avoid disturbing other agents' files. Do not commit unless explicitly assigned. A prompt or delegation is not evidence that the delegated work ran.
- Deliver at least three substantive refinement cycles. Each cycle must state a failure hypothesis, make a material edit, rerun the relevant check, and record evidence; renaming milestones or cosmetic-only passes do not count.
- Explain the implemented mechanism in the new Terra output with code pointers and tiny numerical examples. Do not create a long speculative PRD or full algorithm source in this handoff; implementation belongs to the new Terra build.

## Evidence and scientific claims

- Separate verified facts, assumptions/proposals, unknowns/blockers, acceptance targets, and executed results. Vendor feature descriptions, package imports, mocked calls, and prompts are not successful integrations.
- Prefer primary sources for SDK, package, solver, and challenge facts. Record source URL, checked date/timezone, pinned version/commit where available, and whether each point is documented, source-inspected, or executed.
- Preserve exact dataset provenance. Freeze leakage-safe train/validation/test splits before model tuning; keep deterministic tiny fixtures independent of these splits. Never tune on held-out evaluation outcomes.
- Compare predictive-only, classical, and decision-focused approaches on the same held-out splits and common-random-number (CRN) scenarios. Do not select scenarios or labels post hoc to create a favorable quantum result. Keep matched baselines and blind-test settings frozen before evaluation.
- Never claim quantum advantage, novelty, calibrated risk, feasible dispatch, or reproducibility without direct evidence. Distinguish local simulator, mocked integration, authenticated API, and QPU execution. Quantum measurement counts alone are not a solved business decision.
- Every plan reported feasible must pass an independent validator that recomputes constraints and financial objective from source inputs. Preserve raw solver/circuit output, decoded decisions, repairs, and validation outcomes separately; make fallback use explicit.

## API, robustness, and reporting

- The backend owns quantum SDK calls and secrets; clients call the backend API only. Validate input at boundaries; return stable, non-secret errors for invalid data, unsupported capability, job failure, timeout, infeasibility, and validation failure.
- Persist enough job/run metadata to reproduce local work: IDs, seeds, split/data hashes, settings, dependency versions, backend/execution type, timing, status, and available circuit/job metadata. Mark unavailable fields as unavailable; do not fabricate them.
- Stress checks should cover invalid data, concurrent jobs, timeout, cancellation, restart/recovery, and secret redaction under bounded loads and recorded resource budgets.

## Usage and timebox

- Before each bounded Codex work batch or dispatch, read reliable `get_usage_limits`. Stop dispatch and new jobs when the weekly allowance has **50% or less remaining** (50% or more consumed), interrupt active workers, and prepare a safe handoff. If the reading is unavailable, fail closed: stop new dispatch and report usage unknown. Keep reserve and check frequently. Do not infer usage from tokens, reset the limit, or claim global enforcement beyond the actual reading.
- The Terra build window begins when the new Terra agent actually starts, not during this preflight. Persist that start time and deadline in the Terra output. The deadline is start + 5 wall-clock hours; begin wrap-up 15 minutes before it and stop at the deadline without extending it. Do not wait or sleep just to fill the window, and do not treat elapsed time as proof of completion.
- At the end, report exact paths, checks/experiments actually run, outcomes, failures, unresolved items, resource limits, and delegated work status. Do not claim remote execution from local evidence.
