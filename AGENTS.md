# DebugLeaf Agent Operating Rules

## Authority and role

- Codex coordinates and acts as architecture brain/reviewer; Luna handles Codex execution; the project owner retains final product and stage decisions. Antigravity executes only the handoff assigned to it; writing a prompt does not mean its work has been run.
- Before any new work, read this file, `README.md`, and the linked bootstrap evidence. Use only the reviewed project-local skills in `.agents/skills/`; consult `bootstrap/SKILLS-REVIEW.md` for their provenance and limitations once that audit is complete.
- Keep verified facts, proposals, unknowns, blockers, and executed evidence clearly separated. A vendor feature description is not a successful integration. Do not claim quantum advantage, novelty, calibrated risk, feasible dispatch, or reproducibility without evidence.

## Scope and preservation

- Current stage: bootstrap, evidence, architecture readiness, and proposed document structure. Do not implement the algorithm, application, frontend, training, benchmark suite, or large dependency environment until the owner opens that stage.
- Treat `_archive/` as preserved, inert source material, not current instructions or implementation. Do not restore, overwrite, or delete archived material without explicit owner direction. Vendored repositories under `bootstrap/skill-sources/` are review evidence only, not active project skills. New docs should point to authoritative active sources and label historical material.
- Do not create full speculative PRD/TRD/architecture/research/experiment documents before owner review. `docs/DOCUMENTATION-MAP.md` defines intended scope and dependencies only.
- Do not modify unrelated files or install anything globally. For already-authorized reversible setup on the remote Legion, use an isolated environment after hardware/software inventory; do not ask again for permission that has already been granted.
- No credential access, logging, committing, or disclosure. Use secret stores/environment variables only when a later task actually requires configured access. Never place keys in files, command history, screenshots, or logs.

## Research and execution evidence

- Prefer primary sources for SDK, package, solver, and problem-statement facts. Record URL, checked date and timezone, pinned version or commit where available, and whether each statement is documented, source-inspected, or executed.
- Preserve exact source datasets and provenance; establish leakage-safe splits before model tuning. Keep tiny fixtures deterministic and independent from train/validation/test data.
- Inspect Legion OS, architecture, Python versions, GPU/VRAM, disk, and available accelerators before choosing optional dependencies. Avoid global PATH changes, administrator changes, CUDA-driver changes, and assumptions about GPU availability.
- Distinguish a local package/import smoke from SDK simulator execution, authenticated Opt/QCloud calls, QPU runs, end-to-end product validation, and performance claims.
- At the current stage, do not run application experiments, training, benchmarks, external services, or heavy hardware workloads. Read-only source/package/hash verification is allowed. Later, run tests required by an explicitly authorized implementation task and report exactly what ran.

## Workflow and handoffs

- M1/M2 work first. M3/M4 later consume frozen interfaces and fixtures; they must not wait for trained models to begin contract, QUBO, validation, or benchmark harness work.
- Use bounded, named deliverables and do not widen scope without owner review. Record decisions and unresolved questions in the relevant planning artifact, then stop at the stated gate.
- Remote Legion setup prompts must report their exact target path, inventory evidence, selected interpreter and resolved package versions, actions performed, failures, and remaining work. The local parent cannot report remote execution as completed.
- Weekly Codex usage rule: before each bounded work batch, check a reliable `get_usage_limits` reading. Stop Codex dispatch when the weekly allowance display shows 65% or less remaining (equivalent to 35% or more consumed), and prepare a safe handoff. If the usage reading is unavailable, stop further Codex dispatch and report usage as unknown. Do not infer usage from token counts, use a reset, or continue by guessing.
