# Antigravity Handoff — Legion Environment Bootstrap

**Status:** prepared for the remote Legion, not executed from the development checkout. The parent cannot claim this machine was inspected or prepared.

## Target placeholders

- Legion login/device: `<LEGION_HOST_OR_LABEL>`
- Project checkout directory: `<LEGION_REPO_ROOT>`
- Development source: this checkout is currently unversioned. Use a Git commit SHA if a repository is established before transfer; otherwise use a hashed transfer manifest of the source archive/directory. Do not assume a remote URL exists.
- Primary goal: prepare a small, isolated, reproducible environment only after hardware and OS inventory.

Replace placeholders with known values. Do not request or transmit passwords, API tokens, SSH private keys, or other credentials in prompts, outputs, or logs.

## Authority and scope

Codex coordinates and acts as architecture brain/reviewer; Luna handles Codex execution; the project owner retains final product and stage decisions. The project owner authorizes the necessary small reversible environment setup for this bootstrap stage. Read the project `AGENTS.md`, `README.md`, `bootstrap/SDK-EVIDENCE.md`, and `bootstrap/ARCHITECTURE-READINESS.md` from the specified commit or hashed transfer first. You are the setup execution agent for this handoff.

This task permits inventory, project-local isolated environment creation, and installing only small dependencies needed for SDK import/local capability smoke checks after compatibility is verified. Do not wait for fresh permission for these bounded reversible steps. This authorization excludes algorithm implementation, model training, benchmarks, cloud/QPU calls, secrets, global/system changes, and expensive optional runtimes.

## Stage A — inventory before install

Record, without changing the system:

- OS/version, machine architecture, CPU model/core count, RAM, available disk and filesystem.
- GPU model, VRAM, driver and CUDA runtime only if visible through read-only installed tools; do not install/change drivers.
- Existing Python interpreters and versions, package managers, environment tooling, and shell.
- Network reachability needed for official package sources, with no credentials or private endpoint details in the report.
- Checkout path, revision, and whether source tree has uncommitted user changes. Preserve them.

If hardware inventory, OS, Python support, or disk prevents safe compatible setup, stop before installation and report the blocker. Do not assume a GPU or select CUDA dependencies before this inventory.

## Stage B — isolated setup

1. Choose a project-local `.venv` or an explicitly project-scoped environment at `<LEGION_REPO_ROOT>` using a Python version supported by all selected packages.
2. Resolve the compatible versions of `qpiai-opt`, its QCloud/Quantum integration extra only if the documented integration requires it, `qpiai-quantum`, and `ortools`. Check official package metadata and exact OS/Python wheels. Investigate the known source/release skew: Opt `0.0.9` metadata while its README extra currently pins Quantum `0.3.0`; standalone Quantum release metadata observed in the source handoff is `0.7.5`. Do not silently override the integration pin; only use a compatible documented pairing and record rationale.
3. Keep dependency set minimal. Do not add PyTorch, TensorFlow, JAX, large GPU frameworks, notebook stacks, or optional quantum backends unless a documented import/smoke strictly needs them and inventory shows a suitable environment. Never install globally.
4. Capture an exact lock or freeze file at `<LEGION_REPO_ROOT>/bootstrap/requirements-legion.lock` plus package versions, hashes when available, interpreter, platform, and install command. Do not record secrets.

## Stage C — tiny smoke only

Run only minimal deterministic checks that need no credentials: interpreter/package imports; QpiAI Quantum circuit construction; any local simulator circuit check only if documented to work without authentication; Opt model/QUBO construction and credential-free inspection/feasibility APIs; OR-Tools import and a tiny local toy solve. Do not invoke authenticated Opt solving, QCloud, QPU, training, a project routing algorithm, or a benchmark. Do not mistake constructing a model for solving it or a local toy for project validation.

For each check, report exact command, dependency versions, exit status, relevant non-sensitive output, and whether it proves import, model construction, local simulator execution, or toy optimization. If authentication unexpectedly becomes necessary, stop that check without reading or requesting credentials.

## Safety and change limits

- No administrator privileges, global package installation, global PATH changes, CUDA/driver changes, Windows policy changes, registry edits, or unrelated software installs.
- Do not delete/overwrite existing source, environments, data, checkpoints, or user files. If a target path already contains work, choose a new isolated path and record it.
- Keep logs free of tokens, usernames not needed for evidence, and environment dumps that could expose secrets.
- Preserve project source revision and working tree. Do not commit, push, or deploy.

## Deliverables

Write only the following project-local environment/evidence files, if source checkout is writable and contains no conflicting user edits:

- `<LEGION_REPO_ROOT>/bootstrap/requirements-legion.lock`
- `<LEGION_REPO_ROOT>/bootstrap/LEGION-INVENTORY.md`
- `<LEGION_REPO_ROOT>/bootstrap/LEGION-ENVIRONMENT.md`
- `<LEGION_REPO_ROOT>/.venv/` as the isolated environment if this path is available and empty; otherwise choose and report a fresh project-local environment path.

Report inventory, source commit SHA or transfer-manifest hash, environment location, interpreter, exact package versions/hashes and compatibility source, install commands, each smoke classification/result, deviations, and unresolved unknowns. If source checkout is not writable, return these exact reports without modifying unrelated locations. End by stating clearly that no algorithm, training, benchmark, QCloud/QPU execution, or project-level result was performed.
