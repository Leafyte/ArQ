# DebugLeaf

DebugLeaf is preparing a research prototype for Quant-A-Maze 3.0, QML-01: Delay-Risk-Aware Fleet Dispatch for Contract Logistics. The active statement and rubric are extracted from the official QpiAI problem-statement source under `sources/`. The older team-submitted challenge PDF is historical context only.

## Current stage

Bootstrap and architecture-readiness planning only. B (risk-conditioned warm-start QAOA) plus C (decision-focused VQC) is a research hypothesis. Feasibility, novelty, QpiAI API support, and useful decision impact remain unproven. No core algorithm, training, frontend, or benchmark result is claimed.

## Start here

1. Read [AGENTS.md](AGENTS.md) for scope, evidence, preservation, and handoff rules.
2. Read [the source evidence audit](bootstrap/SDK-EVIDENCE.md) for verified SDK facts, package/version cautions, and unknown capabilities.
3. Read [architecture readiness](bootstrap/ARCHITECTURE-READINESS.md) for provisional gates and sequencing.
4. Use [the documentation map](docs/DOCUMENTATION-MAP.md) to review planned docs before authoring full PRD/TRD/design documents.
5. Use the two Antigravity prompts under `prompts/` only in their named environments. The Legion prompt is ready for the remote machine; it has not been run from this workspace.

## Project-local agent skills

Exactly 25 project-local skill packages are installed under `.agents/skills/`: K-Dense (14), Anthropic (1), wshobson (9), and petergyang (1). The [skill review](bootstrap/SKILLS-REVIEW.md) and [machine-readable manifest](bootstrap/SKILLS-MANIFEST.json) record provenance and review findings; the manifest covers 258 files with no structural errors and marks the static review complete with limits. The [independent content review](bootstrap/SKILLS-INDEPENDENT-REVIEW.md) separately covers its stated subset of eight skills. Native skill activation in the current runtime has not been tested. Current SDK/runtime dependencies have not been installed.

## Verified challenge output

The official problem requires an operationally feasible fleet dispatch plan with vehicle assignments, ordered stops, ETAs and SLA-risk estimates, total financial exposure, and evaluation covering SLA violations, coverage, cost, calibration, solver quality, and runtime. The QpiAI technology rubric reserves 20 points for meaningful integration, QCloud correctness, Opt correctness, and recorded reproducible execution. See `sources/problem-extract.txt` for page-mapped extraction and the source PDF.

## Team order

M1 and M2 establish the problem/data/decision contracts and leakage-safe learning baselines first. M3 and M4 join using frozen interfaces and fixtures, independent of completed model training. This sequencing is proposed and awaits owner review.
