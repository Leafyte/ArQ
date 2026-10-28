# Antigravity Handoff — Development Checkout Research and Docs

**Status:** ready to hand off; not run by the parent. This prompt covers adversarial SDK/prior-art research and proposed documents only. Do not implement core algorithm or runtime code.

## Environment placeholders

- Development checkout: `C:\ommd3076\Proejcts\nmit-hackathon`.
- Remote execution machine: `<LEGION_HOST_OR_LABEL>` (do not assume this machine is the Legion).
- Project: DebugLeaf, Quant-A-Maze 3.0 QML-01, Delay-Risk-Aware Fleet Dispatch for Contract Logistics.

## Authority and scope

Codex coordinates and acts as architecture brain/reviewer; Luna handles Codex execution; the project owner retains final product and stage decisions. You are the Antigravity research and documentation execution agent for this handoff. Read `C:\ommd3076\Proejcts\nmit-hackathon\AGENTS.md`, `README.md`, `bootstrap/SDK-EVIDENCE.md`, `bootstrap/ARCHITECTURE-READINESS.md`, `docs/DOCUMENTATION-MAP.md`, and the official source extraction before doing work. Treat `_archive/` as preserved historical source. The official active problem is the QML-01 source under `sources/`; the older team-submitted EV-charging/QML-02 PDF is not authoritative for current requirements.

The proposed B risk-conditioned warm-start QAOA plus C decision-focused VQC is not finalized, proven novel, or proven feasible. Your task is to challenge it, not endorse it. No code/runtime, no training, frontend, dependency install, benchmarks, cloud/QPU calls, credentials, or deployment.

## Work items

1. Confirm SDK source commits and current published package metadata from official sources. Inspect release-specific Quantum, QCloud, Opt, and OR-Tools documentation/source. Trace exact QAOA availability, initialization/warm-start inputs, QUBO conversion, model constraints, local/remote solver modes, backend choices, differentiation/autodiff, shot/noise settings, and Python/platform requirements. Separate documented, source-inspected, and executed facts. Do not infer API absence from a failed docs fetch.
2. Build a compact primary/peer-reviewed prior-art table for: warm-start QAOA for constrained binary optimization; QAOA for routing/vehicle-routing or route selection; VQC classification and calibration; decision-focused learning with discrete combinatorial optimization; risk-aware dispatch and expected/tail SLA penalty objectives. Include direct DOI/arXiv/publisher links, exact claim supported, competing approaches, and why each item challenges novelty. Avoid broad scraping and blog-led claims.
3. Adversarially assess failure modes: risk score versus ETA distribution, class imbalance/calibration, censored SLA observations, correlated delay/tail risk, objective scaling, QUBO penalty dominance, route candidate completeness, relaxation-to-state mapping, circuit trainability, discrete downstream gradient mismatch, and simulator/QPU resource burden.
4. Review planned structure and cross-references only for `PRD.md`, `TRD.md`, `architecture.md`, `research-hypothesis.md`, `experiment-protocol.md`, and `AGENTS.md`. Do not write full speculative docs or freeze mathematical decisions. Propose no more than the smallest useful set of sections, with dependencies and open decisions.
5. Return 3–5 falsifiable research hypotheses and explicit kill criteria. A hypothesis needs a comparator, outcome measure, and possible disconfirmation. Include exact exhaustive tiny cases, MILP/OR-Tools, classical heuristics, ordinary QAOA, standard VQC, and strong classical ML as comparison families where applicable.

## Evidence and output locations

Write/update only these files in the development checkout:

- `C:\ommd3076\Proejcts\nmit-hackathon\bootstrap\SDK-EVIDENCE.md` — exact source/release evidence and capability unknowns.
- `C:\ommd3076\Proejcts\nmit-hackathon\bootstrap\ARCHITECTURE-READINESS.md` — proposals/open decisions only, no frozen architecture.
- `C:\ommd3076\Proejcts\nmit-hackathon\docs\DOCUMENTATION-MAP.md` — structure/dependency recommendations only.
- Add `C:\ommd3076\Proejcts\nmit-hackathon\bootstrap\PRIOR-ART-REVIEW.md` for the bounded prior-art table and falsifiable hypotheses.

For every claim, label **VERIFIED**, **PROPOSED**, **BLOCKED**, or **NOT TESTED**. Record checked date/timezone, repository commit or exact package release, primary URL and relevant section. Cite sources close to claims. If network/source is unavailable, record the blocker and leave claim unknown. Never fabricate a test or summarize unsupported novelty.

## Acceptance and stop

- Deliver a concise report of exact inspected sources, claims, gaps, and recommended next decisions.
- Do not modify implementation files, add dependencies, run code/tests, train, benchmark, or call remote QpiAI services.
- Do not ask for renewed permission for these already-authorized reversible research/documentation tasks. Stop before any core implementation or cloud/QPU work; Codex reviews, and the project owner sets the next stage.
