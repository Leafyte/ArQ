# Documentation Map — Review Draft

**Purpose:** agree on structure, ownership, and prerequisites before full documents are written. This map does not finalize the mathematical design or the B+C research hypothesis.

## Proposed documents and dependencies

| Document | Purpose and boundary | Depends on / owner |
|---|---|---|
| `PRD.md` | Active QML-01 problem, users, mandatory outputs, measurable acceptance criteria, exclusions and judging constraints. Cite the official source PDF; identify assumptions separately. | Official source extraction; owner approves problem framing. |
| `TRD.md` | Mathematical requirements, constraints, prediction outputs, candidate/solver boundaries, data requirements, metrics and technical constraints. Specify testable requirements, not unverified SDK methods. | Approved PRD; SDK evidence; M1 + M2 proposal. |
| `architecture.md` | Module boundaries, data flow, interface contracts, environment topology, failure paths and portable execution split. | Approved PRD/TRD; SDK and Legion inventory; owner approves design. |
| `research-hypothesis.md` | B+C research claim candidates, prior art, counterarguments, novelty risks, falsifiable hypotheses and stopping criteria. Avoid novelty claims until literature review. | Bounded prior-art review; M1/M2; explicit unknowns. |
| `experiment-protocol.md` | Leakage-safe splits, provenance, baselines, ablations, metrics, seeds, resource limits, result schema and reporting. | Frozen data provenance and interface fixtures; M2 + M4. |
| `AGENTS.md` | Repository scope and evidence rules, safety/preservation, reproducibility, agent division and gates. | Current bootstrap instructions; update only with owner-approved workflow changes. |

## Proposed cross-links

- `PRD.md` owns *what outcome is required*; `TRD.md` defines *how success is made measurable*.
- `architecture.md` links each module to the TRD requirements and frozen interface definitions.
- `research-hypothesis.md` links claims to primary/peer-reviewed citations and records competing explanations.
- `experiment-protocol.md` maps every hypothesis to baselines, splits, metrics, ablations, and artifacts.
- All documents link to `bootstrap/SDK-EVIDENCE.md` for vendor capability status and the authoritative problem source in `sources/`.

## Review gates

1. Owner confirms PRD scope and official challenge interpretation.
2. M1/M2 propose problem, data, prediction, and decision contracts; freeze split/provenance before training.
3. Owner reviews TRD/architecture and the B+C novelty/feasibility questions.
4. M3/M4 receive frozen interfaces and tiny deterministic fixtures; no dependence on trained models to begin solver/validator/reproducibility scaffolding.
5. Write complete documents only after their inputs and owners are accepted.

## First three proposed engineering tasks

1. **M1 — Problem and decision contract:** reconcile all constraints and outputs against official QML-01; specify route, time-window, contract, cost, and feasibility assumptions for owner review.
2. **M2 — Data and baseline contract:** document data provenance, temporal/group leakage controls, risk/ETA target candidates, split policy, and strong classical ML baselines; freeze the split before fitting.
3. **M1 + M3/M4 — Tiny optimization/validation plan:** on hand-approved fixtures, define route-choice QUBO encoding and independent-validator contract, enumerate tiny cases for correctness, and assess relaxation-to-state mapping/penalty-scale risk before any quantum adapter. This is a plan and interface exercise, not implementation or algorithm approval.

## Timebox proposal

For an assumed nine-hour overnight sprint only, use relative blocks rather than a wall-clock start: first 1 hour validate official source/data availability and owner scope; next 2 hours M1/M2 contract and prior-art questions; next 2 hours freeze splits and baseline definitions; next 2 hours M3/M4 fixtures and validator/QUBO review plan; final 2 hours consolidate evidence and owner review. This is a planning allocation, not a claim about the current time or available hardware.
