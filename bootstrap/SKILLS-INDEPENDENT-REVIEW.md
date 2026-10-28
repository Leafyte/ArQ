# Independent Skill Content Review

Date: 2026-10-28. This is a read-only audit of the disjoint skill subset installed under `.agents/skills/`. It does not replace the main installation manifest or the vendor SDK evidence.

## Scope and method

Reviewed the complete `SKILL.md` files, local relative Markdown references, per-skill licenses, and bundled code for these eight skills:

| Skill | Pinned source | Installed files | Assessment |
|---|---|---:|---|
| `qiskit` | [K-Dense scientific-agent-skills, `5f2bc034`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/qiskit) | 16 | Useful Qiskit/IBM reference and local simulator examples. IBM Runtime content is not QpiAI QCloud evidence. |
| `pennylane` | [K-Dense scientific-agent-skills, `5f2bc034`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/pennylane) | 9 | Directly relevant to VQC, differentiation and QAOA research; provider workflows remain separate. |
| `scikit-learn` | [K-Dense scientific-agent-skills, `5f2bc034`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scikit-learn) | 12 | Useful classical baselines, preprocessing and evaluation; adapt its independent-row examples for route/time grouping. |
| `statsmodels` | [K-Dense scientific-agent-skills, `5f2bc034`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/statsmodels) | 11 | Useful for statistical baselines, uncertainty and diagnostics; results still depend on study design and assumptions. |
| `sympy` | [K-Dense scientific-agent-skills, `5f2bc034`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/sympy) | 9 | Useful for exact symbolic derivations and small QUBO checks; never parse or compile untrusted expressions. |
| `networkx` | [K-Dense scientific-agent-skills, `5f2bc034`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/networkx) | 8 | Useful for graph construction and shortest-path support, not a fleet VRPTW optimizer. |
| `architecture-patterns` | [wshobson/agents, `0874a27`](https://github.com/wshobson/agents/tree/0874a27b17a22e2b7c2e2bcd725768e0803cad06/plugins/backend-development/skills/architecture-patterns) | 4 | Good boundary and dependency guidance; keep the initial build small and avoid adopting service/microservice examples by default. |
| `stride-analysis-patterns` | [wshobson/agents, `0874a27`](https://github.com/wshobson/agents/tree/0874a27b17a22e2b7c2e2bcd725768e0803cad06/plugins/security-scanning/skills/stride-analysis-patterns) | 3 | Useful threat checklist, but identifying a threat is not proof that it has been mitigated. |

The staged upstream checkouts are detached at the stated commits. Every source file in each of these selected skill directories matches its installed counterpart by SHA-256. The installer adds one license file per installed skill. All 56 unique relative Markdown references resolve within the installed skills; no missing local reference was found. After the main manifest landed, its rows for these eight skills were independently checked against every installed file path, byte count, and SHA-256; all 72 file rows and license paths matched, and each row lists zero missing local references. The main manifest still labels these rows `STATIC_REVIEW_PENDING`; this document records the completed independent static review without editing that manifest. No skill code, demo, dependency installer, runtime probe, or networked action was executed.

## Static behavior and safety findings

- **Qiskit:** `scripts/check_environment.py` checks package metadata/imports and builds a trivial local circuit. It reports the Python executable and platform details to stdout; it does not access credentials or the network. `scripts/run_local_primitives.py` runs a local primitive example, validates the seed/shots, and caps shots at 1,000,000; it does not submit a job. `scripts/inspect_runtime.py` constructs `QiskitRuntimeService`, selects/queries IBM backend metadata, and therefore uses the user's saved IBM credentials and makes network reads. Its source handles errors without printing credential data and states it submits zero jobs. Treat that command as an explicit IBM account/network action, not a QpiAI run.
- **PennyLane:** No bundled scripts. Device/provider references show optional remote-provider flows and IBM credential examples. Local simulator guidance is separate. Remote provider execution requires a provider/account and network; do not infer that PennyLane's provider path integrates QpiAI.
- **scikit-learn:** `classification_pipeline.py` uses a built-in breast-cancer dataset in its demo mode and prints metrics/class counts; its callable accepts caller data. Its own docstring warns the random row split is not suitable for grouped or temporal observations. `clustering_analysis.py` uses caller-provided arrays and built-in demo data, then writes fixed filenames `clustering_optimization.png` and `clustering_results.png` to the process current directory. Those names can overwrite existing images; run it only from a dedicated scratch directory or change output paths before use. Neither script has an outbound network call.
- **Statsmodels:** The skill consists of documentation/references, no bundled executable scripts. Examples operate on supplied in-memory data. The guide itself calls out missing-value behavior and recommends grouped or chronological validation where the sampling design requires it.
- **SymPy:** The skill explicitly warns that `parse_expr`, string `sympify`, and `lambdify` can execute Python code, and says regex filtering/`local_dict` are not a sandbox. Its code-generation reference repeats the trust warning and flags pickle loading. Keep expressions constructed from trusted symbols/allowlisted grammar and set resource limits for untrusted input. The directory has both `references/core_capabilities.md` and `references/core-capabilities.md`; the skill links to both, so this is redundant naming rather than a broken link.
- **NetworkX:** The graph I/O reference demonstrates Python pickle and explicitly warns to unpickle trusted sources only because loading may execute code. Prefer non-executable graph formats for user-supplied data. File examples use caller-chosen paths; the skill does not impose a project-root sandbox.
- **Architecture patterns:** Documentation only. The HTTP client example illustrates an injected `httpx` port; it is not a bundled network call. Use the dependency-boundary ideas without introducing authentication, databases or microservices that the brief does not require.
- **STRIDE:** Documentation-only taxonomy and checklist. It does not read secrets, inspect files, or make network requests; use it to identify threats, then verify concrete mitigations separately.

The K-Dense skill content advertises broader shell/write permissions in frontmatter for some entries. This review did not exercise those capabilities. Any future use of the bundled scripts should be an explicit decision, with a controlled working directory for file-producing examples and an isolated environment for package installation.

## Project fit and limits

The six scientific package skills are reasonable procedural references for classical baselines, statistical evaluation, exact algebra, graph preprocessing, and simulator-based QML exploration. Their stated package versions and test claims are upstream skill claims; this review did not independently verify APIs, install dependencies, or run examples. They should not be treated as proof that the selected packages fit the external laptop or that any proposed method works.

Qiskit and PennyLane materials concern their own frameworks and, in Qiskit's case, IBM Runtime. The challenge's mandatory QpiAI QCloud and QpiAI Opt integrations must follow the QpiAI SDK sources and tested project adapters. `networkx` can support a road graph or travel-time shortest paths, but cannot substitute for route generation, a constrained dispatch optimizer, or an independent validator. `architecture-patterns` is a guide, not a decision to build a layered application or multiple services.

Skill-level license metadata describes the target package while each installed skill directory carries its own MIT license for the skill text. Review target-package licensing separately before selecting dependencies; the skill bundle does not change dependency licenses.

## Recommendation

Keep these eight project-local skills available. Use them as optional, version-scoped references and select only the relevant skill for a specific task. Do not install package dependencies merely because a skill is present. Before relying on any example or version claim, check official package/QpiAI documentation for the chosen environment and validate against the project's frozen interfaces. The Qiskit IBM Runtime inspection helper should remain unused unless an IBM backend check is separately needed and authorized.

