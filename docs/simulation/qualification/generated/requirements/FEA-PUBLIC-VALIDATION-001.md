# FEA-PUBLIC-VALIDATION-001: Validation published, docs true, no maturity surface

0 PASS, 0 PARTIAL, 0 FAIL, 7 OPEN, 0 TAGGED. Generated; do not edit.

## published_benchmark_tables

Evidence kind: IndependentReference. Criterion: For every physics a table of named benchmark cases (analytic, manufactured and published) is generated from runs and published with the reference value, the computed value, the difference, mesh convergence and wall time, misses included; no value is typed by hand.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## open_solver_comparison

Evidence kind: IndependentReference. Criterion: Formula, RunMat and an open solver agree within stated tolerance side by side on five real parts, with the method and any disagreement published.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## status_and_docs_match_code

Evidence kind: SemanticRealization. Criterion: The simulation docs, status tables and per-physics pages are generated from or checked against the admission registry and runner gate, and every row in the docs-drift list is closed; branch features are documented in the change that adds them.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## guides_published_with_runs

Evidence kind: PublicRun. Criterion: The heat-sink and tuning-fork guides publish with their projects, archives and HTTP checks from accepted runs.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## guide_per_family

Evidence kind: PublicRun. Criterion: Each shippable physics family has a guide with a runnable example, including electrostatics, conduction, magnetostatics, harmonic Maxwell, acoustics and flow.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## no_maturity_surface

Evidence kind: RejectionInvariant. Criterion: No `formulation = experimental` switch, maturity gate in admission or kernels, 'production qualification unsupported' rejection, or the word experimental in docs, examples and file names remains; a finished capability simply runs.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

