# FEA-RELEASE-EVIDENCE-001: Independent formulation evidence and truthful claims

0 PASS, 0 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## verification_with_each_formulation

Evidence kind: PublicationIntegrity. Criterion: Verification is implemented with each family, not postponed until all solvers exist.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## independent_governed_formulation_corpus

Evidence kind: PublicationIntegrity. Criterion: Unit and dimensional consistency tests. Element/cell-level operator tests. Rigid-body, null-space, symmetry, definiteness, or conservation invariants as applicable. Analytic or manufactured solutions. Patch tests where applicable. h-, p-, frequency-, nonlinear-step-, or time-step-convergence studies with expected behavior. Boundary/load/source resultant checks. Global and local balance checks. Inv...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## truthful_per_formulation_evidence

Evidence kind: PublicationIntegrity. Criterion: Each formulation's evidence is tracked independently: `verified`: governing discretization and internal numerical evidence pass; `validated`: independent comparison passes within the documented use class; `release-gated`: verified, validated, documented, and supported across its claimed execution shapes. Do not claim evidence for an entire broad family when only one formulation has it.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

