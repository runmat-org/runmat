# FEA-RELEASE-EVIDENCE-001: Independent formulation evidence and truthful claims

0 PASS, 0 FAIL, 24 OPEN, 0 TAGGED. Generated; do not edit.

## verification_with_each_formulation

Evidence kind: PublicationIntegrity. Criterion: Verification is implemented with each family, not postponed until all solvers exist.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## independent_governed_formulation_corpus

Evidence kind: PublicationIntegrity. Criterion: Unit and dimensional consistency tests. Element/cell-level operator tests. Rigid-body, null-space, symmetry, definiteness, or conservation invariants as applicable. Analytic or manufactured solutions. Patch tests where applicable. h-, p-, frequency-, nonlinear-step-, or time-step-convergence studies with expected behavior. Boundary/load/source resultant checks. Global and local balance checks. Inv...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## truthful_per_formulation_evidence

Evidence kind: PublicationIntegrity. Criterion: Each formulation's evidence is tracked independently: `verified`: governing discretization and internal numerical evidence pass; `validated`: independent comparison passes within the documented use class; `release-gated`: verified, validated, documented, and supported across its claimed execution shapes. Do not claim evidence for an entire broad family when only one formulation has it.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

