# FEA-SHARED-OPERATORS-001: Canonical sparse operators and compatible prepared solves

0 PASS, 0 FAIL, 48 OPEN, 0 TAGGED. Generated; do not edit.

## mathematics_and_execution_boundary

Evidence kind: IndependentReference. Criterion: The analysis system owns the mathematical operator; RunMat's generic compute substrate owns execution representations and placement. This boundary must be explicit before production families proliferate.

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

## declared_sparse_operator_properties

Evidence kind: IndependentReference. Criterion: Every assembled or matrix-free operator declares: scalar domain and precision; dimensions and block structure; symmetry or Hermitian status; positive-definite, semidefinite, indefinite, or unknown classification; null-space and gauge information; sparse pattern identity and numeric-value revision when assembled; constraint-plan identity; partition plan when distributed; apply, transpose-apply, and...

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

## explicit_validated_storage_adapters

Evidence kind: IndependentReference. Criterion: Three sparse representations remain deliberately distinct: MATLAB-facing sparse values, which implement language-visible value semantics; portable execution sparse values and chunked numerical artifacts, which implement transport and persistence; and the canonical analysis operator, which implements numerical semantics and efficient application. Conversions are explicit, validated adapters. `runma...

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

## backend_neutral_preparation

Evidence kind: IndependentReference. Criterion: Physics assembly returns a `DiscreteOperator`, not a CPU, GPU, or distributed solver object. A typed `LinearSolvePlan` then selects a compatible algorithm, preconditioner, tolerance policy, precision policy, and restart policy. Preparation produces one of: a host prepared operator; an accelerator-provider-owned prepared operator; or a distributed prepared operator over an explicit partition and wo...

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

## compatible_solver_and_preconditioner_selection

Evidence kind: IndependentReference. Criterion: Solver selection is based on declared algebraic properties, not a generic `linear` label: Invalid pairings are rejected before solve. CG is never used for an indefinite multiplier system, Helmholtz operator, Maxwell system, or nonsymmetric CFD operator merely because it is available. Preconditioner setup, reuse, invalidation, application, memory, and evidence are first-class parts of the solve pla...

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

## operator_algorithm_contract

Evidence kind: SemanticRealization. Criterion: Real SPD; CG with validated diagonal, incomplete-factorization, or multilevel preconditioning as supported.. Symmetric/Hermitian indefinite; MINRES or an explicitly supported direct method.. Real/complex nonsymmetric; GMRES or another governed Krylov method with restart evidence.. Saddle-point/block system; Block-aware Krylov and Schur-complement preconditioning.. Generalized eigenproblem; Lanczos...

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

