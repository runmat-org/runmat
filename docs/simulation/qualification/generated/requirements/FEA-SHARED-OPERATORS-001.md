# FEA-SHARED-OPERATORS-001: Canonical sparse operators and compatible prepared solves

0 PASS, 3 PARTIAL, 0 FAIL, 45 OPEN, 0 TAGGED. Generated; do not edit.

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
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::algebra::tests::sparse_reduction_matches_dense_lifting_and_preserves_operator_properties` (crates/runmat-analysis/fea/src/constraints/algebra.rs:661) - Sparse reduction matches dense lifting and preserves operator properties. |
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
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::adaptation::structural_recovery::qualification_tests::elastic_rigidity_static_path_agrees_with_the_proving_factorization` (crates/runmat-analysis/fea/src/adaptation/structural_recovery/qualification_tests.rs:42) - The elastic rigidity path (no global factor, AMG-PCG, true-residual acceptance) matches the proving factorization's displacement to 1e-6 on a structured cantile...<br>`runmat-analysis-fea[lib]::adaptation::structural_recovery::qualification_tests::effective_ceiling_selects_the_static_solve_path` (crates/runmat-analysis/fea/src/adaptation/structural_recovery/qualification_tests.rs:48) - The planner selects the elastic rigidity path exactly when the effective ceiling, min(authored, machine), lies below the proving factorization's plan; recorded ...<br>`runmat-analysis-fea[lib]::adaptation::structural_recovery::qualification_tests::static_solve_fallback_is_admitted_against_the_effective_ceiling` (crates/runmat-analysis/fea/src/adaptation/structural_recovery/qualification_tests.rs:54) - The elastic rigidity fallback is admitted only when its resident memory and scratch fit the effective ceiling; below both plans the planner fails with a resourc...<br>`runmat-analysis-fea[lib]::algebra::elastic_rigidity::tests::face_glued_and_anchored_tetrahedra_are_proved` (crates/runmat-analysis/fea/src/algebra/elastic_rigidity/tests.rs:96) - Face-glued elastic tetrahedra with a three-node anchor are proved positive definite without a global factor.<br>`runmat-analysis-fea[lib]::algebra::elastic_rigidity::tests::hinge_glued_by_two_nodes_is_refused_as_a_mechanism` (crates/runmat-analysis/fea/src/algebra/elastic_rigidity/tests.rs:112) - A hinge (two tetrahedra sharing only an edge) is a mechanism and is refused with a typed reason naming the unanchored component.<br>`runmat-analysis-fea[lib]::algebra::elastic_rigidity::tests::collinear_or_missing_anchor_is_refused` (crates/runmat-analysis/fea/src/algebra/elastic_rigidity/tests.rs:138) - An anchor that leaves a rotation free (two fixed nodes) or no anchor at all is refused; the refusal counts the constrained DOFs.<br>`runmat-analysis-fea[lib]::algebra::elastic_rigidity::tests::element_without_energy_margin_or_rigid_kernel_is_refused` (crates/runmat-analysis/fea/src/algebra/elastic_rigidity/tests.rs:161) - An element block that is indefinite on the rigid-motion complement, or that does not annihilate rigid motions, is refused naming the element.<br>`runmat-analysis-fea[lib]::solve::krylov::envelope::tests::invalid_policy_is_rejected_by_solver_before_any_operator_or_reduction` (crates/runmat-analysis/fea/src/solve/krylov/envelope.rs:182) - An invalid policy is rejected before any operator or reduction.<br>`runmat-analysis-fea[lib]::solve::linear::tests::noncoercive_operator_is_rejected_before_preconditioner_setup` (crates/runmat-analysis/fea/src/solve/linear/tests.rs:364) - A noncoercive operator is rejected before preconditioner setup. |
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
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::general::tests::independent_nonsymmetric_and_indefinite_references_preserve_general_operator` (crates/runmat-analysis/fea/src/solve/general/tests.rs:25) - Independent nonsymmetric and indefinite references preserve the general operator path; SPD and complex cases are covered by sibling tests. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

