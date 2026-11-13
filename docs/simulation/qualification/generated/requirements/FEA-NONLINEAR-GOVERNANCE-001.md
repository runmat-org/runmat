# FEA-NONLINEAR-GOVERNANCE-001: Converged nonlinear states, supported laws and independent qualification

0 PASS, 3 PARTIAL, 0 FAIL, 47 OPEN, 0 TAGGED. Generated; do not edit.

## operator_constitutive_contact

Evidence kind: AccuracyAcceptance. Criterion: Implement nonlinear state, residual, and tangent assembly. Implement supported plastic integration and history. Implement contact search, enforcement, friction, and tangent coupling.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| nonlinear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::contact::energy::opening_releases_stored_tangential_energy_without_inventing_friction_heat` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/energy.rs:65) - A coupled solid contact solve releases stored tangential energy on opening for both orders; constitutive plasticity is covered in the operator requirement. |
| nonlinear_structural | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::contact::energy::opening_releases_stored_tangential_energy_without_inventing_friction_heat` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/energy.rs:65) - A coupled solid contact solve releases stored tangential energy on opening for both orders; constitutive plasticity is covered in the operator requirement. |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | native_process | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | durable | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## inequality_restart_converged_base

Evidence kind: AccuracyAcceptance. Criterion: Validate contact as a nonlinear inequality system alongside, but distinct from, equality constraints. Implement restart and supported transfer of nonlinear state. Publish only converged nonlinear base states that satisfy the declared transition envelope; distinguish physical state from an initial guess or incomplete checkpoint.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| nonlinear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-execution[lib]::tests::accuracy_plan_binding_tests::nonlinear_accuracy_context_binds_the_authored_accuracy_plan` (crates/runmat-analysis/execution/src/tests/accuracy_plan_binding.rs:46) - The nonlinear accuracy source context accepts a completed block only under the authored accuracy plan and rejects a re-identified plan or loosened criterion; co...<br>`runmat-analysis-fea[lib]::execution::nonlinear_solid::tests::default_nonlinear_plan_is_decided_by_achieved_tangent_solve_residuals` (crates/runmat-analysis/fea/src/execution/nonlinear_solid/tests.rs:205) - The default structural-nonlinear accuracy plan (mandatory algebraic global-equilibrium residual bound) is decided by the achieved tangent-solve residual the rec... |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | native_process | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | durable | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## downstream_checkpoints_release

Evidence kind: AccuracyAcceptance. Criterion: Qualify selected nonlinear base states for buckling or prestressed downstream procedures only after their tangent, contact, material-history, and transition evidence passes. Add nonlinear task checkpoints and bounded retries. Add release evidence and documentation for each formulation separately.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| nonlinear_structural | tet4 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | native_process | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | durable | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## executable_laws_independent_physics

Evidence kind: AccuracyAcceptance. Criterion: Keep state-dependent follower loads, controllers, and constitutive/user laws outside ordinary excitation callbacks; reject them until the separate executable-law ABI and tangent contract exist. Equilibrium is satisfied by the implemented state-dependent operator. Constitutive and contact benchmarks pass independently.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| nonlinear_structural | tet4 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | native_process | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | durable | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## sensitivity_conflict_diagnostics

Evidence kind: AccuracyAcceptance. Criterion: Contact complementarity, equality-constraint violation, and overlap/conflict diagnostics pass independently. Load-step, mesh, and contact-parameter sensitivity are governed.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| nonlinear_structural | tet4 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | native_process | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | durable | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| nonlinear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

