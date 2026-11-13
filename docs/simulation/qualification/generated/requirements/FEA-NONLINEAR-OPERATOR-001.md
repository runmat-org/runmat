# FEA-NONLINEAR-OPERATOR-001: State-dependent nonlinear mechanics and supported plastic laws

0 PASS, 6 PARTIAL, 0 FAIL, 34 OPEN, 0 TAGGED. Generated; do not edit.

## kinematics_tangent_newton

Evidence kind: IndependentReference. Criterion: Define nonlinear kinematics supported initially and reject unsupported combinations. Assemble internal force and consistent or documented algorithmic tangent from current state. Implement Newton iteration with real residual and increment criteria, line search, and load/time stepping.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| small_strain_j2 | tet4 | numerical_kernel | OPEN |  |
| small_strain_j2 | tet10 | numerical_kernel | OPEN |  |
| small_strain_j2 | tet4 | public_study | OPEN |  |
| small_strain_j2 | tet10 | public_study | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | native_process | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | durable | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | remote_execution | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | parallel_cpu | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | domain_decomposed | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## point_state_constitutive_laws

Evidence kind: IndependentReference. Criterion: Store integration-point constitutive state. Implement return mapping and tangent for each explicitly supported plastic law. Feed plastic state into equilibrium and recovery.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| small_strain_j2 | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::physical_newton_matches_independent_uniaxial_plastic_bar_and_cold_restart_both_orders` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs:95) - Physical Newton matches an independent uniaxial plastic bar and a cold restart for both orders; cyclic paths and other laws are not covered. |
| small_strain_j2 | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::physical_newton_matches_independent_uniaxial_plastic_bar_and_cold_restart_both_orders` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs:95) - Physical Newton matches an independent uniaxial plastic bar and a cold restart for both orders; cyclic paths and other laws are not covered. |
| small_strain_j2 | tet4 | public_study | OPEN |  |
| small_strain_j2 | tet10 | public_study | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | native_process | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | durable | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | remote_execution | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | parallel_cpu | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | domain_decomposed | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## canonical_state_physical_cutbacks

Evidence kind: IndependentReference. Criterion: Persist state in canonical artifacts suitable for restart and adaptation transfer. Add cutback/retry policies driven by physical convergence rather than synthetic scalar damping.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| small_strain_j2 | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::physical_newton_matches_independent_uniaxial_plastic_bar_and_cold_restart_both_orders` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs:95) - The plastic bar solve restarts cold from persisted state for both orders; adaptation transfer of state is not covered. |
| small_strain_j2 | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::physical_newton_matches_independent_uniaxial_plastic_bar_and_cold_restart_both_orders` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs:95) - The plastic bar solve restarts cold from persisted state for both orders; adaptation transfer of state is not covered. |
| small_strain_j2 | tet4 | public_study | OPEN |  |
| small_strain_j2 | tet10 | public_study | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | native_process | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | durable | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | remote_execution | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | parallel_cpu | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | domain_decomposed | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## references

Evidence kind: IndependentReference. Criterion: Evidence includes constitutive point tests, elastoplastic bars, cyclic paths where supported, large-displacement beams, limit-point cases only when the continuation method supports them, and mesh/load-step convergence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| small_strain_j2 | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::nonlinear::solid::continuum_qualification::manufactured_fully_yielded_cube_refines_and_checks_load_step_sensitivity` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/continuum_qualification.rs:116) - Original uniform Tet4 continuum gate asserts displacement/stress/point-history/energy/reaction refinement and load schedule sensitivity; linked executable is un...<br>`runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::physical_newton_matches_independent_uniaxial_plastic_bar_and_cold_restart_both_orders` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs:95) - Independent uniaxial plastic bar reference with cold restart for both orders; the rest of the listed references (cyclic paths, thick cylinder) are not covered. |
| small_strain_j2 | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::nonlinear_solid::tests::integration::curved_tet10_nonlinear_assembly_integrates_on_the_shared_rule_points` (crates/runmat-analysis/fea/src/assembly/nonlinear_solid/tests/integration.rs:77) - a curved Tet10 integrates its per-point nonlinear state on the converged points of the shared Tet10 element-validity rule (not the requested Keast15), and its e...<br>`runmat-analysis-fea[lib]::solve::nonlinear::solid::continuum_qualification::fresh_integration::fresh_tet10_integration_trajectories_meet_original_eight_step_exactness` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/continuum_qualification/fresh_integration.rs:13) - fresh trajectories under Keast15, Duffy5 and Duffy8 are exact in one increment and their eight-increment path error (the force-controlled yield front crossing e...<br>`runmat-analysis-fea[lib]::solve::nonlinear::solid::continuum_qualification::manufactured_fully_yielded_tet10_cube_is_exact_on_both_load_schedules` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/continuum_qualification.rs:133) - one increment reproduces the polynomial J2 solution to roundoff; eight increments carry a discrete path error because the force-controlled yield front crosses e...<br>`runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::physical_newton_matches_independent_uniaxial_plastic_bar_and_cold_restart_both_orders` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests.rs:95) - Independent uniaxial plastic bar reference with cold restart for both orders; the rest of the listed references (cyclic paths, thick cylinder) are not covered. |
| small_strain_j2 | tet4 | public_study | OPEN |  |
| small_strain_j2 | tet10 | public_study | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | native_process | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | durable | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | remote_execution | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | parallel_cpu | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | domain_decomposed | OPEN |  |
| small_strain_j2 | canonical_mixed_topology | claimed_accelerator | OPEN |  |

