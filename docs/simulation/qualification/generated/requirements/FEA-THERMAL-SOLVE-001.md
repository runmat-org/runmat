# FEA-THERMAL-SOLVE-001: Distinct steady and transient solves with nonlinear closure

0 PASS, 11 PARTIAL, 0 FAIL, 4 OPEN, 0 TAGGED. Generated; do not edit.

## steady

Evidence kind: IndependentReference. Criterion: Solve stationary conductivity/source/boundary equilibrium and verify the physical residual on the original operator.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::fixed_hot_slab_matches_independent_gray_surface_scalar_equilibrium` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests.rs:42) - Tet4/Tet10 fixed-hot slab against independent scalar bisection; original fields, power and retained nonlinear evidence checked. |
| thermal_steady | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::fixed_hot_slab_matches_independent_gray_surface_scalar_equilibrium` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests.rs:42) - Tet4/Tet10 fixed-hot slab against independent scalar bisection; original fields, power and retained nonlinear evidence checked. |

## transient

Evidence kind: TemporalConvergence. Criterion: Document the transient integration method, initial-state semantics and sampling; demonstrate its temporal order independently of spatial error.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_transient | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::temporal::manufactured_continuum_cooling_retains_first_order_backward_euler_and_power_closure` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests/temporal.rs:9) - Tet4/Tet10 manufactured radiating cooling, 8/16/32 steps, first-order temperature/gradient/flux/storage and power closure; not temperature-dependent conductivit... |
| thermal_transient | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::temporal::manufactured_continuum_cooling_retains_first_order_backward_euler_and_power_closure` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests/temporal.rs:9) - Tet4/Tet10 manufactured radiating cooling, 8/16/32 steps, first-order temperature/gradient/flux/storage and power closure; not temperature-dependent conductivit... |

## nonlinear

Evidence kind: AccuracyAcceptance. Criterion: Temperature-dependent property and radiation iterations meet documented physical residual, field-increment and energy-closure criteria. Failed or exhausted iterations cannot produce an accepted state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## conduction_multigrid

Evidence kind: ExecutionEquivalence. Criterion: An AMG-preconditioned CG steady conduction solve, chosen when the factor does not fit the admitted budget, agrees with the proving direct factorization within its tolerance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::conduction_tests::conduction_multigrid_matches_the_proving_factorization` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/conduction_tests.rs:90) - On a two-layer joint box with a contact conductance, a budget below the proving factor selects the conduction multigrid (AMG-PCG, no proving factor), and its te... |

## conduction_multigrid_usage

Evidence kind: RejectionInvariant. Criterion: A ledger refusal inside the multigrid conduction solve stays a typed usage refusal.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::conduction_tests::conduction_multigrid_accounting_refusal_is_typed_usage` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/conduction_tests.rs:128) - A usage allowance that runs out inside the conduction multigrid solve refuses the run with the typed usage error. |

## conduction_multigrid_anchor

Evidence kind: RejectionInvariant. Criterion: A conduction model with a body that has no prescribed temperature or positive convection is refused as undetermined before the multigrid path solves.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::conduction_tests::conduction_multigrid_refuses_an_unanchored_body` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/conduction_tests.rs:159) - A body with no prescribed temperature and no convection is refused on the conduction multigrid path with an error naming the undetermined temperature. |

## conduction_multigrid_plan

Evidence kind: ResourceAdmission. Criterion: Solver selection compares the whole preassembly totals of the factor and multigrid paths against the admitted budget.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::planning::resource::thermal_operator::tests::conduction_multigrid_solver_storage_grows_slower_than_the_factor` (crates/runmat-analysis/fea/src/planning/resource/thermal_operator/tests.rs:299) - The conduction multigrid solver term (AMG, PCG vectors, proof inventory) is below the proving factor's on box meshes and the gap grows with the mesh; on these s... |

## steady_reservation

Evidence kind: ResourceAdmission. Criterion: The steady thermal reservation stays at or above the measured whole-run heap peak.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::conduction_tests::steady_run_peak_fits_the_preassembly_reservation` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/conduction_tests.rs:216) - The measured requested-heap peak of a whole steady linear run, on both solve paths, fits the preassembly reservation the run admits. |

## support_derivation_memory

Evidence kind: ResourceAdmission. Criterion: Thermal support-pattern derivation stays within its distinct-entry planning law.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::planning::resource::thermal_operator::tests::thermal_support_derivation_peak_fits_the_planning_law` (crates/runmat-analysis/fea/src/planning/resource/thermal_operator/tests.rs:406) - The thermal support derivation (projection, clique support, symbolic factor analysis) never requests more heap than the planning law at the exact support size, ... |

## support_derivation_memory_interface

Evidence kind: ResourceAdmission. Criterion: Support-pattern derivation with finite-conductance interfaces stays within its planning law.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::conduction_tests::split_interface_support_derivation_fits_the_planning_law` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/conduction_tests.rs:190) - With a finite contact conductance the split projection and the interface cliques stay within the planning law. |

