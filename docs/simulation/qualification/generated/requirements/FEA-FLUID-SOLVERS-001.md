# FEA-FLUID-SOLVERS-001: Robust, scalable and accurate laminar flow solvers

0 PASS, 2 PARTIAL, 0 FAIL, 34 OPEN, 0 TAGGED. Generated; do not edit.

## coupled_newton_continuation

Evidence kind: IndependentReference. Criterion: A fully coupled pressure-based Newton solver with pseudo-transient continuation, Reynolds and rheology continuation and line search converges the Ghia lid-driven cavity at Re 1000 from rest in at most 15 Newton steps, with psi_min 0.1189366 at (0.5308, 0.5652) against the Botella-Peyret reference; stagnation is reported in the run evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::symmetry::thin_cells_on_symmetry_planes_converge_promptly` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/symmetry.rs:270) - thin cells on a symmetry plane (spanwise grading z⁴, first layer 1/256 of the span) converge in at most five coupled steps with the assembled momentum Jacobian ...<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::coupled_solve_reaches_the_segregated_fixed_point_in_far_fewer_outer_iterations` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs:624) - the coupled Picard solve (FGMRES, SIMPLE block preconditioner) reaches the segregated solver's discrete solution in a fraction of the outer iterations; Newton l... |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |

## residual_history

Evidence kind: PublicationIntegrity. Criterion: Per-iteration residual history and nonlinear and linear iteration counts publish as results for every flow run.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |

## block_linear_solvers

Evidence kind: PerformanceMeasurement. Criterion: Flow linear systems solve with FGMRES using block-triangular preconditioning (PCD or LSC Schur approximations, augmented Lagrangian at high Reynolds number) and AMG on velocity and pressure blocks, through the shared sparse layer including its GPU path; linear iterations stay within 1.5 times from 100 thousand to 10 million cells on the 3D cavity.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::symmetry::assembled_momentum_block_matches_the_matrix_free_rows` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/symmetry.rs:292) - the assembled two-ring momentum Jacobian reproduces the matrix-free momentum rows to roundoff on a graded slab with symmetry planes, a pressure outlet, inflow a... |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet4 | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | parallel_cpu | OPEN |  |

## second_order_time

Evidence kind: TemporalConvergence. Criterion: BDF2 with adaptive steps is the default transient scheme and a rotational pressure-correction projection scheme serves long transients: Schafer-Turek 2D-2 (Re 100) gives c_D,max 3.22 to 3.24, c_L,max 0.99 to 1.01 and Strouhal number 0.295 to 0.305; 2D-3 gives c_D,max 2.950921 and c_L,max 0.47795 (references checked against the paper before use).

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |

## accurate_discretization

Evidence kind: SpatialConvergence. Criterion: Second-order coupled finite volume with non-orthogonal correction and polyhedral dual, and P2-P1 or VMS finite elements on Tet10, reach design order on Kovasznay flow at Re 40 and reproduce Schafer-Turek 2D-1 (c_D 5.579535, c_L 0.0106189, pressure drop 0.1175201) and 3D-1Z (c_D 6.185333, c_L 0.009401, pressure drop 0.171007).

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |

