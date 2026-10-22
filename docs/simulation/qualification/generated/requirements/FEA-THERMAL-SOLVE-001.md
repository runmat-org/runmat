# FEA-THERMAL-SOLVE-001: Distinct steady and transient solves with nonlinear closure

0 PASS, 4 FAIL, 4 OPEN, 0 TAGGED. Generated; do not edit.

## steady

Evidence kind: IndependentReference. Criterion: Solve stationary conductivity/source/boundary equilibrium and verify the physical residual on the original operator.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | FAIL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::fixed_hot_slab_matches_independent_gray_surface_scalar_equilibrium` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests.rs:39) - Tet4/Tet10 fixed-hot slab against independent scalar bisection; original fields, power and retained nonlinear evidence checked. |
| thermal_steady | tet10 | public_study | FAIL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::fixed_hot_slab_matches_independent_gray_surface_scalar_equilibrium` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests.rs:39) - Tet4/Tet10 fixed-hot slab against independent scalar bisection; original fields, power and retained nonlinear evidence checked. |

## transient

Evidence kind: TemporalConvergence. Criterion: Document the transient integration method, initial-state semantics and sampling; demonstrate its temporal order independently of spatial error.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_transient | tet4 | public_study | FAIL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::temporal::manufactured_continuum_cooling_retains_first_order_backward_euler_and_power_closure` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests/temporal.rs:9) - Tet4/Tet10 manufactured radiating cooling, 8/16/32 steps, first-order temperature/gradient/flux/storage and power closure; not temperature-dependent conductivit... |
| thermal_transient | tet10 | public_study | FAIL | `runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::temporal::manufactured_continuum_cooling_retains_first_order_backward_euler_and_power_closure` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests/temporal.rs:9) - Tet4/Tet10 manufactured radiating cooling, 8/16/32 steps, first-order temperature/gradient/flux/storage and power closure; not temperature-dependent conductivit... |

## nonlinear

Evidence kind: AccuracyAcceptance. Criterion: Temperature-dependent property and radiation iterations meet documented physical residual, field-increment and energy-closure criteria. Failed or exhausted iterations cannot produce an accepted state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

