# FEA-THERMAL-ACCURACY-001: Independent thermal references and refinement qualification

0 PASS, 3 PARTIAL, 0 FAIL, 11 OPEN, 0 TAGGED. Generated; do not edit.

## references

Evidence kind: IndependentReference. Criterion: Pass independent patch, internally heated slab, plate and cylinder, convection and lumped-capacity references across the supported orders and physical boundary/source regimes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## space

Evidence kind: SpatialConvergence. Criterion: Demonstrate spatial convergence of temperature, gradient, flux, stored heat and boundary transfer against independent continuum references while bounding temporal contamination.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::thermal_workload::tests::initial_fields::continuum::resolved_tet4_heated_cube_cold_cooling_has_continuum_spatial_convergence` (crates/runmat-analysis/fea/src/execution/thermal_workload/tests/initial_fields/continuum.rs:132) - Tet4 heated-cube cold cooling continuum temperature, flux, stored heat and power with temporal contamination bound; does not prove Tet10 spatial gate. |
| thermal_transient | tet10 | public_study | OPEN |  |

## time

Evidence kind: TemporalConvergence. Criterion: Demonstrate temporal convergence against independent continuum references on spatially controlled meshes, including heating and cooling and nonlinear boundary regimes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_transient | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::thermal_workload::tests::initial_fields::continuum::uniform_ramp_heating_has_continuum_backward_euler_time_convergence` (crates/runmat-analysis/fea/src/execution/thermal_workload/tests/initial_fields/continuum.rs:19) - Both-order uniform ramp heating, independent continuum error and energy law; no nonuniform mesh convergence credit. |
| thermal_transient | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::thermal_workload::tests::initial_fields::continuum::uniform_ramp_heating_has_continuum_backward_euler_time_convergence` (crates/runmat-analysis/fea/src/execution/thermal_workload/tests/initial_fields/continuum.rs:19) - Both-order uniform ramp heating, independent continuum error and energy law; no nonuniform mesh convergence credit. |

## gate

Evidence kind: AccuracyAcceptance. Criterion: Mandatory thermal accuracy assessments bind original fields, energy diagnostics and realization evidence to the result and block acceptance when missing, failed or indeterminate.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

