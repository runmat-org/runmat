# FEA-THERMAL-INPUT-001: Exact thermal boundaries and shared excitation realization

2 PASS, 0 FAIL, 10 OPEN, 0 TAGGED. Generated; do not edit.

## boundaries

Evidence kind: IndependentReference. Criterion: Resolve exact face targets and integrate prescribed temperature, inward/outward heat-flux convention, convection and radiation laws with documented units, normals and nonlinear domain restrictions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::radiation::exact_steady_slab_radiation_residual_tangent_and_prescribed_power_match_continuum` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/radiation.rs:32) - Tet4 slab exact gray-surface law, finite-difference tangent and physical power; does not cover arbitrary enclosure radiation. |
| thermal_steady | tet10 | numerical_kernel | OPEN |  |
| thermal_transient | tet4 | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::signals::time_signals_drive_prescribed_temperature_flux_and_ambient_temperature` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/signals.rs:69) - Tet4 assembly samples prescribed temperature, flux and ambient-time signals; no generated/field-artifact equivalence credit. |
| thermal_transient | tet10 | numerical_kernel | OPEN |  |

## histories

Evidence kind: SemanticRealization. Criterion: Realize time-, parameter- and field-dependent heat generation, heat flux, ambient temperature and prescribed temperature through shared signals and authenticated realized field artifacts; reject unsupported samples and incompatible fields.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## equivalence

Evidence kind: ExecutionEquivalence. Criterion: Explicit, generated and field-artifact versions of equivalent inputs yield equivalent governed source and boundary energy under the same physical and numerical controls.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

