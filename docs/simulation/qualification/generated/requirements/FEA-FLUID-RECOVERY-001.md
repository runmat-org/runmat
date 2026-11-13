# FEA-FLUID-RECOVERY-001: Original fluid fields and physical conservation recovery

0 PASS, 2 PARTIAL, 0 FAIL, 14 OPEN, 0 TAGGED. Generated; do not edit.

## fields

Evidence kind: OriginalStateComparison. Criterion: Retain original cell fields and owner-face flux; recover face traces, stresses and wall quantities with declared location, units, frames and sign conventions. Cold consumers validate original source and reject changed fields/tractions rather than silently re-solving them.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::fluid::tests::original_velocity_pressure_flux_traction_and_resource_substitution_reject` (crates/runmat-analysis/fea/src/execution/fluid/tests.rs:521) - Portable original field/traction/resource substitution rejection; no process termination.<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::boundary_balances::original_couette_boundary_shear_and_cold_attribution_are_preserved` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/boundary_balances.rs:121) - Couette boundary shear/source attribution preserved under cold recovery; one fixture only. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## balance

Evidence kind: PhysicalBalance. Criterion: Report global, per-cell and per-boundary signed mass and momentum components from original forces/fluxes with physical normalization. Algebraic residual convergence alone cannot pass physical acceptance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::boundary_balances::grouped_original_translation_reports_signed_mass_and_pressure_forces` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/boundary_balances.rs:13) - Independent constant translation exact group signs/pressure forces; no general wall-shear convergence. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

