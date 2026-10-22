# FEA-THERMAL-FIELDS-001: Physical thermal recovery and energy accounting

0 PASS, 4 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## fields

Evidence kind: OriginalStateComparison. Criterion: Retain recoverable original thermal fields, integration inventories and boundary-face transfers with explicit locations, components, units and canonical topology; recover physical conductivity times temperature gradient with the documented sign.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## global_energy

Evidence kind: PhysicalBalance. Criterion: Account for supplied heat, boundary transfer, prescribed reactions and transient stored energy without counting internal conduction as external power or hiding opposing flows through cancellation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::internal_conduction_is_not_external_power_and_uniform_flux_is_exactly_zero` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:62) - Both-order physical heat-recovery identities and uniform zero flux; no arbitrary regional partition balance.<br>`runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::conductive_prescribed_reactions_have_analytical_sign_and_cancel_internal_pairs` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:114) - Both-order conductive prescribed reaction signs and internal-pair cancellation; no transient continuum convergence credit. |
| thermal_steady | tet10 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::internal_conduction_is_not_external_power_and_uniform_flux_is_exactly_zero` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:62) - Both-order physical heat-recovery identities and uniform zero flux; no arbitrary regional partition balance.<br>`runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::conductive_prescribed_reactions_have_analytical_sign_and_cancel_internal_pairs` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:114) - Both-order conductive prescribed reaction signs and internal-pair cancellation; no transient continuum convergence credit. |
| thermal_transient | tet4 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::internal_conduction_is_not_external_power_and_uniform_flux_is_exactly_zero` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:62) - Both-order physical heat-recovery identities and uniform zero flux; no arbitrary regional partition balance.<br>`runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::conductive_prescribed_reactions_have_analytical_sign_and_cancel_internal_pairs` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:114) - Both-order conductive prescribed reaction signs and internal-pair cancellation; no transient continuum convergence credit. |
| thermal_transient | tet10 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::internal_conduction_is_not_external_power_and_uniform_flux_is_exactly_zero` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:62) - Both-order physical heat-recovery identities and uniform zero flux; no arbitrary regional partition balance.<br>`runmat-analysis-fea[lib]::physics::thermal::assembly::tests::energy_recovery::conductive_prescribed_reactions_have_analytical_sign_and_cancel_internal_pairs` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/energy_recovery.rs:114) - Both-order conductive prescribed reaction signs and internal-pair cancellation; no transient continuum convergence credit. |

## regional_energy

Evidence kind: PhysicalBalance. Criterion: Compute per-region energy balances and consistent interface exchange so regional totals reconcile with the global balance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

