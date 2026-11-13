# FEA-ACOUSTIC-SOURCE-001: Shared complex source spectra with conserved realization

0 PASS, 4 PARTIAL, 0 FAIL, 2 OPEN, 0 TAGGED. Generated; do not edit.

## resultant

Evidence kind: PhysicalBalance. Criterion: Total volume flow preserves the authored complex patch resultant independent of area or refinement and agrees with equivalent normal velocity without duplicating that resultant.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::volume_velocity::total_volume_flow_area_phase_rotation_and_power_refine_against_continuum` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/volume_velocity.rs:92) - Area scaling, complex phase and independent flow/power refinement on slab fixtures. |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::volume_velocity::total_volume_flow_area_phase_rotation_and_power_refine_against_continuum` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/volume_velocity.rs:92) - Area scaling, complex phase and independent flow/power refinement on slab fixtures. |

## incident

Evidence kind: SpatialConvergence. Criterion: Incident/reflected/transmitted pressure and net power converge to independent references; incoming power is not counted twice at small amplitude.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::incident::incoming_pressure_reflection_transmission_and_net_power_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/incident.rs:37) - Planar incident/impedance fixture, phase and net-power refinement only. |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::incident::incoming_pressure_reflection_transmission_and_net_power_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/incident.rs:37) - Planar incident/impedance fixture, phase and net-power refinement only. |

## spectra

Evidence kind: ExecutionEquivalence. Criterion: Explicit and frozen generated pressure/normal-velocity/volume-flow/incident spectra produce equivalent original complex fields and boundary power across the authored frequency inventory, including cold publication.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

