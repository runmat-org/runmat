# FEA-ACOUSTIC-TRANSIENT-001: Acoustic transient analysis

0 PASS, 1 PARTIAL, 0 FAIL, 11 OPEN, 0 TAGGED. Generated; do not edit.

## duct_pulse

Evidence kind: IndependentReference. Criterion: A one-dimensional pulse in a rigid duct matches d'Alembert travel time and amplitude to the declared tolerance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_transient | tet4 | numerical_kernel | OPEN |  |
| acoustic_transient | tet4 | public_study | OPEN |  |
| acoustic_transient | tet10 | numerical_kernel | OPEN |  |
| acoustic_transient | tet10 | public_study | OPEN |  |

## pressure_time_history_publication

Evidence kind: PublicationIntegrity. Criterion: An `acoustic transient` run publishes the complete ordered time history (time axis and one pressure snapshot per step including the initial state) as retained fields that are finite, bounded by the authored end time and readable cold, with the prescribed boundary carrying the authored pulse exactly.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_transient | tet4 | numerical_kernel | OPEN |  |
| acoustic_transient | tet4 | public_study | OPEN |  |
| acoustic_transient | tet10 | numerical_kernel | OPEN |  |
| acoustic_transient | tet10 | public_study | PARTIAL | `runmat[test:typed_fea_geometry_cli]::acoustic::response::pressure_transient_publishes_its_complete_time_history` (crates/runmat-cli/tests/typed_fea_geometry_cli/acoustic/response.rs:120) - A quadratic-mesh pulse study run through the CLI publishes 21 native pressure snapshots and the complete authored history on a 21-point time axis ending at 2 ms... |

## energy_balance

Evidence kind: PhysicalBalance. Criterion: The transient energy balance (stored plus radiated plus dissipated minus supplied) closes to the declared tolerance, and time-step convergence is demonstrated.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_transient | tet4 | numerical_kernel | OPEN |  |
| acoustic_transient | tet4 | public_study | OPEN |  |
| acoustic_transient | tet10 | numerical_kernel | OPEN |  |
| acoustic_transient | tet10 | public_study | OPEN |  |

