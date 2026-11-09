# FEA-ACOUSTIC-BOUNDARY-001: Exact typed acoustic boundaries and complex constraints

0 PASS, 7 PARTIAL, 0 FAIL, 1 OPEN, 0 TAGGED. Generated; do not edit.

## lifting

Evidence kind: IndependentReference. Criterion: Complex prescribed pressure lifting agrees with an independently assembled reduced block system.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::radiation::exact_operators::complex_prescribed_pressure_lifting_matches_independent_reduced_block_system` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/radiation/exact_operators.rs:337) - Tet10 lifted complex system checked against independent dense oracle; no generic solve or conditioning claim. |

## phase

Evidence kind: SpatialConvergence. Criterion: Periodic and Bloch/Floquet constraints preserve phase and paired traces, pressure, velocity and signed power against independent references under refinement.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::periodic::anchored_bloch_plane_wave_pressure_mean_velocity_and_signed_intensity_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/periodic.rs:84) - Anchored slab plane wave and declared paired planes; no unanchored pole/global conditioning credit. |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::periodic::anchored_bloch_plane_wave_pressure_mean_velocity_and_signed_intensity_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/periodic.rs:84) - Anchored slab plane wave and declared paired planes; no unanchored pole/global conditioning credit. |

## radiation

Evidence kind: SpatialConvergence. Criterion: Supported radiation operator and recovered pressure, phase, mean velocity and signed power satisfy independent references and spatial convergence on their declared geometry/frequency envelope.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::radiation::local_radiation_pressure_phase_mean_velocity_and_signed_power_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/radiation.rs:155) - Local radiation reference and original spatial gates; witness linkage is not a passing verdict or nonlocal exterior-wave qualification. |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::radiation::local_radiation_pressure_phase_mean_velocity_and_signed_power_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/radiation.rs:155) - Local radiation reference and original spatial gates; witness linkage is not a passing verdict or nonlocal exterior-wave qualification. |

## reject

Evidence kind: RejectionInvariant. Criterion: Reject unmatched periodic pairs, unsupported radiation/incident regions, conflicting prescribed pressure and overlapping sources/natural boundaries without guessing a target or constraint.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::periodic::periodic_rejects_unmatched_natural_overlap_and_conflicting_pressure` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/periodic.rs:132) - Periodic pairing, overlap and conflicting pressure rejection only. |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::periodic::periodic_rejects_unmatched_natural_overlap_and_conflicting_pressure` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/periodic.rs:132) - Periodic pairing, overlap and conflicting pressure rejection only. |

