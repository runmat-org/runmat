# FEA-ACOUSTIC-NUMERICS-001: Independent acoustic fields, resonance and physical balances

0 PASS, 5 PARTIAL, 0 FAIL, 5 OPEN, 0 TAGGED. Generated; do not edit.

## duct

Evidence kind: SpatialConvergence. Criterion: Independent duct/impedance traveling waves verify complex pressure/velocity phase, intensity, energy and boundary power under mesh refinement.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::continuum::tet10_traveling_wave_phase_pressure_velocity_and_power_converge` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs:79) - Tet10 slab at two frequencies, three refinements; exact traveling-wave references, not all acoustic geometries. |

## cavity

Evidence kind: IndependentReference. Criterion: Rigid cavity references verify modal/resonance frequencies, fields and phase reversal on both sides of poles under refinement; pole response is not accepted solely because its physical right-hand side is zero.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::continuum::tet4_and_tet10_forced_rigid_cavity_converge_on_both_sides_of_axial_resonance` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs:145) - Forced axial cavity at 0.9/1.1 times first pole, phase/pressure/energy convergence; no complete multidimensional modal-spectrum credit. |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::continuum::tet4_and_tet10_forced_rigid_cavity_converge_on_both_sides_of_axial_resonance` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs:145) - Forced axial cavity at 0.9/1.1 times first pole, phase/pressure/energy convergence; no complete multidimensional modal-spectrum credit. |

## conditioning

Evidence kind: IndependentReference. Criterion: Report source-independent resonance conditioning or sensitivity diagnostics over the supported frequency envelope and independently verify modal frequencies. A sufficient nonsingularity certificate does not establish good conditioning, bounded forward error or complete modal-spectrum accuracy.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |

## uniqueness

Evidence kind: IndependentReference. Criterion: The sparse LU that solves the original reduced operator certifies its nonsingularity (estimated right-inverse defect contraction below the authored limit) before physical residual/power closure is accepted; above the limit the solve fails closed. No artificial damping or sampled small matrix substitutes for the operator.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::response::reduced::tests::certificate::original_canonical_lossless_discrete_pole_fails_closed_despite_zero_rhs` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/response/reduced/tests/certificate.rs:10) - Independent Tet4 antisymmetric discrete pole fails the factorization certificate despite a zero right-hand side; the same operator off the pole is certified. |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |

## cold_claim

Evidence kind: OriginalStateComparison. Criterion: Retained factorization certificates bind the authored limit, original operator identity, result and numerical receipt. Cold hashes or decoded producer claims never become trusted mathematical proof; independent replay requires refactoring the original operator.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::acoustic::tests::certificate::retained_certificate_survives_solve_free_cold_lineage_and_tampering_fails` (crates/runmat-analysis/fea/src/execution/acoustic/tests/certificate.rs:6) - Retained certificate/receipt provenance and tamper rejection; no cold refactorization. |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |

