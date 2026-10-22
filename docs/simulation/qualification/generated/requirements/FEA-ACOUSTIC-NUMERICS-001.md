# FEA-ACOUSTIC-NUMERICS-001: Independent acoustic fields, resonance and physical balances

0 PASS, 5 FAIL, 5 OPEN, 0 TAGGED. Generated; do not edit.

## duct

Evidence kind: SpatialConvergence. Criterion: Independent duct/impedance traveling waves verify complex pressure/velocity phase, intensity, energy and boundary power under mesh refinement.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::continuum::tet10_traveling_wave_phase_pressure_velocity_and_power_converge` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs:76) - Tet10 slab at two frequencies, three refinements; exact traveling-wave references, not all acoustic geometries. |

## cavity

Evidence kind: IndependentReference. Criterion: Rigid cavity references verify modal/resonance frequencies, fields and phase reversal on both sides of poles under refinement; pole response is not accepted solely because its physical right-hand side is zero.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::continuum::tet4_and_tet10_forced_rigid_cavity_converge_on_both_sides_of_axial_resonance` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs:140) - Forced axial cavity at 0.9/1.1 times first pole, phase/pressure/energy convergence; no complete multidimensional modal-spectrum credit. |
| acoustic_harmonic | tet10 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::continuum::tet4_and_tet10_forced_rigid_cavity_converge_on_both_sides_of_axial_resonance` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/continuum.rs:140) - Forced axial cavity at 0.9/1.1 times first pole, phase/pressure/energy convergence; no complete multidimensional modal-spectrum credit. |

## conditioning

Evidence kind: IndependentReference. Criterion: Report source-independent resonance conditioning or sensitivity diagnostics over the supported frequency envelope and independently verify modal frequencies. A sufficient nonsingularity certificate does not establish good conditioning, bounded forward error or complete modal-spectrum accuracy.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |

## uniqueness

Evidence kind: IndependentReference. Criterion: Source-independent original reduced-operator evidence establishes requested uniqueness before accepting physical residual/power closure. An unsuccessful sufficient check is indeterminate, not a singularity diagnosis; no artificial damping or sampled small matrix substitutes for the operator.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::response::reduced::tests::uniqueness::original_canonical_lossless_discrete_pole_is_not_proved_by_zero_rhs` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/response/reduced/tests/uniqueness.rs:52) - Independent Tet4 antisymmetric discrete pole; requested diagnostic refuses zero-RHS residual acceptance, without asserting exact floating rank. |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |

## cold_claim

Evidence kind: OriginalStateComparison. Criterion: Retained uniqueness records bind original controls, operator/result and numerical receipt. Cold hashes or decoded producer claims never become trusted mathematical proof; independent replay requires original operator and admitted witness/candidate reconstruction.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::execution::acoustic::tests::uniqueness::required_diagnostic_survives_solve_free_cold_lineage_and_tampering_fails` (crates/runmat-analysis/fea/src/execution/acoustic/tests/uniqueness.rs:7) - Retained producer claim/receipt provenance and tamper rejection; no cold mathematical uniqueness reproof. |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |

