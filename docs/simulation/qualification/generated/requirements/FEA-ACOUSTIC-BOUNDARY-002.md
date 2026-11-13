# FEA-ACOUSTIC-BOUNDARY-002: Radiation, frequency-dependent impedance and air damping

0 PASS, 3 PARTIAL, 0 FAIL, 17 OPEN, 0 TAGGED. Generated; do not edit.

## spherical_radiation

Evidence kind: IndependentReference. Criterion: A spherical radiation boundary on an outer sphere or hemisphere converges toward the exact impedance: a pulsating sphere's surface pressure is within 2 percent of rho c u i k a / (1 + i k a) at ka = 1.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

## complex_frequency_dependent_impedance

Evidence kind: IndependentReference. Criterion: An impedance boundary takes a complex value and a frequency table, and a `delany_bazley` porous layer (flow resistivity, thickness); normal-incidence absorption of a 50 mm layer matches the Delany-Bazley closed form; the room study's treated and bare ceilings run.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::complex_impedance_termination_converges_to_the_two_wave_duct_solution` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests.rs:718) - a tabulated complex termination converges to the two-wave duct solution; Delany-Bazley is checked in core only.<br>`runmat-analysis-language[test:impedance_tables]::delany_bazley_and_sampled_impedance_tables_lower_into_their_boundaries` (crates/runmat-analysis/language/tests/impedance_tables.rs:58) - Language lowering of Delany-Bazley and sampled impedance tables only; no solve against the closed-form absorption. |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-language[test:impedance_tables]::delany_bazley_and_sampled_impedance_tables_lower_into_their_boundaries` (crates/runmat-analysis/language/tests/impedance_tables.rs:58) - Language lowering of Delany-Bazley and sampled impedance tables only; no solve against the closed-form absorption. |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

## region_damping

Evidence kind: IndependentReference. Criterion: Loss factor or complex sound speed per region is accepted at assembly (now rejected) and a damped duct resonance peak width matches the closed form.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::damped_duct_converges_to_the_complex_wavenumber_standing_wave_and_dissipates_its_input` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests.rs:379) - a lossy duct (loss factor 2ζ) converges at second order to the complex-wavenumber standing wave and dissipates its input power; resonance width not asserted. |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |
| acoustic_transient | tet4 | numerical_kernel | OPEN |  |
| acoustic_transient | tet4 | public_study | OPEN |  |
| acoustic_transient | tet10 | numerical_kernel | OPEN |  |
| acoustic_transient | tet10 | public_study | OPEN |  |

## bloch_band_gap

Evidence kind: IndependentReference. Criterion: A periodic unit cell with Bloch boundaries yields the band gap of a rod array, compared against an independent plane-wave expansion.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

