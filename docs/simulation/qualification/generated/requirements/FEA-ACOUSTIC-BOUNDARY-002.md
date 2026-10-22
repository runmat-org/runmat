# FEA-ACOUSTIC-BOUNDARY-002: Radiation, frequency-dependent impedance and air damping

0 PASS, 0 FAIL, 20 OPEN, 0 TAGGED. Generated; do not edit.

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
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

## region_damping

Evidence kind: IndependentReference. Criterion: Loss factor or complex sound speed per region is accepted at assembly (now rejected) and a damped duct resonance peak width matches the closed form.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
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

