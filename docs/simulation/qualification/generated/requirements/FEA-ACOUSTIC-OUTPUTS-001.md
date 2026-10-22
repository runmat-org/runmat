# FEA-ACOUSTIC-OUTPUTS-001: Sound pressure level, transmission loss, intensity and power

0 PASS, 0 PARTIAL, 0 FAIL, 20 OPEN, 0 TAGGED. Generated; do not edit.

## sound_pressure_level

Evidence kind: IndependentReference. Criterion: The `level` output and an SPL field (dB re 20 micropascal, rms, optional A-weighting) publish per frequency from a harmonic study; a plane wave of 1 Pa amplitude reads 91.0 dB; point outputs interpolate the field at an authored position.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

## transmission_and_insertion_loss

Evidence kind: IndependentReference. Criterion: `transmission_loss` from incident and transmitted power (two-port) over a one-third-octave sweep 50 Hz to 2 kHz reproduces the simple expansion chamber TL = 10 log10(1 + 1/4 (m - 1/m)^2 sin^2(kL)) up to the first cross mode; the muffler study reports peak TL, its band, the lowest TL above the first peak and the bands below 15 dB.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

## intensity_and_sound_power

Evidence kind: IndependentReference. Criterion: The active intensity field and `sound_power` through a named surface publish; power through a sphere around a monopole equals rho c k^2 Q^2 / (8 pi) within 1 percent.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

## frequency_set_generation

Evidence kind: PublicRun. Criterion: A study generates one-third-octave band centers (and other authored frequency sets), runs each frequency, and publishes outputs in authored order with per-frequency status.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

## output_arithmetic_across_analyses

Evidence kind: SemanticRealization. Criterion: Outputs of two analyses in one study (treated and bare room) combine into a difference output such as the SPL reduction at the operator position.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

