# FEA-ACOUSTIC-OUTPUTS-001: Sound pressure level, transmission loss, intensity and power

0 PASS, 1 PARTIAL, 1 FAIL, 18 OPEN, 0 TAGGED. Generated; do not edit.

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
| acoustic_harmonic | tet10 | public_study | FAIL | `runmat[test:check_cli]::two_fluid_interface_example_publishes_the_plane_wave_transmission_loss` (crates/runmat-cli/tests/check_cli.rs:1198) - transmission_loss on the public two-fluid interface example (m = 4) lies within 0.1 dB of 1.938 dB at 500, 1000 and 2000 Hz; no muffler or one-third-octave swee... |

## intensity_and_sound_power

Evidence kind: IndependentReference. Criterion: The active intensity field and `sound_power` through a named surface publish; power through a sphere around a monopole equals rho c k^2 Q^2 / (8 pi) within 1 percent.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::acoustic::tests::transmission_loss::transmission_loss_of_a_layer_matches_the_plane_wave_reference` (crates/runmat-analysis/fea/src/execution/acoustic/tests/transmission_loss.rs:208) - transmission loss through a plane layer between an incident and an anechoic port matches the closed-form plane-wave result per frequency and converges under ref...<br>`runmat-analysis-fea[lib]::execution::acoustic::tests::sound_power_through_the_matched_end_is_the_plane_wave_intensity_times_area` (crates/runmat-analysis/fea/src/execution/acoustic/tests.rs:178) - plane-wave power through a matched duct end equals |p|²A/(2ρc) and the accounted input power to 0.1 %; the monopole sphere reference is not run. |
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

