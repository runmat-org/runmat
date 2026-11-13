# FEA-EM-FREQUENCY-001: Multi-frequency harmonic electromagnetics and fields

0 PASS, 2 PARTIAL, 1 FAIL, 5 OPEN, 0 TAGGED. Generated; do not edit.

## multi_frequency_study

Evidence kind: PublicRun. Criterion: A single study runs 3, 10 and 30 kHz (or any authored frequency set) as one analysis with per-frequency fields and status in authored order.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | FAIL | `runmat[test:check_cli]::induction_gear_example_runs_its_three_frequency_harmonic_study` (crates/runmat-cli/tests/check_cli.rs:1271) - The induction-gear example runs three authored frequencies with a current-driven coil terminal from `runmat run`, one case per frequency, to a completed, numeri... |

## skin_effect_reference

Evidence kind: IndependentReference. Criterion: Conductor current density and AC resistance reproduce the skin-depth reference delta = sqrt(2 / (omega mu sigma)) for a round conductor or slab within the declared tolerance on a mesh that resolves delta.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::maxwell::tests::terminal::current_driven_strip_impedance_converges_to_the_skin_effect_solution` (crates/runmat-analysis/fea/src/physics/maxwell/tests/terminal.rs:253) - A current-driven conducting strip at t/delta of 0.5, 2 and 4 on 8, 16 and 32 cells: impedance converges to the closed-form skin-effect solution (error below 2e-... |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

## air_truncation

Evidence kind: SpatialConvergence. Criterion: Interior fields and coil loss change by less than the declared tolerance when the surrounding air domain is enlarged, so the outer boundary choice is documented as adequate.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

## ac_loss_power

Evidence kind: PhysicalBalance. Criterion: Cycle-averaged loss in a body (the induction-hardening gear at 10 kHz) equals the integral of J dot E over the body and matches terminal power minus stored-energy exchange.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::maxwell::tests::terminal::current_driven_strip_impedance_converges_to_the_skin_effect_solution` (crates/runmat-analysis/fea/src/physics/maxwell/tests/terminal.rs:253) - On the strip, the Joule loss equals the delivered complex power within 1e-8 and the Poynting and charge-continuity residuals stay below 1e-8; no multi-body gear... |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

