# FEA-EM-FREQUENCY-001: Multi-frequency harmonic electromagnetics and fields

0 PASS, 0 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## multi_frequency_study

Evidence kind: PublicRun. Criterion: A single study runs 3, 10 and 30 kHz (or any authored frequency set) as one analysis with per-frequency fields and status in authored order.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

## skin_effect_reference

Evidence kind: IndependentReference. Criterion: Conductor current density and AC resistance reproduce the skin-depth reference delta = sqrt(2 / (omega mu sigma)) for a round conductor or slab within the declared tolerance on a mesh that resolves delta.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
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
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

