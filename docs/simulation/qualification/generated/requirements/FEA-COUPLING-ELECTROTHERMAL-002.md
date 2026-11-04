# FEA-COUPLING-ELECTROTHERMAL-002: Two-way and transient Joule and AC-loss heating

0 PASS, 1 PARTIAL, 0 FAIL, 23 OPEN, 0 TAGGED. Generated; do not edit.

## two_way_conductivity_feedback

Evidence kind: IndependentReference. Criterion: `electrothermal steady` with sigma(T) = sigma0 / (1 + alpha (T - T0)) iterates to a converged fixed point; the busbar with alpha = 0.00393 per K reproduces power equal to V times I.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrothermal | tet4 | numerical_kernel | OPEN |  |
| electrothermal | tet4 | public_study | OPEN |  |
| electrothermal | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::electrothermal::tests::coupled::two_way_joule_heating_matches_the_closed_form_bar` (crates/runmat-analysis/fea/src/execution/electrothermal/tests/coupled.rs:108) - Two-way coupled steady Joule heating on a Tet10 bar with temperature-dependent resistivity iterates at least three sweeps; temperature matches the closed-form c... |
| electrothermal | tet10 | public_study | OPEN |  |
| electrothermal | edge_elements | numerical_kernel | OPEN |  |
| electrothermal | edge_elements | public_study | OPEN |  |

## transient_electrothermal

Evidence kind: TemporalConvergence. Criterion: `electrothermal transient` runs with the same coupling and matches a lumped adiabatic heating reference over a short time and steady state over a long time.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrothermal | tet4 | numerical_kernel | OPEN |  |
| electrothermal | tet4 | public_study | OPEN |  |
| electrothermal | tet10 | numerical_kernel | OPEN |  |
| electrothermal | tet10 | public_study | OPEN |  |
| electrothermal | edge_elements | numerical_kernel | OPEN |  |
| electrothermal | edge_elements | public_study | OPEN |  |

## ac_loss_into_transient_heat

Evidence kind: PhysicalBalance. Criterion: Cycle-averaged harmonic loss from the gear and inductor drives a 3 s transient heat step with the loss field mapped conservatively (total power preserved), reporting peak gear surface temperature.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrothermal | tet4 | numerical_kernel | OPEN |  |
| electrothermal | tet4 | public_study | OPEN |  |
| electrothermal | tet10 | numerical_kernel | OPEN |  |
| electrothermal | tet10 | public_study | OPEN |  |
| electrothermal | edge_elements | numerical_kernel | OPEN |  |
| electrothermal | edge_elements | public_study | OPEN |  |

## public_route

Evidence kind: PublicRun. Criterion: Electrothermal studies are admitted at `runmat run` and run remotely.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrothermal | tet4 | numerical_kernel | OPEN |  |
| electrothermal | tet4 | public_study | OPEN |  |
| electrothermal | tet10 | numerical_kernel | OPEN |  |
| electrothermal | tet10 | public_study | OPEN |  |
| electrothermal | edge_elements | numerical_kernel | OPEN |  |
| electrothermal | edge_elements | public_study | OPEN |  |

