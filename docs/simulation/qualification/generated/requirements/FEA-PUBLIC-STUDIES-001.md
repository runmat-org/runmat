# FEA-PUBLIC-STUDIES-001: Published studies run from the public command

0 PASS, 0 PARTIAL, 1 FAIL, 68 OPEN, 0 TAGGED. Generated; do not edit.

## docs_examples_run

Evidence kind: PublicRun. Criterion: Every file in docs/simulation/examples and docs/fea/examples completes with `runmat run` on the release build; the generated catalog records pass and wall time for each.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | FAIL | `runmat[test:check_cli]::structural_static_step_documentation_example_runs_and_publishes_fields` (crates/runmat-cli/tests/check_cli.rs:275) - The structural static STEP documentation example runs from the CLI and publishes fields; the other examples are separate cases.<br>`runmat[test:check_cli]::response_spectrum_documentation_example_runs_its_two_step_study` (crates/runmat-cli/tests/check_cli.rs:424) - The response-spectrum documentation example runs its two-step study.<br>`runmat[test:check_cli]::random_response_documentation_example_exposes_paged_statistical_fields` (crates/runmat-cli/tests/check_cli.rs:528) - The random-response documentation example runs and exposes paged statistical fields.<br>`runmat[test:check_cli]::acoustic_cavity_example_runs_its_certified_frequency_sweep` (crates/runmat-cli/tests/check_cli.rs:631) - The acoustic cavity harmonic example runs its frequency sweep to a completed, numerically acceptable run. |

## each_family_runs_local_and_remote

Evidence kind: PublicRun. Criterion: For each physics family named on the site, its worked example runs locally and with `--cluster` and publishes the same fields.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| modal_structural | canonical_tetrahedra | public_study | OPEN |  |
| modal_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | remote_execution | OPEN |  |
| structural_stability | canonical_tetrahedra | public_study | OPEN |  |
| structural_stability | canonical_tetrahedra | remote_execution | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | public_study | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | remote_execution | OPEN |  |
| thermal_steady | canonical_tetrahedra | public_study | OPEN |  |
| thermal_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| thermal_transient | canonical_tetrahedra | public_study | OPEN |  |
| thermal_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| electrostatic | canonical_tetrahedra | public_study | OPEN |  |
| electrostatic | canonical_tetrahedra | remote_execution | OPEN |  |
| electrical_conduction | canonical_tetrahedra | public_study | OPEN |  |
| electrical_conduction | canonical_tetrahedra | remote_execution | OPEN |  |
| magnetostatic | canonical_tetrahedra | public_study | OPEN |  |
| magnetostatic | canonical_tetrahedra | remote_execution | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | public_study | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | remote_execution | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | public_study | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | public_study | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | remote_execution | OPEN |  |
| thermoelastic | canonical_tetrahedra | public_study | OPEN |  |
| thermoelastic | canonical_tetrahedra | remote_execution | OPEN |  |
| electrothermal | canonical_tetrahedra | public_study | OPEN |  |
| electrothermal | canonical_tetrahedra | remote_execution | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | remote_execution | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | public_study | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_modal | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_modal | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_transient | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | remote_execution | OPEN |  |

## website_hero_studies_run

Evidence kind: PublicRun. Criterion: Each hero study in oct-13-2026/website/physics/studies (structural cantilever and fork, heat sink, induction hardening, busbar, current sensor, electrode, muffler, cabin modes, speaker, room, cold plate and its sweep, enclosure, carotid, static mixer, journal bearing, filter) runs from its full `.fea` file, not only its `.today.fea` variant, and publishes every output its page quotes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## chained_and_coupled_studies_run

Evidence kind: PublicRun. Criterion: Thermal into preload into buckling or prestressed dynamics, electrothermal, conjugate heat transfer and fluid-structure studies are admitted and run from the public command with converged coupling and interface residuals published.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | canonical_tetrahedra | public_study | OPEN |  |
| thermoelastic | canonical_tetrahedra | remote_execution | OPEN |  |
| electrothermal | canonical_tetrahedra | public_study | OPEN |  |
| electrothermal | canonical_tetrahedra | remote_execution | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | remote_execution | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | public_study | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | remote_execution | OPEN |  |

## transient_structural_runs

Evidence kind: PublicRun. Criterion: Direct and modal transient structural studies, including a study driven by a generated load, run on Tet4 and Tet10 without a transient block bound failure, a stack overflow or a codec limit at 1,001 or more signal samples.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet4 | remote_execution | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | remote_execution | OPEN |  |

## buckling_and_random_at_practical_size

Evidence kind: PublicRun. Criterion: Buckling and random-response studies run on the 4,655-node cantilever within the default publication budget, producing the load factor and RMS response the page quotes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| structural_stability | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |

## published_conduction_energy_gate

Evidence kind: PublicRun. Criterion: The published steady conduction example passes its energy gate and publishes heat flux consistent with k times the temperature difference over the length.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |

## parameter_sweeps_with_assessments

Evidence kind: PublicRun. Criterion: A study sweeps a typed input (convection film coefficient from still air to forced air; cold-plate mass flow) and evaluates an assessment per case (peak temperature below 85 C; lowest pump power meeting the limit), publishing per-case verdicts in authored order.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | canonical_tetrahedra | public_study | OPEN |  |
| thermal_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | remote_execution | OPEN |  |

## bracket_static_study

Evidence kind: PublicRun. Criterion: The demo bracket STEP with steel, 2 kN on the top face and both mounting holes fixed meshes and solves from the public command and publishes von Mises stress, peak stress and peak displacement.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet10 | public_study | OPEN |  |

