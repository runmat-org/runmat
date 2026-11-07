# FEA-THERMAL-PUBLIC-001: Thermal studies the pages run

0 PASS, 3 PARTIAL, 0 FAIL, 21 OPEN, 0 TAGGED. Generated; do not edit.

## film_coefficient_input

Evidence kind: PhysicalBalance. Criterion: A convection film coefficient is a typed input quantity (`heat_transfer_coefficient`), usable as a sweep parameter and on named faces with ambient temperature, and heat flux into and out of the part balances.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-language[test:study_realization]::film_coefficient_and_emissivity_inputs_realize_per_bound_case` (crates/runmat-analysis/language/tests/study_realization.rs:297) - Film-coefficient and emissivity inputs realize per bound sweep case at the language layer; no solve is run.<br>`runmat-analysis-language[test:study_realization]::film_coefficient_input_rejects_a_mismatched_quantity` (crates/runmat-analysis/language/tests/study_realization.rs:422) - A film-coefficient input rejects a mismatched quantity. |
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-language[test:study_realization]::film_coefficient_and_emissivity_inputs_realize_per_bound_case` (crates/runmat-analysis/language/tests/study_realization.rs:297) - Film-coefficient and emissivity inputs realize per bound sweep case at the language layer; no solve is run.<br>`runmat-analysis-language[test:study_realization]::film_coefficient_input_rejects_a_mismatched_quantity` (crates/runmat-analysis/language/tests/study_realization.rs:422) - A film-coefficient input rejects a mismatched quantity. |
| thermal_steady | tet10 | public_study | OPEN |  |

## radiation_to_ambient

Evidence kind: IndependentReference. Criterion: Gray surface-to-ambient radiation (emissivity) with convection on the same faces runs and closes its energy balance; a radiating plate matches sigma epsilon A (T^4 - T_inf^4).

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::radiation::exact_steady_slab_radiation_residual_tangent_and_prescribed_power_match_continuum` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/radiation.rs:37) - A Tet4 slab with a gray surface matches the Stefan-Boltzmann law with emissivity 0.8 at 350 K surface against a 300 K ambient; convection on the same face and t... |
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | numerical_kernel | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | numerical_kernel | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | numerical_kernel | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## heat_sink_balance

Evidence kind: PhysicalBalance. Criterion: The KiCad heat sink (aluminum, about 10 W on the mounting face, 35 W/m2/K convection to 25 C) solves and heater power in equals convection out within the accuracy target, with the peak temperature rise reported.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | OPEN |  |
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | numerical_kernel | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |

## cooling_requirement_sweep

Evidence kind: PublicRun. Criterion: A sweep of film coefficient from still air to a fan evaluates peak temperature against 85 C per case and reports the minimum cooling that passes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | OPEN |  |
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | numerical_kernel | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |

## power_cycle_transient

Evidence kind: IndependentReference. Criterion: A transient power cycle runs and matches the lumped-capacitance reference for a high-conductivity body and the Fourier series for a slab.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_transient | tet4 | numerical_kernel | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | numerical_kernel | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

