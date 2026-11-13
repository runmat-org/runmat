# FEA-THERMAL-INPUT-001: Exact thermal boundaries and shared excitation realization

0 PASS, 5 PARTIAL, 0 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## boundaries

Evidence kind: IndependentReference. Criterion: Resolve exact face targets and integrate prescribed temperature, inward/outward heat-flux convention, convection and radiation laws with documented units, normals and nonlinear domain restrictions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::numerical::exact_face_flux_and_convection_preserve_integrated_power_and_ambient_state` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/numerical.rs:227) - Exact face flux and convection preserve integrated power and the ambient state on a Tet4 fixture.<br>`runmat-analysis-fea[lib]::physics::thermal::assembly::tests::radiation::exact_steady_slab_radiation_residual_tangent_and_prescribed_power_match_continuum` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/radiation.rs:37) - Tet4 slab exact gray-surface law, finite-difference tangent and physical power; does not cover arbitrary enclosure radiation.<br>`runmat-analysis-fea[lib]::physics::thermal::assembly::tests::target_resolution::thermal_boundaries_use_resolved_entities_instead_of_authored_aliases` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/target_resolution.rs:9) - Prescribed temperature and heat flux apply to the bound resolved boundary entity, not to the entity named by a deliberately wrong alias, for both element orders...<br>`runmat-analysis-fea[lib]::pipeline::thermal::physical::interface_tests::published_state_and_element_means_read_each_side_of_the_joint` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/interface_tests.rs:225) - The manifest and published state carry second-side joint temperatures and element means read each body's side, matching the closed form at every centroid; Tet4 ...<br>`runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::radiating_bar_matches_the_gray_surface_closed_form_and_balances_power` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests.rs:186) - The validation table's radiating-bar check: 1000 W/m^2 into one end of a 1 m steel bar (k = 45 W/m/K) that radiates from the other (emissivity 0.8, 300 K surrou... |
| thermal_steady | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::target_resolution::thermal_boundaries_use_resolved_entities_instead_of_authored_aliases` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/target_resolution.rs:9) - Prescribed temperature and heat flux apply to the bound resolved boundary entity, not to the entity named by a deliberately wrong alias, for both element orders...<br>`runmat-analysis-fea[lib]::pipeline::thermal::physical::nonlinear::tests::radiating_bar_matches_the_gray_surface_closed_form_and_balances_power` (crates/runmat-analysis/fea/src/pipeline/thermal/physical/nonlinear/tests.rs:186) - The validation table's radiating-bar check: 1000 W/m^2 into one end of a 1 m steel bar (k = 45 W/m/K) that radiates from the other (emissivity 0.8, 300 K surrou... |
| thermal_transient | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::signals::time_signals_drive_prescribed_temperature_flux_and_ambient_temperature` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/signals.rs:69) - Tet4 assembly samples prescribed temperature, flux and ambient-time signals; no generated/field-artifact equivalence credit. |
| thermal_transient | tet10 | numerical_kernel | OPEN |  |

## histories

Evidence kind: SemanticRealization. Criterion: Realize time-, parameter- and field-dependent heat generation, heat flux, ambient temperature and prescribed temperature through shared signals and authenticated realized field artifacts; reject unsupported samples and incompatible fields.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## equivalence

Evidence kind: ExecutionEquivalence. Criterion: Explicit, generated and field-artifact versions of equivalent inputs yield equivalent governed source and boundary energy under the same physical and numerical controls.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | PARTIAL | `runmat-execution-runner-native[lib]::typed_fea::thermal_signals::checked_named_thermal_inputs_match_tables_and_independent_source_boundary_energy` (crates/runmat-execution-runner-native/src/typed_fea/tests/thermal_signals.rs:89) - A linear-order transient with function-generated heating and initial temperature checks and runs through the typed study path and matches tables and an independ... |
| thermal_transient | tet10 | public_study | OPEN |  |

## heat_generation

Evidence kind: PhysicalBalance. Criterion: Volumetric heat generation and total-power heat sources on named regions integrate to the declared heat rate.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::study::compilation::heat_rate::tests::total_heat_rate_is_p_over_the_meshed_region_volume_and_balances` (crates/runmat-analysis/fea/src/study/compilation/heat_rate/tests.rs:111) - a total heat rate on a solid region compiles to P / V over the region's meshed volume, and the steady energy balance generates and removes exactly P. |

