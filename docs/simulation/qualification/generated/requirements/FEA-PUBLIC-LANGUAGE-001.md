# FEA-PUBLIC-LANGUAGE-001: Authored language surface the published studies rely on

0 PASS, 5 PARTIAL, 0 FAIL, 12 OPEN, 0 TAGGED. Generated; do not edit.

## located_diagnostics

Evidence kind: SemanticRealization. Criterion: `runmat check` on a `.fea` file prints `error[code]: message`, the file:line:col location, the source line with a caret underline and label, the expected and found units, trims leading indentation, suppresses cascade errors from a failed dependency, and ends with the `checked ...: N error(s), M warning(s)` summary; `--json` carries the same spans.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat[test:check_cli]::fea_check_prints_located_diagnostics_like_m_files` (crates/runmat-cli/tests/check_cli.rs:917) - Checking a study with a force given as a traction prints error[fea.semantic.model], the file:line:col arrow, the source line, a caret underline labeled with the... |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## default_load_case

Evidence kind: SemanticRealization. Criterion: A model with loads and no `load_case` applies them in one default case and `runmat check` accepts it; a model with several loads and an explicit case set is unchanged.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat[test:check_cli]::fea_check_prints_located_diagnostics_like_m_files` (crates/runmat-cli/tests/check_cli.rs:917) - The same file with the unit corrected and no load_case checks cleanly, so the load joins the default case; multiple loads with an explicit case set are not comp... |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## unit_prefixes

Evidence kind: SemanticRealization. Criterion: Unit prefixes kW, MW, kA, kV, mA, MN, mN, kHz and the other SI prefixes parse with correct scale and dimension, and `degC` is accepted with the documented offset rule.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[lib]::semantic::units::tests::electrical_prefixes_scale_to_si` (crates/runmat-analysis/language/src/semantic/units.rs:287) - kV/mm, kA, mT and MHz parse with the correct scale and dimension; kW, MW, mA, MN and mN are not asserted here. |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## check_matches_run

Evidence kind: RejectionInvariant. Criterion: `runmat check` validates procedure members, state references and name selectors so that a study it accepts cannot fail at run for those reasons.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## output_expression_forms

Evidence kind: SemanticRealization. Criterion: Output forms the pages show check with dimensions, lower and run: `pressure_drop(analysis, from =, to =)`, `transmission_loss(analysis, incident =, transmitted =)`, `sound_power(analysis, through =)`, `interpolate(output, at = [x, y, z])`, `mean(field, over = time)`, `max(field, over = region)`, field names such as `field.face.wall_shear_stress`, and arithmetic between outputs of different analyse...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:acoustic_power_outputs]::sound_power_and_intensity_flux_lower_to_the_same_normal_flux_observation` (crates/runmat-analysis/language/tests/acoustic_power_outputs.rs:47) - sound_power and intensity flux lower to the same normal-flux observation; the other output forms are separate.<br>`runmat-analysis-language[test:acoustic_power_outputs]::flux_rejects_volume_regions_and_scalar_fields` (crates/runmat-analysis/language/tests/acoustic_power_outputs.rs:75) - Flux outputs reject volume regions and scalar fields. |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## unit_and_source_forms

Evidence kind: SemanticRealization. Criterion: `rpm` is a unit, units with fractional powers (such as Pa*s^0.45 for a power-law consistency) parse with exact dimensions, and total power on a body (`heat_generation on <body> = 2 W`) is accepted alongside W/m^3 and converts to the correct volumetric source.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## input_forms

Evidence kind: SemanticRealization. Criterion: Typed input forms the pages show are admitted: `input film : heat_transfer_coefficient`, frequency tables and generated one-third-octave center sets, `samples = file "...csv"` with `interpolation = periodic_spline`, impedance tables over frequency with a `delany_bazley` model, `normal_velocity` read from a retained state of another analysis, `species <name> { diffusivity }` with `fluid steady { tr...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:impedance_tables]::delany_bazley_and_sampled_impedance_tables_lower_into_their_boundaries` (crates/runmat-analysis/language/tests/impedance_tables.rs:58) - Delany-Bazley and sampled impedance tables lower into their acoustic boundaries; the other input forms (film coefficient, CSV samples, species, porous, lubricat...<br>`runmat-analysis-language[test:impedance_tables]::impedance_tables_reject_unknown_models_quantities_and_unsorted_samples` (crates/runmat-analysis/language/tests/impedance_tables.rs:113) - Impedance tables reject unknown models, wrong quantities and unsorted samples. |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## browser_check

Evidence kind: SemanticRealization. Criterion: `.fea` files type-check in the browser with the same diagnostics as the CLI, and the browser never runs or meshes silently.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## matlab_generated_signals

Evidence kind: PublicRun. Criterion: `import function` of a MATLAB function that generates a load over time checks, solves a transient with it and returns the response shown on the runtime page.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

