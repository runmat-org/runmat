# FEA-PUBLIC-LANGUAGE-001: Authored language surface the published studies rely on

0 PASS, 0 FAIL, 13 OPEN, 0 TAGGED. Generated; do not edit.

## located_diagnostics

Evidence kind: SemanticRealization. Criterion: `runmat check` on a `.fea` file prints `error[code]: message`, the file:line:col location, the source line with a caret underline and label, the expected and found units, trims leading indentation, suppresses cascade errors from a failed dependency, and ends with the `checked ...: N error(s), M warning(s)` summary; `--json` carries the same spans.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## default_load_case

Evidence kind: SemanticRealization. Criterion: A model with loads and no `load_case` applies them in one default case and `fea check` accepts it; a model with several loads and an explicit case set is unchanged.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## unit_prefixes

Evidence kind: SemanticRealization. Criterion: Unit prefixes kW, MW, kA, kV, mA, MN, mN, kHz and the other SI prefixes parse with correct scale and dimension, and `degC` is accepted with the documented offset rule.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## check_matches_run

Evidence kind: RejectionInvariant. Criterion: `fea check` validates procedure members, state references and name selectors so that a study it accepts cannot fail at run for those reasons.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## output_expression_forms

Evidence kind: SemanticRealization. Criterion: Output forms the pages show check with dimensions, lower and run: `pressure_drop(analysis, from =, to =)`, `transmission_loss(analysis, incident =, transmitted =)`, `sound_power(analysis, through =)`, `interpolate(output, at = [x, y, z])`, `mean(field, over = time)`, `max(field, over = region)`, field names such as `field.face.wall_shear_stress`, and arithmetic between outputs of different analyse...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## input_forms

Evidence kind: SemanticRealization. Criterion: Typed input forms the pages show are admitted: `input film : heat_transfer_coefficient`, frequency tables and generated one-third-octave center sets, `samples = file "...csv"` with `interpolation = periodic_spline`, impedance tables over frequency with a `delany_bazley` model, and `normal_velocity` read from a retained state of another analysis.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

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

