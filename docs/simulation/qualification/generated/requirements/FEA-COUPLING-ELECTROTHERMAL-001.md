# FEA-COUPLING-ELECTROTHERMAL-001: Original Joule-heat transfer and electrical/thermal iteration

0 PASS, 4 PARTIAL, 0 FAIL, 2 OPEN, 0 TAGGED. Generated; do not edit.

## heat

Evidence kind: PhysicalBalance. Criterion: Transfer original solved-field Joule density and total power with exact point/element topology, units and authenticated source/target lineage; terminal electrical power and supplied thermal power agree.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrothermal | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::electrothermal::tests::retained_conduction_drives_independent_quadratic_thermal_patch_and_power` (crates/runmat-analysis/fea/src/execution/electrothermal/tests.rs:75) - Both-order constant-conductivity same-mesh steady one-way conduction to thermal slab; 12 W terminal/supplied/boundary power, Tet10 quadratic temperature and col...<br>`runmat-analysis-fea[lib]::execution::electrothermal::tests::tet4_refinement_and_voltage_sign_gauge_preserve_independent_heat_and_power` (crates/runmat-analysis/fea/src/execution/electrothermal/tests.rs:149) - Tet4 two-mesh interior temperature refinement and Tet10 voltage sign/gauge invariance at constant conductivity; no two-way thermal iteration. |
| electrothermal | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::electrothermal::tests::retained_conduction_drives_independent_quadratic_thermal_patch_and_power` (crates/runmat-analysis/fea/src/execution/electrothermal/tests.rs:75) - Both-order constant-conductivity same-mesh steady one-way conduction to thermal slab; 12 W terminal/supplied/boundary power, Tet10 quadratic temperature and col...<br>`runmat-analysis-fea[lib]::execution::electrothermal::tests::tet4_refinement_and_voltage_sign_gauge_preserve_independent_heat_and_power` (crates/runmat-analysis/fea/src/execution/electrothermal/tests.rs:149) - Tet4 two-mesh interior temperature refinement and Tet10 voltage sign/gauge invariance at constant conductivity; no two-way thermal iteration. |

## feedback

Evidence kind: AccuracyAcceptance. Criterion: Update temperature-dependent conductivity and re-solve the actual electrical and thermal equations under documented iteration controls; retain electrical residual, thermal residual, field change and energy closure for every accepted coupled state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrothermal | tet4 | public_study | OPEN |  |
| electrothermal | tet10 | public_study | OPEN |  |

## cold

Evidence kind: ColdReplay. Criterion: Cold transfer authenticates the complete original electrical result and intended thermal consumer before use; reject foreign lineage, altered power/inventory, unsupported equations, feedback and release intent.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrothermal | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::electrothermal::tests::source::retained_source_cold_binding_authenticates_original_result_and_target` (crates/runmat-analysis/fea/src/execution/electrothermal/tests/source.rs:8) - Both-order retained source/target binding and mutated density rejection in one process; no native termination/restart or remote transfer. |
| electrothermal | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::electrothermal::tests::source::retained_source_cold_binding_authenticates_original_result_and_target` (crates/runmat-analysis/fea/src/execution/electrothermal/tests/source.rs:8) - Both-order retained source/target binding and mutated density rejection in one process; no native termination/restart or remote transfer. |

