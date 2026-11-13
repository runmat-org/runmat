# FEA-COUPLING-THERMOELASTIC-001: Solved-temperature thermoelastic transitions and closure

0 PASS, 2 PARTIAL, 0 FAIL, 4 OPEN, 0 TAGGED. Generated; do not edit.

## strain

Evidence kind: IndependentReference. Criterion: Evaluate temperature-dependent properties and thermal strain at integration points using the declared reference temperature and original thermal field; retain transfer error and mechanical equilibrium with thermal balance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::post::solid::thermoelastic_tests::solved_linear_temperature_recovers_pointwise_stress_and_integrated_energy` (crates/runmat-analysis/fea/src/post/solid/thermoelastic_tests.rs:196) - Both-order single-tetrahedron recovery from an supplied linear temperature array: fixed point stresses/integrated energy and Tet10 compatible free displacement....<br>`runmat-analysis-fea[lib]::post::solid::thermoelastic_tests::thermal_eigenstrain_recovery_is_stress_free_for_free_expansion_and_compressive_when_fixed` (crates/runmat-analysis/fea/src/post/solid/thermoelastic_tests.rs:307) - Both-order single-element uniform eigenstrain free/fixed recovery; not coupled equilibrium, temperature-dependent property iteration or mapped transition. |
| thermoelastic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::post::solid::thermoelastic_tests::solved_linear_temperature_recovers_pointwise_stress_and_integrated_energy` (crates/runmat-analysis/fea/src/post/solid/thermoelastic_tests.rs:196) - Both-order single-tetrahedron recovery from an supplied linear temperature array: fixed point stresses/integrated energy and Tet10 compatible free displacement....<br>`runmat-analysis-fea[lib]::post::solid::thermoelastic_tests::thermal_eigenstrain_recovery_is_stress_free_for_free_expansion_and_compressive_when_fixed` (crates/runmat-analysis/fea/src/post/solid/thermoelastic_tests.rs:307) - Both-order single-element uniform eigenstrain free/fixed recovery; not coupled equilibrium, temperature-dependent property iteration or mapped transition. |

## transition

Evidence kind: OriginalStateComparison. Criterion: Thermal preload may feed only supported prestressed modal, harmonic and buckling branches after complete physical state, constraints, tangent, transition and mandatory accuracy evidence pass. An initial guess or incomplete checkpoint is not a base state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | public_study | OPEN |  |
| thermoelastic | tet10 | public_study | OPEN |  |

## two_way

Evidence kind: PhysicalBalance. Criterion: Name one-way and two-way policies explicitly. Implement two-way deformation-dependent thermal exchange only inside its supported envelope and iterate to field and energy closure; reject unsupported feedback.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | public_study | OPEN |  |
| thermoelastic | tet10 | public_study | OPEN |  |

