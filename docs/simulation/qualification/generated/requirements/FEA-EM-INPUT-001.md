# FEA-EM-INPUT-001: Typed electrical and magnetic sources and rank-correct constraints

0 PASS, 6 PARTIAL, 0 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## scalar_constraints

Evidence kind: SemanticRealization. Criterion: Support prescribed potential, equipotential terminals, floating conductors, integral charge/current conditions and gauge/reference handling; diagnose rank deficiency, conflicts, unsupported combinations and incorrect source units before solving.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::periodic_faces_share_their_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:642) - A charged slab with a translation-periodic right face matches the same slab grounded on both faces to 1e-8, and the grounded face carries the full flux; termina... |
| electrostatic | tet10 | numerical_kernel | OPEN |  |
| electrical_conduction | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::missing_anchors_conflicts_unsupported_intent_and_invalid_material_fail_closed` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:382) - Tet4 rejects missing anchor, conflicting potentials, floating conductor, bad material, target and wrong harmonic intent; rejection is not implementation of floa... |
| electrical_conduction | tet10 | numerical_kernel | OPEN |  |

## magnetic_constraints

Evidence kind: SemanticRealization. Criterion: Declare gauge, symmetry, periodic and supported complex phase-shifted relations for magnetic and Maxwell operators, with source compatibility and rank evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| magnetostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::authored_material_current_ground_and_gauge_compile_into_physical_solve` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:200) - Affine Tet4/Tet10 volume-current, full grounded-boundary and gauge plan compared with same kernel; rejects missing gauge, surface-current substitution and wrong...<br>`runmat-analysis-fea[lib]::physics::magnetic::tests::prescribed_tangential_potential_drives_the_exact_uniform_field` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:384) - A prescribed tangential vector potential on one face with grounds and insulated sides drives the exact uniform flux density A = (0, a x/L, 0); gauge, periodic a... |
| magnetostatic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::authored_material_current_ground_and_gauge_compile_into_physical_solve` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:200) - Affine Tet4/Tet10 volume-current, full grounded-boundary and gauge plan compared with same kernel; rejects missing gauge, surface-current substitution and wrong... |

## excitation

Evidence kind: ExecutionEquivalence. Criterion: Shared signals and realized artifacts preserve voltage/charge/current/current-density and phasor magnitude, phase, exact region realization, parameter histories where admissible and source/power balances; no family-private history or callback authority.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| electrostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::charge_density::exact_volume_region_and_signed_superposition_preserve_charge_and_fields` (crates/runmat-analysis/fea/src/physics/electrical/tests/charge_density.rs:178) - Both orders, constant real volume-charge exact target and signed superposition preserve potential, net/absolute charge and two projection records; no generated ... |
| electrostatic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::charge_density::exact_volume_region_and_signed_superposition_preserve_charge_and_fields` (crates/runmat-analysis/fea/src/physics/electrical/tests/charge_density.rs:178) - Both orders, constant real volume-charge exact target and signed superposition preserve potential, net/absolute charge and two projection records; no generated ... |
| electrical_conduction | tet4 | numerical_kernel | OPEN |  |
| electrical_conduction | tet10 | numerical_kernel | OPEN |  |
| magnetostatic | tet4 | numerical_kernel | OPEN |  |
| magnetostatic | tet10 | numerical_kernel | OPEN |  |

