# FEA-STRUCTURAL-CONSTRAINT-001: Complete typed structural constraints and constitutive connectors

0 PASS, 8 PARTIAL, 0 FAIL, 76 OPEN, 0 TAGGED. Generated; do not edit.

## fixed_component_motion

Evidence kind: IndependentReference. Criterion: The supported structural surface provides fixed support, component-wise prescribed translation, and component-wise prescribed rotation where rotational DOFs exist.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::transient_physical::physical_tet4_prescribed_motion_keeps_absolute_constraint_kinematics` (crates/runmat-analysis/fea/src/tests/transient_physical.rs:597) - Tet4 prescribed motion keeps absolute constraint kinematics.<br>`runmat-analysis-fea[lib]::tests::transient_physical::fully_prescribed_tet4_advances_the_complete_motion_schedule` (crates/runmat-analysis/fea/src/tests/transient_physical.rs:1024) - A fully prescribed Tet4 advances the complete motion schedule. |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## normal_symmetry_rigid_coupling

Evidence kind: IndependentReference. Criterion: The supported structural surface provides normal displacement/roller support, symmetry/antisymmetry, and rigid coupling to a reference point.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::construction_tests::both_order_surface_weights_normals_and_coupling_rows_match_independent_geometry` (crates/runmat-analysis/fea/src/constraints/construction_tests.rs:25) - Both-order surface weights, normals and coupling rows match independent geometry.<br>`runmat-analysis-fea[lib]::constraints::tests::normal_displacement_and_antisymmetry_compile_distinct_normal_tangent_spaces` (crates/runmat-analysis/fea/src/constraints/tests.rs:270) - Normal displacement and antisymmetry compile distinct normal and tangent spaces on a fixture mesh (Tet4). |
| linear_structural | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::construction_tests::both_order_surface_weights_normals_and_coupling_rows_match_independent_geometry` (crates/runmat-analysis/fea/src/constraints/construction_tests.rs:25) - Both-order surface weights, normals and coupling rows match independent geometry. |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::mixed::tests::explicit_shell_symmetry_uses_the_authored_surface_normal` (crates/runmat-analysis/fea/src/constraints/mixed/tests.rs:175) - Shell symmetry uses the authored surface normal. |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## distributing_ties_equations

Evidence kind: IndependentReference. Criterion: The supported structural surface provides distributing reference-point coupling, ties over matching/supported nonmatching surfaces, and general linear multipoint equations.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::tests::reference_point_wrench_solves_through_distributing_coupling` (crates/runmat-analysis/fea/src/constraints/tests.rs:1569) - A reference-point wrench solves through distributing coupling on a fixture solid. |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::mixed::tests::explicit_shell_distributing_coupling_retains_reference_point_evidence` (crates/runmat-analysis/fea/src/constraints/mixed/tests.rs:208) - Shell distributing coupling retains reference-point evidence. |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## periodicity_releases

Evidence kind: IndependentReference. Criterion: The supported structural surface provides translational periodicity, cyclic symmetry, and beam/shell releases and supported joint relations.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## typed_connector_inventory

Evidence kind: IndependentReference. Criterion: The supported structural surface provides spring, dashpot, and elastic-foundation connectors.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::tests::explicit_connectors_assemble_physical_stiffness_and_damping_operators` (crates/runmat-analysis/fea/src/assembly/mod.rs:2571) - Explicit connectors assemble physical stiffness and damping operators on a beam fixture; only the connector kinds in that fixture are exercised.<br>`runmat-analysis-fea[lib]::assembly::tests::explicit_connector_targets_and_foundation_semantics_fail_closed` (crates/runmat-analysis/fea/src/assembly/mod.rs:2671) - Connector targets and foundation semantics fail closed. |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## connectors

Evidence kind: SemanticRealization. Criterion: Connectors are typed constitutive objects and contribute to stiffness, damping, or residual assembly. They are not represented as exact equation rows.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::connector_qualification::explicit_axial_connector_matches_the_parallel_spring_reference` (crates/runmat-analysis/fea/src/tests/connector_qualification.rs:17) - An explicit axial connector matches the parallel-spring reference. |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

