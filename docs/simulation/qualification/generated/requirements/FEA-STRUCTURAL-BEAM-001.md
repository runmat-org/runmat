# FEA-STRUCTURAL-BEAM-001: Physical three-dimensional beams and explicit mixed-model coupling

0 PASS, 2 PARTIAL, 0 FAIL, 26 OPEN, 0 TAGGED. Generated; do not edit.

## physical_beam_mixed_dofs

Evidence kind: IndependentReference. Criterion: Retain the real three-dimensional beam formulation and verify transformation, shear/bending assumptions, mass, and section recovery. Define explicit node/constraint coupling between beams, shells, and solids. Assemble mixed models into one global degree-of-freedom system.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::elements::beam::tests::beam_transformation_is_block_orthonormal` (crates/runmat-analysis/fea/src/assembly/elements/beam.rs:438) - One oblique beam frame transformation orthonormality only; no mixed-model or global beam benchmark claim.<br>`runmat-analysis-fea[lib]::assembly::tests::mixed_beam_shell_connectivity_assembles_one_shared_dof_system` (crates/runmat-analysis/fea/src/assembly/mod.rs:3936) - Mixed beam and shell connectivity assembles one shared DOF system.<br>`runmat-analysis-fea[lib]::post::fields::tests::scoped_two_element_beam_recovery_preserves_physical_assembly_fields` (crates/runmat-analysis/fea/src/post/fields/tests.rs:267) - Two-element beam recovery preserves the physical assembly fields.<br>`runmat-analysis-fea[lib]::tests::mixed_beam_shell_model_solves_and_recovers_each_formulation` (crates/runmat-analysis/fea/src/tests.rs:816) - A mixed beam and shell model solves and recovers each formulation. |
| linear_structural | shell | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::tests::mixed_beam_shell_connectivity_assembles_one_shared_dof_system` (crates/runmat-analysis/fea/src/assembly/mod.rs:3936) - Mixed beam and shell connectivity assembles one shared DOF system.<br>`runmat-analysis-fea[lib]::tests::mixed_beam_shell_model_solves_and_recovers_each_formulation` (crates/runmat-analysis/fea/src/tests.rs:816) - A mixed beam and shell model solves and recovers each formulation. |
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

## coupling_references

Evidence kind: IndependentReference. Criterion: Reject incompatible or disconnected couplings. Add frame, cantilever, torsion, and mixed solid-beam benchmarks.

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

