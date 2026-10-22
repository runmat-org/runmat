# FEA-STRUCTURAL-BEAM-001: Physical three-dimensional beams and explicit mixed-model coupling

0 PASS, 1 FAIL, 27 OPEN, 0 TAGGED. Generated; do not edit.

## physical_beam_mixed_dofs

Evidence kind: IndependentReference. Criterion: Retain the real three-dimensional beam formulation and verify transformation, shear/bending assumptions, mass, and section recovery. Define explicit node/constraint coupling between beams, shells, and solids. Assemble mixed models into one global degree-of-freedom system.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::assembly::elements::beam::tests::beam_transformation_is_block_orthonormal` (crates/runmat-analysis/fea/src/assembly/elements/beam.rs:331) - One oblique beam frame transformation orthonormality only; no mixed-model or global beam benchmark claim. |
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

