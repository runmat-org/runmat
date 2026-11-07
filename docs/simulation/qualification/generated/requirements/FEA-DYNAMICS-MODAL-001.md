# FEA-DYNAMICS-MODAL-001: Physical modal extraction, eigenspaces and reusable participation state

0 PASS, 3 PARTIAL, 0 FAIL, 53 OPEN, 0 TAGGED. Generated; do not edit.

## physical_mass_base_state

Evidence kind: IndependentReference. Criterion: Use the physical structural stiffness and mass operators. Treat unstressed modal extraction and prestressed modal extraction as distinct typed procedures; the latter consumes a validated converged base state through the sequence contract. Support documented consistent/lumped mass selection.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | beam | numerical_kernel | OPEN |  |
| modal_structural | shell | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::pipeline::modal::rod_qualification::solid_rod_modal_frequency_and_retained_response_have_independent_evidence` (crates/runmat-analysis/fea/src/pipeline/modal/rod_qualification.rs:315) - Axial affine Tet4 slab rod frequency refinement and modal truncation against direct harmonic response. Does not cover beam/shell modes, repeated eigenspaces, ev... |
| modal_structural | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::pipeline::modal::rod_qualification::solid_rod_tet10_modal_frequency_and_retained_response_have_independent_evidence` (crates/runmat-analysis/fea/src/pipeline/modal/rod_qualification.rs:330) - Axial affine Tet10 slab rod frequency refinement and modal truncation against direct harmonic response; beam and shell modes, repeated eigenspaces and remote en... |
| modal_structural | beam | public_study | OPEN |  |
| modal_structural | shell | public_study | OPEN |  |
| modal_structural | canonical_mixed_topology | native_process | OPEN |  |
| modal_structural | canonical_mixed_topology | durable | OPEN |  |
| modal_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| modal_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| modal_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| modal_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## eigenpairs_eigenspaces

Evidence kind: IndependentReference. Criterion: Handle constraints, rigid modes, repeated and clustered eigenvalues. Mass-normalize modes and report residual and orthogonality measures. Make mode ordering deterministic without pretending that an arbitrary basis inside a repeated eigenspace is unique.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | beam | numerical_kernel | OPEN |  |
| modal_structural | shell | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |
| modal_structural | beam | public_study | OPEN |  |
| modal_structural | shell | public_study | OPEN |  |
| modal_structural | canonical_mixed_topology | native_process | OPEN |  |
| modal_structural | canonical_mixed_topology | durable | OPEN |  |
| modal_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| modal_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| modal_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| modal_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## participation_partitioning

Evidence kind: IndependentReference. Criterion: Publish participation factors, effective modal mass by direction, and cumulative captured mass for later forced-response procedures. Partition matrix-vector work only after serial eigensolver evidence is complete.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | beam | numerical_kernel | OPEN |  |
| modal_structural | shell | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |
| modal_structural | beam | public_study | OPEN |  |
| modal_structural | shell | public_study | OPEN |  |
| modal_structural | canonical_mixed_topology | native_process | OPEN |  |
| modal_structural | canonical_mixed_topology | durable | OPEN |  |
| modal_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| modal_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| modal_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| modal_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## references

Evidence kind: IndependentReference. Criterion: Evidence includes axial bar and beam modes, three-dimensional solids, repeated modes, mass orthogonality, residuals, and mesh convergence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | beam | numerical_kernel | OPEN |  |
| modal_structural | shell | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::pipeline::modal::rod_qualification::solid_rod_tet10_modal_frequency_and_retained_response_have_independent_evidence` (crates/runmat-analysis/fea/src/pipeline/modal/rod_qualification.rs:330) - Axial rod modal frequency against the continuum reference with retained-response truncation checked against direct harmonic response; no beam modes, repeated mo... |
| modal_structural | beam | public_study | OPEN |  |
| modal_structural | shell | public_study | OPEN |  |
| modal_structural | canonical_mixed_topology | native_process | OPEN |  |
| modal_structural | canonical_mixed_topology | durable | OPEN |  |
| modal_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| modal_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| modal_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| modal_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

