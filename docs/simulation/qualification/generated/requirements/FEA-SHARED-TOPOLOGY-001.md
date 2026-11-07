# FEA-SHARED-TOPOLOGY-001: Canonical analysis topology and deterministic projections

0 PASS, 2 PARTIAL, 0 FAIL, 30 OPEN, 0 TAGGED. Generated; do not edit.

## canonical_input

Evidence kind: IndependentReference. Criterion: The analysis projection is the central prerequisite for every family. It consumes a validated canonical `SolverMeshArtifact` and produces immutable, indexed analysis data.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## projection_inventory

Evidence kind: IndependentReference. Criterion: Dense node-index mapping from canonical node IDs. Element index, canonical identity, order, node connectivity, region ownership, orientation, Jacobian data, volume, centroid, and characteristic length. Quadrature points, weights, shape functions, and gradients for each supported element/order. Boundary-face index, canonical identity, adjacent element, local face, orientation, outward normal, area,...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::topology::tests::canonical_tetrahedron_projection_has_indexed_oriented_geometry` (crates/runmat-analysis/fea/src/topology/tests.rs:79) - Single affine Tet10 tetrahedron checks node/element/face inventory, positive volume, integrated shapes and world gradient, outward normals and repeatable projec... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## projection_invariants

Evidence kind: IndependentReference. Criterion: Every element has exactly one supported physical-region/material resolution or a typed failure. Every applied boundary condition and load resolves to an allowed topological dimension. Boundary orientation agrees with adjacent element orientation. Interface sides cannot be silently empty, multiply owned, or self-paired. Conservation-oriented topology has reciprocal owner/neighbour relationships. Pr...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::topology::tests::projection_rejects_a_substituted_mesh_digest` (crates/runmat-analysis/fea/src/topology/tests.rs:166) - Tet4 node-coordinate substitution without digest resealing rejects projection; no exhaustive provenance or interface invalid-case corpus. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## projection_reference_corpus

Evidence kind: IndependentReference. Criterion: Tet4 and Tet10 geometry and quadrature invariants. Boundary normals and integrated areas on analytic primitives. Multi-region and multi-material ownership. Pressure resultant and moment recovery. Body-force volume integration. Interface side resolution and invalid ambiguity cases. Finite-volume divergence theorem checks. Stable projection identity across serialization and execution shapes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

