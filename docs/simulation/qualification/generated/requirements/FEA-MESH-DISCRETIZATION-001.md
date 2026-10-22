# FEA-MESH-DISCRETIZATION-001: Valid tetrahedral meshes on exact geometry

0 PASS, 0 FAIL, 27 OPEN, 0 TAGGED. Generated; do not edit.

## curve_discretization

Evidence kind: IndependentReference. Criterion: Curve discretization respects exact edge and pcurve domains, tiny edges, periodic curves and seams on real parts, within the declared chordal deviation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## surface_discretization

Evidence kind: ResourceAdmission. Criterion: Constructive trimmed-surface triangulation handles seams, poles, many trims and dense faces within the memory budget, and a dense surface of at least 90,000 nodes and 180,000 triangles is retained and published within cache-entry and codec limits.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## volume_conformity

Evidence kind: IndependentReference. Criterion: Three-dimensional constrained Delaunay recovers segments and facets, carves exterior, nested voids and regions, and preserves conformal interfaces and contact pairs; the exact boundary is matched to the declared tolerance, with a real-part corpus beside the fixtures.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## volume_quality_convergence

Evidence kind: ResourceAdmission. Criterion: Volume quality refinement terminates within the authored work and time budget on the bracket, heat sink and tuning fork, ending with every element at or above the declared quality floor, and refinement yields to coupled repair fairly with stale rejections invalidated and true global exhaustion detected.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## tet10_elevation

Evidence kind: IndependentReference. Criterion: Tet10 elevation yields valid curved elements without near-slivers: positive curved Jacobian everywhere and a scaled-Jacobian floor stated per part, including thin airfoil sections and the tuning fork.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## sizing_controls

Evidence kind: SemanticRealization. Criterion: Global, regional, face, edge and feature-based size controls and graded sizing are exposed in the `.fea` mesh block and drive the one metric field; a graded heat-sink surface and a channel-resolving cold-plate mesh are produced from authored controls without manual partitioning.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## solver_relevant_acceptance

Evidence kind: SpatialConvergence. Criterion: Mesh acceptance includes solver patch and convergence relevance on real parts: a recorded refinement series shows the target quantity converging, so coarse Tet4 meshes are flagged where stiffness is not credible.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## mesh_evidence_artifact

Evidence kind: PublicationIntegrity. Criterion: Every mesh ships an evidence artifact with per-stage counters, invariants, quality distributions (minimum, mean, p95, p99, maximum of chordal deviation, normal deviation, metric edge length, radius-edge ratio and scaled Jacobian), peak memory and the seed and digest.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## independent_mesher_differential

Evidence kind: IndependentReference. Criterion: A differential comparison against an independent mesher (OCCT incremental mesh) on a box, a curved part and a real part records volume, gradient energy and mismatches with their dispositions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

