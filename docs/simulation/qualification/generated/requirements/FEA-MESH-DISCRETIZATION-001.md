# FEA-MESH-DISCRETIZATION-001: Valid tetrahedral meshes on exact geometry

0 PASS, 6 PARTIAL, 1 FAIL, 23 OPEN, 0 TAGGED. Generated; do not edit.

## curve_discretization

Evidence kind: IndependentReference. Criterion: Curve discretization respects exact edge and pcurve domains, tiny edges, periodic curves and seams on real parts, within the declared chordal deviation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-curve[lib]::shared::tests::nonperiodic_closed_face_rejects_distinct_pcurve_endpoints` (crates/runmat-meshing/curve/src/shared/tests.rs:148) - A nonperiodic closed face with distinct pcurve endpoints is rejected; tiny edges and periodic curves are separate cases.<br>`runmat-meshing-curve[lib]::shared::tests::constructive_curve_discretization_fails_at_its_hard_node_limit` (crates/runmat-meshing/curve/src/shared/tests.rs:290) - Constructive curve discretization fails at its hard node limit. |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## surface_discretization

Evidence kind: ResourceAdmission. Criterion: Constructive trimmed-surface triangulation handles seams, poles, many trims and dense faces within the memory budget, and a dense surface of at least 90,000 nodes and 180,000 triangles is retained and published within cache-entry and codec limits.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-surface[lib]::face_pipeline_tests::publication_scale::representative_dense_face_publication_fits_default_component_memory` (crates/runmat-meshing/surface/src/face_pipeline_tests/publication_scale.rs:87) - A dense face of 90,601 nodes and 180,000 triangles has a bounded publication representation inside the 512 MiB default component memory and round-trips through ... |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## volume_conformity

Evidence kind: IndependentReference. Criterion: Three-dimensional constrained Delaunay recovers segments and facets, carves exterior, nested voids and regions, and preserves conformal interfaces and contact pairs; the exact boundary is matched to the declared tolerance, with a real-part corpus beside the fixtures.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-tetrahedron[lib]::cavity::constrained::tests::shared_face_split::shared_face_split_preserves_component_boundary_and_volume` (crates/runmat-meshing/tetrahedron/src/cavity/constrained/tests/shared_face_split.rs:8) - A shared-face split preserves component boundary and volume; segment and facet recovery and nested voids are separate tests. |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## volume_quality_convergence

Evidence kind: ResourceAdmission. Criterion: Volume quality refinement terminates within the authored work and time budget on the bracket, heat sink and tuning fork, ending with every element at or above the declared quality floor, and refinement yields to coupled repair fairly with stale rejections invalidated and true global exhaustion detected.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-tetrahedron[lib]::cdt::volume_mesh::quality_convergence::schedule::tests::failed_source_window_yields_and_rejected_repair_resumes_refinement` (crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_convergence/schedule.rs:96) - A failed source window yields to repair and a rejected repair resumes refinement; fairness on real parts is not asserted.<br>`runmat-meshing-tetrahedron[lib]::cdt::volume_mesh::quality_convergence::tests::exhausted_insertion_or_repair_pass_budget_fails_without_a_mesh` (crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_convergence/tests.rs:185) - Exhausted insertion or repair pass budgets fail without publishing a mesh; convergence on the bracket, heat sink and fork within the authored time budget is not... |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## analytic_surface_coverage

Evidence kind: IndependentReference. Criterion: Surface meshing handles every analytic surface kind including spheres, tori, cones and cylinders with poles, seams and umbilic points: a plain sphere, a hemisphere, spherical fillets and domes, and a carotid bulb and sinus mesh with the directional curvature metric defined at umbilic points.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## tet10_elevation

Evidence kind: IndependentReference. Criterion: Tet10 elevation yields valid curved elements without near-slivers: positive curved Jacobian everywhere and a scaled-Jacobian floor stated per part, including thin airfoil sections and the tuning fork.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-tetrahedron[lib]::cdt::solver_topology::order_elevation::jacobian::tests::interval_certificate_accepts_linear_mapping_and_rejects_curved_inversion` (crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/jacobian/tests.rs:90) - The Jacobian interval certificate accepts a linear mapping and rejects a curved inversion.<br>`runmat-meshing-tetrahedron[lib]::cdt::solver_topology::tests::deterministic_tet10_elevation_shares_exact_boundary_midpoints` (crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/tests.rs:297) - Tet10 elevation is deterministic and shares exact boundary midpoints; the scaled-Jacobian floor on thin or curved real parts is not asserted. |
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
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-core[lib]::contracts::canonical::artifact_tests::artifact_and_evidence_round_trip_with_complete_canonical_topology` (crates/runmat-meshing/core/src/contracts/canonical/artifact_tests.rs:294) - A mesh artifact and its evidence round-trip with complete canonical topology; per-stage counters and quality distributions are asserted in sibling tests. |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## independent_mesher_differential

Evidence kind: IndependentReference. Criterion: A differential comparison against an independent mesher (OCCT incremental mesh) on a box, a curved part and a real part records volume, gradient energy and mismatches with their dispositions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | FAIL | `runmat-execution-runner-native[test:occt_exact_meshing]::differential::runmat_and_independent_meshers_agree_on_box_invariants` (crates/runmat-execution-runner-native/tests/occt_exact_meshing/differential.rs:19) - RunMat and the OCCT incremental mesher agree on box invariants (volume and gradient energy); box only, no curved or real part. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

