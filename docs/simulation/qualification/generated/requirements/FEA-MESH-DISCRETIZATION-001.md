# FEA-MESH-DISCRETIZATION-001: Valid tetrahedral meshes on exact geometry

0 PASS, 7 PARTIAL, 0 FAIL, 23 OPEN, 0 TAGGED. Generated; do not edit.

## curve_discretization

Evidence kind: IndependentReference. Criterion: Curve discretization respects exact edge and pcurve domains, tiny edges, periodic curves and seams on real parts, within the declared chordal deviation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-curve[lib]::shared::tests::nonperiodic_closed_face_rejects_distinct_pcurve_endpoints` (crates/runmat-meshing/curve/src/shared/tests.rs:162) - A nonperiodic closed face with distinct pcurve endpoints is rejected; tiny edges and periodic curves are separate cases.<br>`runmat-meshing-curve[lib]::shared::tests::constructive_curve_discretization_fails_at_its_hard_node_limit` (crates/runmat-meshing/curve/src/shared/tests.rs:319) - Constructive curve discretization fails at its hard node limit. |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## surface_discretization

Evidence kind: ResourceAdmission. Criterion: Constructive trimmed-surface triangulation handles seams, poles, many trims and dense faces within the memory budget, and a dense surface of at least 90,000 nodes and 180,000 triangles is retained and published within cache-entry and codec limits.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-surface[lib]::exact_cdt::triangulation::tests::a_point_just_inside_a_hull_edge_keeps_the_hull_edge` (crates/runmat-meshing/surface/src/exact_cdt/triangulation/tests.rs:158) - Regression: a point just inside a hull edge keeps the hull edge in the exact face triangulation; the dense-surface and many-trim cases are separate tests.<br>`runmat-meshing-surface[lib]::exact_metric::resolved::periodic_bounds_tests::roundoff_outside_partial_period_bounds_lands_on_the_bound` (crates/runmat-meshing/surface/src/exact_metric/resolved.rs:463) - periodic chart coordinates within roundoff of partial-period trimmed bounds (hv-electrode rounded quarter torus)<br>`runmat-meshing-surface[lib]::face_pipeline_tests::publication_scale::representative_dense_face_publication_fits_default_component_memory` (crates/runmat-meshing/surface/src/face_pipeline_tests/publication_scale.rs:87) - A dense face of 90,601 nodes and 180,000 triangles has a bounded publication representation inside the 512 MiB default component memory and round-trips through ... |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## volume_conformity

Evidence kind: IndependentReference. Criterion: Three-dimensional constrained Delaunay recovers segments and facets, carves exterior, nested voids and regions, and preserves conformal interfaces and contact pairs; the exact boundary is matched to the declared tolerance, with a real-part corpus beside the fixtures.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-tetrahedron[lib]::cavity::constrained::tests::shared_face_split::shared_face_split_preserves_component_boundary_and_volume` (crates/runmat-meshing/tetrahedron/src/cavity/constrained/tests/shared_face_split.rs:8) - A shared-face split preserves component boundary and volume; segment and facet recovery and nested voids are separate tests.<br>`runmat-meshing-tetrahedron[lib]::cdt::point_set::tests::a_point_on_a_flat_hull_cap_replaces_the_cells_whose_circumcircle_holds_it` (crates/runmat-meshing/tetrahedron/src/cdt/point_set/tests.rs:310) - Regression: a point on a flat hull cap replaces exactly the cells whose circumcircle contains it in the incremental point-set construction; facet recovery and c...<br>`runmat-meshing-tetrahedron[lib]::cdt::point_set::tests::box_surface_lattice_stays_delaunay_in_any_insertion_order` (crates/runmat-meshing/tetrahedron/src/cdt/point_set/tests.rs:357) - Regression (EV battery module, 22 shared-interface boxes): boundary nodes of axis-aligned boxes land on hull edges between collinear nodes; whatever cell the wa...<br>`runmat-meshing-tetrahedron[lib]::cdt::volume_mesh::components::tests::regions_joined_by_an_interface_facet_form_one_component` (crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/components/tests.rs:230) - Conformal multi-region interfaces mesh in one component so both sides share every interface Steiner point (fsi-compliant-tube bore). |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## volume_quality_convergence

Evidence kind: ResourceAdmission. Criterion: Volume quality refinement terminates within the authored work and time budget on the bracket, heat sink and tuning fork, ending with every element at or above the declared quality floor, and refinement yields to coupled repair fairly with stale rejections invalidated and true global exhaustion detected.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-core[lib]::quality::boundary_face::tests::shared_face_rule_rejects_folded_and_edge_on_faces` (crates/runmat-meshing/core/src/quality/boundary_face.rs:382) - The shared boundary-face rule admits flat and gently curved six-node faces and rejects folded faces, faces nearly edge-on to their corner plane, and malformed f...<br>`runmat-meshing-tetrahedron[lib]::cdt::solver_topology::order_elevation::geometry::evaluation::tests::periodic_surface_midpoint_crosses_the_seam_without_a_second_seam_image` (crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/geometry/evaluation/tests.rs:103) - A seam node carrying only its u = 0 image pairs with a node near u = 2 pi across the seam; the coax Tet10 study exercises the whole elevation.<br>`runmat-meshing-tetrahedron[lib]::cdt::solver_topology::order_elevation::relaxation_tests::relaxation_keeps_boundary_midsides_within_the_chordal_deviation` (crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/mod.rs:628) - An uncertifiable Tet10 relaxes interior midsides to chords and boundary midsides only to the chordal deviation from their exact image; real parts are exercised ...<br>`runmat-meshing-tetrahedron[lib]::cdt::solver_topology::tests::tet10_curve_midpoints_do_not_trim_classify_their_own_pcurve_images` (crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/tests.rs:404) - Curve midpoints whose pcurve images a trim classifier reports outside (the pin fitting v1 STEP) are admitted by curve/surface agreement; concave-trim surface fa...<br>`runmat-meshing-tetrahedron[lib]::cdt::volume_mesh::quality_convergence::schedule::tests::failed_source_window_yields_and_rejected_repair_resumes_refinement` (crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_convergence/schedule.rs:101) - A failed source window yields to repair and a rejected repair resumes refinement; fairness on real parts is not asserted.<br>`runmat-meshing-tetrahedron[lib]::cdt::volume_mesh::quality_convergence::tests::exhausted_insertion_or_repair_pass_budget_fails_without_a_mesh` (crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_convergence/tests.rs:187) - Exhausted insertion or repair pass budgets fail without publishing a mesh; convergence on the bracket, heat sink and fork within the authored time budget is not...<br>`runmat-meshing-tetrahedron[lib]::cdt::volume_refinement::tests::size_refinement_splits_the_longest_edge_with_an_interior_endpoint` (crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/tests.rs:449) - The size-defect edge candidate splits only an edge with an interior endpoint and lies strictly inside its source; whole-part runtime is measured on the cantilev... |
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
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-core[lib]::quality::tet10_element::tests::bracket_element_377_fails_the_shared_rule_and_its_straightened_repair_passes` (crates/runmat-meshing/core/src/quality/tet10_element.rs:938) - the bracket element that passed the mesher's positivity certificate but failed analysis stiffness quadrature is rejected by the one rule both now apply, and its...<br>`runmat-meshing-tetrahedron[lib]::cdt::solver_topology::order_elevation::jacobian::tests::interval_certificate_accepts_linear_mapping_and_rejects_curved_inversion` (crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/jacobian/tests.rs:93) - The Jacobian interval certificate accepts a linear mapping and rejects a curved inversion.<br>`runmat-meshing-tetrahedron[lib]::cdt::solver_topology::tests::deterministic_tet10_elevation_shares_exact_boundary_midpoints` (crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/tests.rs:299) - Tet10 elevation is deterministic and shares exact boundary midpoints; the scaled-Jacobian floor on thin or curved real parts is not asserted. |
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
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-core[lib]::contracts::canonical::artifact_tests::artifact_and_evidence_round_trip_with_complete_canonical_topology` (crates/runmat-meshing/core/src/contracts/canonical/artifact_tests.rs:302) - A mesh artifact and its evidence round-trip with complete canonical topology; per-stage counters and quality distributions are asserted in sibling tests. |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## independent_mesher_differential

Evidence kind: IndependentReference. Criterion: A differential comparison against an independent mesher (OCCT incremental mesh) on a box, a curved part and a real part records volume, gradient energy and mismatches with their dispositions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | PARTIAL | `runmat-execution-runner-native[test:occt_exact_meshing]::differential::runmat_and_independent_meshers_agree_on_box_invariants` (crates/runmat-execution-runner-native/tests/occt_exact_meshing/differential.rs:20) - RunMat and the OCCT incremental mesher agree on box invariants (volume and gradient energy); box only, no curved or real part. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

