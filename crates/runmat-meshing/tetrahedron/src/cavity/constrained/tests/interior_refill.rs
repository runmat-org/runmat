use super::*;

#[test]
fn candidate_orphan_interior_face_counts_report_global_orphans() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let lower = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let upper = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 2, 1, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };

    assert_eq!(
        candidate_orphan_interior_face_counts(&cavity, &[lower.clone()]),
        (1, 0)
    );
    assert_eq!(
        candidate_orphan_interior_face_counts(&cavity, &[lower, upper]),
        (0, 2)
    );
}

#[test]
fn centroid_interior_refill_candidate_recovers_split_boundary_tetrahedron_cavity() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    validate_constrained_cavity(&cavity).expect("split boundary fixture should be valid");
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let refill = centroid_interior_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        refill_options(),
    )
    .expect("centroid interior refill should evaluate")
    .expect("centroid interior refill should recover the split boundary cavity");

    assert_eq!(refill.inserted_nodes.len(), 1);
    assert_eq!(refill.inserted_nodes[0].node_id, 8);
    assert_eq!(refill.tetrahedra.len(), cavity.boundary_faces.len());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("centroid interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        refill_options().volume_relative_tolerance,
    )
    .expect("centroid interior refill should preserve volume");
}

#[test]
fn interior_star_quality_diagnostic_bins_candidate_quality() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let candidates = vec![
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [3.0, 3.0, 3.0],
        },
    ];

    let diagnostic = diagnostic_interior_star_quality(
        &cavity,
        &nodes,
        &candidates,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.01,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("interior star diagnostic should evaluate");

    assert_eq!(diagnostic.candidate_count, 1);
    assert_eq!(diagnostic.pass_count, 1);
    assert!(diagnostic.max_min_scaled_jacobian >= 0.01);
    assert!(!diagnostic.min_scaled_jacobian_bins.is_empty());
    assert_eq!(
        diagnostic.rejected_by_reason,
        BTreeMap::from([("interior_point_outside_cavity", 1)])
    );
}

#[test]
fn two_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
    ];
    let options = refill_options();

    let refill = two_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("two-interior refill should evaluate")
    .expect("two-interior refill should recover the cavity");

    assert_eq!(refill.inserted_nodes, interior_candidates);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("two-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("two-interior refill should preserve volume");
}

#[test]
fn multi_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
        ConstrainedCavityNode {
            node_id: 12,
            coordinates_m: [0.50, 0.25, 0.0],
        },
    ];
    let options = refill_options();

    let refill = multi_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("multi-interior refill should evaluate")
    .expect("multi-interior refill should recover the cavity");

    assert!(!refill.inserted_nodes.is_empty());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("multi-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("multi-interior refill should preserve volume");
}

#[test]
fn multi_interior_exact_cover_failure_reports_boundary_face_without_addable_candidate() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let options = refill_options();
    let lower_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let lower_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], lower_points, options)
            .expect("fixture tetrahedron should pass quality gates");

    assert_eq!(
        multi_interior_exact_cover_failure_reason(&cavity, &[lower_tetrahedron], options),
        "multi_interior_exact_cover_boundary_face_no_addable_candidate"
    );
}

#[test]
fn exact_cover_trace_reports_boundary_face_without_addable_candidate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: vec![
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 2],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 3],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
        ],
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 3, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
    ];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);

    let (selected, trace) = search.search_with_trace();

    assert!(selected.is_none());
    assert_eq!(
        trace.dead_end,
        Some(BoundaryExactCoverDeadEnd {
            reason: "boundary_face_no_addable_candidate",
            face: Some([0, 1, 2]),
            depth: 0,
            selected_tetrahedra: Vec::new(),
            selected_roles: Vec::new(),
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.0,
            target_volume_m3: 1.0,
        })
    );
    assert_eq!(
        trace.dead_end_reason_counts,
        BTreeMap::from([("boundary_face_no_addable_candidate", 1)])
    );
    assert_eq!(
        trace.dead_end_faces_by_reason,
        BTreeMap::from([(
            "boundary_face_no_addable_candidate",
            BTreeSet::from([[0, 1, 2]])
        )])
    );
}
